//! Kryptex connection, failover, line framing, and share accounting.

use std::{
    collections::{HashMap, VecDeque},
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    task::{Context as TaskContext, Poll},
    time::Duration,
};

use anyhow::{bail, ensure, Context, Result};
use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::ServerName;
use serde_json::Value;
use tokio::{
    io::{split, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
        ReadBuf, ReadHalf, WriteHalf},
    net::TcpStream,
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
    time::timeout,
};
use tokio_rustls::{client::TlsStream, TlsConnector};

use crate::protocol::{
    reject_reason, reply, KryptexProtocol, PoolEndpoint, PoolJob, VerifiedShare,
    MAX_LINE_BYTES, MAX_SUBMIT_BYTES,
};
use crate::crypto::{ClientHandshake, SessionKeys, MAX_FRAME_BYTES};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);

pub trait PoolStream: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send {}
impl<T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send> PoolStream for T {}

struct EncryptedStream {
    inner: TcpStream,
    session: SessionKeys,
    encrypted_read: Vec<u8>,
    plaintext_read: VecDeque<u8>,
    encrypted_write: Vec<u8>,
    pending_plaintext_len: usize,
}

impl EncryptedStream {
    fn new(inner: TcpStream, session: SessionKeys) -> Self {
        Self {
            inner,
            session,
            encrypted_read: Vec::new(),
            plaintext_read: VecDeque::new(),
            encrypted_write: Vec::new(),
            pending_plaintext_len: 0,
        }
    }

    fn protocol_error(error: impl std::fmt::Display) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, error.to_string())
    }

    fn drain_plaintext(&mut self, output: &mut ReadBuf<'_>) {
        let count = self.plaintext_read.len().min(output.remaining());
        if count == 0 {
            return;
        }
        let (first, second) = self.plaintext_read.as_slices();
        let first_count = count.min(first.len());
        output.put_slice(&first[..first_count]);
        if first_count < count {
            output.put_slice(&second[..count - first_count]);
        }
        self.plaintext_read.drain(..count);
    }
}

impl AsyncRead for EncryptedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        output: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        loop {
            self.drain_plaintext(output);
            if output.filled().len() > 0 {
                return Poll::Ready(Ok(()));
            }

            if self.encrypted_read.len() >= 4 {
                let declared = u32::from_be_bytes(
                    self.encrypted_read[..4]
                        .try_into()
                        .expect("encrypted frame length is four bytes"),
                ) as usize;
                if declared > MAX_FRAME_BYTES + 64 {
                    return Poll::Ready(Err(Self::protocol_error(
                        "encrypted proxy frame exceeds the maximum size",
                    )));
                }
                if self.encrypted_read.len() >= declared + 4 {
                    let frame: Vec<u8> = self.encrypted_read.drain(..declared + 4).collect();
                    let plaintext = match self.session.decrypt(&frame) {
                        Ok(plaintext) => plaintext,
                        Err(error) => return Poll::Ready(Err(Self::protocol_error(error))),
                    };
                    self.plaintext_read.extend(plaintext);
                    continue;
                }
            }

            let mut chunk = [0u8; 64 * 1024];
            let mut read_buf = ReadBuf::new(&mut chunk);
            match Pin::new(&mut self.inner).poll_read(cx, &mut read_buf) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(())) if read_buf.filled().is_empty() => {
                    if self.encrypted_read.is_empty() {
                        return Poll::Ready(Ok(()));
                    }
                    return Poll::Ready(Err(Self::protocol_error(
                        "proxy closed with a truncated encrypted frame",
                    )));
                }
                Poll::Ready(Ok(())) => {
                    self.encrypted_read.extend_from_slice(read_buf.filled());
                }
            }
        }
    }
}

impl AsyncWrite for EncryptedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        input: &[u8],
    ) -> Poll<io::Result<usize>> {
        if input.is_empty() && self.encrypted_write.is_empty() {
            return Poll::Ready(Ok(0));
        }
        if self.encrypted_write.is_empty() {
            self.encrypted_write = match self.session.encrypt(input) {
                Ok(frame) => frame,
                Err(error) => return Poll::Ready(Err(Self::protocol_error(error))),
            };
            self.pending_plaintext_len = input.len();
        }
        while !self.encrypted_write.is_empty() {
            let pending = self.encrypted_write.clone();
            match Pin::new(&mut self.inner).poll_write(cx, &pending) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "proxy closed while writing encrypted frame",
                    )))
                }
                Poll::Ready(Ok(written)) => {
                    self.encrypted_write.drain(..written);
                }
            }
        }
        let written = self.pending_plaintext_len;
        self.pending_plaintext_len = 0;
        Poll::Ready(Ok(written))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        if !self.encrypted_write.is_empty() {
            match Pin::new(&mut *self).poll_write(cx, &[]) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(_)) => {}
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
            }
        }
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        match Pin::new(&mut *self).poll_flush(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Ready(Ok(())) => Pin::new(&mut self.inner).poll_shutdown(cx),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectKind {
    Stale,
    Duplicate,
    LowDifficulty,
    Format,
    InvalidProof,
    Unauthorized,
    Other,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShareStats {
    pub submitted: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub stale: u64,
    pub duplicate: u64,
    pub low_difficulty: u64,
    pub format: u64,
    pub invalid_proof: u64,
    pub unauthorized: u64,
}

impl ShareStats {
    fn record_reject(&mut self, kind: RejectKind) {
        self.rejected += 1;
        match kind {
            RejectKind::Stale => self.stale += 1,
            RejectKind::Duplicate => self.duplicate += 1,
            RejectKind::LowDifficulty => self.low_difficulty += 1,
            RejectKind::Format => self.format += 1,
            RejectKind::InvalidProof => self.invalid_proof += 1,
            RejectKind::Unauthorized => self.unauthorized += 1,
            RejectKind::Other => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitResult {
    Accepted,
    Rejected { kind: RejectKind, reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolEvent {
    Job(PoolJob),
    Reply { id: u64, accepted: bool, reason: Option<String> },
    Other,
}

enum SubmitCommand {
    Share {
        share: VerifiedShare,
        reply: oneshot::Sender<std::result::Result<SubmitResult, String>>,
    },
}

/// Handle used by the GPU worker while the network actor owns the Stratum socket.
///
/// Keeping the socket in its own task is required for stale-job handling: the actor can consume a
/// new `mining.notify` while the CUDA thread is inside the synchronous tile loop.
#[derive(Clone)]
pub struct KryptexHandle {
    submit_tx: mpsc::Sender<SubmitCommand>,
}

impl KryptexHandle {
    pub async fn submit_verified(&self, share: VerifiedShare) -> Result<SubmitResult> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.submit_tx
            .send(SubmitCommand::Share {
                share,
                reply: reply_tx,
            })
            .await
            .context("sending verified share to Kryptex actor")?;
        let result = reply_rx
            .await
            .context("Kryptex actor stopped before share result")?;
        result.map_err(|error| anyhow::anyhow!(error))
    }
}

pub struct KryptexSession {
    protocol: KryptexProtocol,
    reader: BufReader<ReadHalf<Box<dyn PoolStream>>>,
    writer: WriteHalf<Box<dyn PoolStream>>,
    next_id: u64,
    current_job: Option<PoolJob>,
    pub stats: ShareStats,
    last_keepalive: tokio::time::Instant,
}

impl KryptexSession {
    pub async fn connect(endpoint: &PoolEndpoint, protocol: KryptexProtocol) -> Result<Self> {
        let stream = connect_stream(endpoint).await?;
        let (read, write) = split(stream);
        let mut session = Self {
            protocol,
            reader: BufReader::new(read),
            writer: write,
            next_id: 1,
            current_job: None,
            stats: ShareStats::default(),
            last_keepalive: tokio::time::Instant::now(),
        };
        session.authorize().await?;
        Ok(session)
    }

    pub fn current_job(&self) -> Option<&PoolJob> {
        self.current_job.as_ref()
    }

    async fn authorize(&mut self) -> Result<()> {
        let base_id = self.next_id;
        self.next_id = self.next_id.saturating_add(2);
        for message in self.protocol.handshake(base_id) {
            self.send(&message).await?;
        }
        let authorize_id = self.protocol.authorize_id(base_id);
        loop {
            let value = self.receive().await?;
            if let Some(job) = self.protocol.parse_job(&value)? {
                if job.requires_update() {
                    bail!("update required: Kryptex advertised unsupported cert_version {:?}", job.cert_version);
                }
                self.current_job = Some(job);
                continue;
            }
            if value.get("id").and_then(Value::as_u64) == Some(authorize_id) {
                ensure!(reply(&value, authorize_id)?, "Kryptex authorization rejected: {}", reject_reason(&value));
                return Ok(());
            }
        }
    }

    pub async fn next_event(&mut self) -> Result<PoolEvent> {
        self.maybe_keepalive().await?;
        let value = self.receive().await?;
        if let Some(job) = self.protocol.parse_job(&value)? {
            if job.requires_update() {
                bail!("update required: Kryptex advertised unsupported cert_version {:?}", job.cert_version);
            }
            self.current_job = Some(job.clone());
            return Ok(PoolEvent::Job(job));
        }
        if let Some(id) = value.get("id").and_then(Value::as_u64) {
            let accepted = reply(&value, id)?;
            return Ok(PoolEvent::Reply {
                id,
                accepted,
                reason: (!accepted).then(|| reject_reason(&value)),
            });
        }
        Ok(PoolEvent::Other)
    }

    pub async fn submit_verified(&mut self, share: &VerifiedShare) -> Result<SubmitResult> {
        let current = self
            .current_job
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("cannot submit without a current pool job"))?;
        ensure!(current.job_id == share.job_id, "refusing stale share before network send");
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let message = self.protocol.submit(id, share)?;
        let encoded = serde_json::to_vec(&message).context("encoding share request")?;
        ensure!(encoded.len() + 1 <= MAX_SUBMIT_BYTES, "share request exceeds wire limit");
        self.send(&message).await?;
        self.stats.submitted += 1;
        loop {
            match self.next_event().await? {
                PoolEvent::Job(_) => {}
                PoolEvent::Reply {
                    id: response_id,
                    accepted,
                    reason,
                } if response_id == id => {
                    if accepted {
                        self.stats.accepted += 1;
                        return Ok(SubmitResult::Accepted);
                    }
                    let reason = reason.unwrap_or_else(|| "pool rejected share".to_owned());
                    let kind = classify_reject(&reason);
                    self.stats.record_reject(kind);
                    return Ok(SubmitResult::Rejected { kind, reason });
                }
                _ => {}
            }
        }
    }

    /// Moves this session into a network actor and returns a latest-job watch channel.
    ///
    /// `cancel` is raised and `generation` is advanced before every new job is published. The GPU
    /// worker checks both values between tiles, so a candidate from an older job is discarded
    /// before it reaches the submit command.
    pub fn spawn_actor(
        self,
        cancel: Arc<AtomicBool>,
        generation: Arc<AtomicU64>,
    ) -> (
        KryptexHandle,
        watch::Receiver<Option<PoolJob>>,
        JoinHandle<Result<()>>,
    ) {
        let (job_tx, job_rx) = watch::channel(self.current_job.clone());
        if self.current_job.is_some() {
            generation.store(1, Ordering::Release);
        }
        let (submit_tx, mut submit_rx) = mpsc::channel(1);
        let handle = KryptexHandle { submit_tx };
        let KryptexSession {
            protocol,
            reader,
            writer,
            next_id,
            current_job,
            stats,
            last_keepalive,
        } = self;
        let (reader_tx, mut reader_rx) = mpsc::channel(32);
        let (writer_tx, mut writer_rx) = mpsc::channel(32);
        let reader_task = tokio::spawn(async move {
            let mut reader = reader;
            loop {
                let value = receive_from(&mut reader).await.map_err(|error| format!("{error:#}"));
                if reader_tx.send(value).await.is_err() {
                    return;
                }
            }
        });
        let writer_task = tokio::spawn(async move {
            let mut writer = writer;
            while let Some(value) = writer_rx.recv().await {
                send_to(&mut writer, &value).await?;
            }
            Ok::<(), anyhow::Error>(())
        });
        let task = tokio::spawn(async move {
            let mut next_id = next_id;
            let mut current_job = current_job;
            let mut stats = stats;
            let mut last_keepalive = last_keepalive;
            let mut pending: HashMap<u64, oneshot::Sender<std::result::Result<SubmitResult, String>>> =
                HashMap::new();
            let actor_cancel = cancel;
            let actor_generation = generation;
            loop {
                tokio::select! {
                    value = reader_rx.recv() => {
                        let Some(value) = value else {
                            bail!("Kryptex reader task stopped");
                        };
                        let value = value.map_err(|error| anyhow::anyhow!(error))?;
                        if let Some(job) = protocol.parse_job(&value)? {
                            if job.requires_update() {
                                bail!("update required: Kryptex advertised unsupported cert_version {:?}", job.cert_version);
                            }
                            current_job = Some(job.clone());
                            publish_job(job, &job_tx, &actor_cancel, &actor_generation);
                            continue;
                        }
                        if let Some(id) = value.get("id").and_then(Value::as_u64) {
                            let accepted = reply(&value, id)?;
                            if let Some(reply_tx) = pending.remove(&id) {
                                if accepted {
                                    stats.accepted += 1;
                                    let _ = reply_tx.send(Ok(SubmitResult::Accepted));
                                } else {
                                    let reason = reject_reason(&value);
                                    let kind = classify_reject(&reason);
                                    stats.record_reject(kind);
                                    let _ = reply_tx.send(Ok(SubmitResult::Rejected { kind, reason }));
                                }
                            } else {
                                tracing::debug!(id, accepted, "Kryptexからの応答");
                            }
                        }
                    }
                    command = submit_rx.recv() => {
                        let Some(SubmitCommand::Share { share, reply }) = command else {
                            reader_task.abort();
                            writer_task.abort();
                            return Ok(());
                        };
                        let Some(current) = current_job.as_ref() else {
                            let _ = reply.send(Err("cannot submit without a current pool job".to_owned()));
                            continue;
                        };
                        if current.job_id != share.job_id {
                            let _ = reply.send(Err("refusing stale share before network send".to_owned()));
                            continue;
                        }
                        let id = next_id;
                        next_id = next_id.saturating_add(1);
                        let message = protocol.submit(id, &share)?;
                        let encoded = serde_json::to_vec(&message).context("encoding share request")?;
                        ensure!(encoded.len() + 1 <= MAX_SUBMIT_BYTES, "share request exceeds wire limit");
                        writer_tx.send(message).await.context("sending share to Kryptex writer")?;
                        stats.submitted += 1;
                        pending.insert(id, reply);
                    }
                    _ = tokio::time::sleep_until(last_keepalive + KEEPALIVE_INTERVAL) => {
                        let id = next_id;
                        next_id = next_id.saturating_add(1);
                        writer_tx.send(serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "method": "mining.ping",
                            "params": []
                        })).await.context("sending Kryptex keepalive")?;
                        last_keepalive = tokio::time::Instant::now();
                    }
                }
            }
        });
        (handle, job_rx, task)
    }

    async fn maybe_keepalive(&mut self) -> Result<()> {
        if self.last_keepalive.elapsed() >= KEEPALIVE_INTERVAL {
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.send(&serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "mining.ping",
                "params": []
            }))
            .await?;
            self.last_keepalive = tokio::time::Instant::now();
        }
        Ok(())
    }

    async fn send(&mut self, value: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(value).context("encoding Stratum message")?;
        ensure!(!bytes.contains(&b'\n') && !bytes.contains(&b'\r'), "Stratum message contains a newline");
        ensure!(bytes.len() < MAX_LINE_BYTES, "Stratum message is too large");
        bytes.push(b'\n');
        self.writer.write_all(&bytes).await.context("writing Stratum message")?;
        self.writer.flush().await.context("flushing Stratum message")
    }

    async fn receive(&mut self) -> Result<Value> {
        let mut line = Vec::new();
        let n = self
            .reader
            .read_until(b'\n', &mut line)
            .await
            .context("reading Stratum message")?;
        ensure!(n != 0, "pool closed the Stratum connection");
        ensure!(line.len() <= MAX_LINE_BYTES, "pool sent an oversized Stratum message");
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        ensure!(!line.is_empty(), "pool sent a blank Stratum message");
        serde_json::from_slice(&line).context("decoding Stratum JSON")
    }
}

async fn send_to(writer: &mut WriteHalf<Box<dyn PoolStream>>, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value).context("encoding Stratum message")?;
    ensure!(!bytes.contains(&b'\n') && !bytes.contains(&b'\r'), "Stratum message contains a newline");
    ensure!(bytes.len() < MAX_LINE_BYTES, "Stratum message is too large");
    bytes.push(b'\n');
    writer.write_all(&bytes).await.context("writing Stratum message")?;
    writer.flush().await.context("flushing Stratum message")
}

async fn receive_from(reader: &mut BufReader<ReadHalf<Box<dyn PoolStream>>>) -> Result<Value> {
    let mut line = Vec::new();
    let n = reader
        .read_until(b'\n', &mut line)
        .await
        .context("reading Stratum message")?;
    ensure!(n != 0, "pool closed the Stratum connection");
    ensure!(line.len() <= MAX_LINE_BYTES, "pool sent an oversized Stratum message");
    while matches!(line.last(), Some(b'\n' | b'\r')) {
        line.pop();
    }
    ensure!(!line.is_empty(), "pool sent a blank Stratum message");
    serde_json::from_slice(&line).context("decoding Stratum JSON")
}

fn publish_job(
    job: PoolJob,
    job_tx: &watch::Sender<Option<PoolJob>>,
    cancel: &AtomicBool,
    generation: &AtomicU64,
) {
    cancel.store(true, Ordering::Release);
    generation.fetch_add(1, Ordering::AcqRel);
    job_tx.send_replace(Some(job));
}

pub struct KryptexPoolSet {
    endpoints: Vec<PoolEndpoint>,
    retry_delay: Duration,
}

impl KryptexPoolSet {
    pub fn new(endpoints: Vec<PoolEndpoint>) -> Result<Self> {
        ensure!(!endpoints.is_empty() && endpoints.len() <= 3, "Kryptex supports one to three pool endpoints");
        Ok(Self {
            endpoints,
            retry_delay: Duration::from_secs(2),
        })
    }

    pub fn endpoints(&self) -> &[PoolEndpoint] {
        &self.endpoints
    }

    /// Tries each endpoint once in configured order. The caller owns the reconnect loop and can
    /// back off between calls without losing the current job's stale-share boundary.
    pub async fn connect(&self, protocol: KryptexProtocol) -> Result<KryptexSession> {
        let mut errors = VecDeque::new();
        for endpoint in &self.endpoints {
            match KryptexSession::connect(endpoint, protocol.clone()).await {
                Ok(session) => return Ok(session),
                Err(error) => errors.push_back(format!("{}: {error:#}", endpoint.address())),
            }
        }
        bail!("all Kryptex endpoints failed: {}", errors.into_iter().collect::<Vec<_>>().join("; "))
    }

    pub async fn connect_with_retry(
        &self,
        protocol: KryptexProtocol,
        cancellation: &mut tokio::sync::watch::Receiver<bool>,
    ) -> Result<KryptexSession> {
        let mut delay = self.retry_delay;
        loop {
            if *cancellation.borrow() {
                bail!("pool reconnect cancelled");
            }
            match self.connect(protocol.clone()).await {
                Ok(session) => return Ok(session),
                Err(error) => {
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                    changed = cancellation.changed() => {
                            changed.context("pool cancellation channel closed")?;
                        }
                    }
                    delay = (delay * 2).min(Duration::from_secs(60));
                    tracing_error(&error);
                }
            }
        }
    }
}

pub(crate) async fn connect_stream(endpoint: &PoolEndpoint) -> Result<Box<dyn PoolStream>> {
    let tcp = timeout(CONNECT_TIMEOUT, TcpStream::connect((endpoint.host.as_str(), endpoint.port)))
        .await
        .context("connecting to pool timed out")?
        .with_context(|| format!("connecting to {}", endpoint.address()))?;
    tcp.set_nodelay(true).context("setting pool TCP_NODELAY")?;
    if endpoint.e2e {
        return timeout(CONNECT_TIMEOUT, async {
            if endpoint.legacy_prl7 {
                connect_encrypted_proxy(tcp).await
            } else {
                crate::tunnel::connect(tcp).await
            }
        }).await.context("proxy handshake timed out")?;
    }
    if !endpoint.tls {
        return Ok(Box::new(tcp));
    }
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));
    let name = ServerName::try_from(endpoint.host.clone())
        .map_err(|_| anyhow::anyhow!("invalid TLS server name {}", endpoint.host))?;
    let stream: TlsStream<TcpStream> = connector
        .connect(name, tcp)
        .await
        .with_context(|| format!("TLS handshake with {}", endpoint.address()))?;
    Ok(Box::new(stream))
}

async fn connect_encrypted_proxy(mut tcp: TcpStream) -> Result<Box<dyn PoolStream>> {
    let handshake = ClientHandshake::random(None);
    tcp.write_all(&handshake.hello())
        .await
        .context("writing proxy client hello")?;
    tcp.flush().await.context("flushing proxy client hello")?;

    let mut prefix = [0u8; 4];
    tcp.read_exact(&mut prefix)
        .await
        .context("reading proxy server hello length")?;
    let length = u32::from_be_bytes(prefix) as usize;
    ensure!(
        length <= MAX_FRAME_BYTES + 64,
        "proxy server hello exceeds the maximum frame size"
    );
    let mut frame = vec![0u8; length + 4];
    frame[..4].copy_from_slice(&prefix);
    tcp.read_exact(&mut frame[4..])
        .await
        .context("reading proxy server hello")?;
    let session = handshake
        .finish(&frame)
        .map_err(|error| anyhow::anyhow!("proxy handshake failed: {error}"))?;
    Ok(Box::new(EncryptedStream::new(tcp, session)))
}

fn classify_reject(reason: &str) -> RejectKind {
    let text = reason.to_ascii_lowercase();
    if text.contains("stale") || text.contains("job not found") || text.contains("expired") {
        RejectKind::Stale
    } else if text.contains("duplicate") || text.contains("already submitted") {
        RejectKind::Duplicate
    } else if text.contains("low diff") || text.contains("difficulty") || text.contains("target") {
        RejectKind::LowDifficulty
    } else if text.contains("format")
        || text.contains("decode")
        || text.contains("base64")
        || text.contains("gzip")
        || text.contains("bincode")
    {
        RejectKind::Format
    } else if text.contains("invalid") || text.contains("verify") || text.contains("proof") {
        RejectKind::InvalidProof
    } else if text.contains("unauthor") || text.contains("login") || text.contains("worker") {
        RejectKind::Unauthorized
    } else {
        RejectKind::Other
    }
}

fn tracing_error(error: &anyhow::Error) {
    tracing::warn!(error = %format!("{error:#}"), "プールへの再接続に失敗しました");
}
