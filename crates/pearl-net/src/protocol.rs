//! Pure Kryptex Stratum v1/v2 messages and the local share-verification gate.
//!
//! Kryptex's Pearl dialect is intentionally kept separate from generic Stratum assumptions:
//! `mining.notify.params` is an object, the pool-supplied target is used verbatim, and a share is
//! submitted as `plain_proof`; V1 carries base64(bincode(PlainProof)), while V2 carries
//! base64(gzip-level-1(bincode(PlainProof) without the trailing `moe=None` byte)).

use std::io::Write;

use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use flate2::{Compression, GzBuilder};
use pearl_ref::{Problem, TileResult};
use primitive_types::U256;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_SUBMIT_BYTES: usize = 2 * 1024 * 1024;
pub const SUPPORTED_CERT_VERSION: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolEndpoint {
    pub host: String,
    pub port: u16,
    pub tls: bool,
    /// Bare `host:port` endpoints use the hex-line protocol through tunel.py.
    pub e2e: bool,
    #[serde(default)]
    pub legacy_prl7: bool,
}

impl PoolEndpoint {
    /// Parse direct Stratum, a local tunnel, or an explicit prl7:// endpoint.
    ///
    /// The environment-only miner configuration uses the bare form, for example
    /// `localhost:3000`, to select the encrypted miner-to-proxy transport.
    pub fn parse(url: &str) -> Result<Self> {
        let legacy_prl7 = url.starts_with("prl7://");
        let (tls, e2e, rest) = if let Some(value) = url.strip_prefix("stratum+ssl://") {
            (true, false, value)
        } else if let Some(value) = url.strip_prefix("stratum+tcp://") {
            (false, false, value)
        } else if let Some(value) = url.strip_prefix("prl7://") {
            (false, true, value)
        } else if let Some(value) = url.strip_prefix("tunnel://") {
            (false, true, value)
        } else {
            ensure!(!url.contains("://"), "unsupported endpoint scheme; use tunnel://host:port through tunel.py");
            (false, true, url)
        };
        let (host, port) = rest
            .rsplit_once(':')
            .ok_or_else(|| anyhow::anyhow!("pool endpoint must be host:port: {url}"))?;
        ensure!(!host.is_empty(), "pool endpoint host is empty");
        let port = port.parse::<u16>().context("pool endpoint port is invalid")?;
        ensure!(port != 0, "pool endpoint port is zero");
        Ok(Self {
            host: host.to_owned(),
            port,
            tls,
            e2e,
            legacy_prl7,
        })
    }

    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KryptexMode {
    /// Captured Kryptex session: array authorize and plain bincode proof.
    V1,
    /// Kryptex v2 session: object authorize and gzip-compressed legacy proof wire format.
    V2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolJob {
    pub job_id: String,
    /// Official 76-byte serialized `IncompleteBlockHeader`.
    pub header: [u8; 76],
    /// Big-endian target exactly as sent by Kryptex.
    pub target: U256,
    pub target_be: [u8; 32],
    pub height: Option<u64>,
    pub difficulty: Option<u64>,
    pub cert_version: Option<u32>,
}

impl PoolJob {
    pub fn requires_update(&self) -> bool {
        self.cert_version != Some(SUPPORTED_CERT_VERSION)
    }
}

#[derive(Clone)]
pub struct KryptexProtocol {
    pub wallet: String,
    pub worker: String,
    pub password: String,
    pub agent: String,
    pub mode: KryptexMode,
}

impl std::fmt::Debug for KryptexProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KryptexProtocol")
            .field("wallet", &self.wallet)
            .field("worker", &self.worker)
            .field("password", &"<redacted>")
            .field("agent", &self.agent)
            .field("mode", &self.mode)
            .finish()
    }
}

impl KryptexProtocol {
    pub fn new(wallet: String, worker: String, password: String, mode: KryptexMode) -> Self {
        Self {
            wallet,
            worker,
            password,
            agent: format!("pearl-miner/{}", env!("CARGO_PKG_VERSION")),
            mode,
        }
    }

    pub fn authorize_id(&self, base_id: u64) -> u64 {
        match self.mode {
            KryptexMode::V1 => base_id + 1,
            KryptexMode::V2 => base_id,
        }
    }

    /// Builds the exact request sequence. Kryptex v1 may leave subscribe unanswered; the client
    /// therefore waits only for the authorize response while still accepting early jobs.
    pub fn handshake(&self, base_id: u64) -> Vec<Value> {
        match self.mode {
            KryptexMode::V1 => vec![
                request(
                    Some(base_id),
                    "mining.subscribe",
                    json!([self.agent]),
                ),
                request(
                    Some(base_id + 1),
                    "mining.authorize",
                    json!([format!("{}.{}", self.wallet, self.worker), self.password]),
                ),
            ],
            KryptexMode::V2 => vec![request(
                Some(base_id),
                "mining.authorize",
                json!({
                    "wallet": format!("{}.{}", self.wallet, self.worker),
                    "agent": self.agent,
                    "password": self.password,
                    "type": "v2"
                }),
            )],
        }
    }

    pub fn submit(&self, id: u64, share: &VerifiedShare) -> Result<Value> {
        let encoded = match self.mode {
            KryptexMode::V1 => STANDARD.encode(&share.proof_bytes),
            KryptexMode::V2 => {
                // Kryptex's V2 adapter consumes the bincode None tag for `moe` before gzip.
                // Keep this wire transformation isolated from the consensus proof bytes.
                ensure!(
                    share.proof_bytes.last() == Some(&0),
                    "Kryptex V2 proof is missing the trailing moe=None tag"
                );
                let legacy = &share.proof_bytes[..share.proof_bytes.len() - 1];
                let mut encoder = GzBuilder::new()
                    .mtime(0)
                    .write(Vec::new(), Compression::new(1));
                encoder
                    .write_all(legacy)
                    .context("gzip-encoding Kryptex V2 share")?;
                STANDARD.encode(encoder.finish().context("finishing Kryptex V2 share")?)
            }
        };
        let params = match self.mode {
            KryptexMode::V1 => json!({
                "worker": format!("{}.{}", self.wallet, self.worker),
                "job_id": share.job_id,
                "plain_proof": encoded,
            }),
            KryptexMode::V2 => json!({
                "hs": share.hashrate,
                "job_id": share.job_id,
                "plain_proof": encoded,
            }),
        };
        let message = request(Some(id), "mining.submit", params);
        let bytes = serde_json::to_vec(&message).context("encoding Kryptex share")?;
        ensure!(bytes.len() + 1 <= MAX_SUBMIT_BYTES, "encoded share exceeds wire limit");
        Ok(message)
    }

    pub fn parse_job(&self, value: &Value) -> Result<Option<PoolJob>> {
        if value.get("method").and_then(Value::as_str) != Some("mining.notify") {
            return Ok(None);
        }
        let params = value
            .get("params")
            .ok_or_else(|| anyhow::anyhow!("Kryptex notify has no params"))?;
        let job_id = params
            .get("job_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Kryptex notify has no job_id"))?;
        ensure!(!job_id.is_empty() && job_id.len() <= 128, "invalid Kryptex job_id");
        let header_hex = params
            .get("header")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Kryptex notify has no header"))?;
        let header: [u8; 76] = hex::decode(header_hex)
            .context("Kryptex header is not hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("Kryptex header is not 76 bytes"))?;
        let target_hex = params
            .get("target")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Kryptex notify has no target"))?
            .strip_prefix("0x")
            .unwrap_or_else(|| params["target"].as_str().unwrap());
        ensure!(!target_hex.is_empty() && target_hex.len() <= 64, "invalid Kryptex target");
        let target_be_vec = hex::decode(format!("{target_hex:0>64}"))
            .context("Kryptex target is not hex")?;
        let target_be: [u8; 32] = target_be_vec
            .try_into()
            .map_err(|_| anyhow::anyhow!("Kryptex target is not 32 bytes"))?;
        let target = U256::from_big_endian(&target_be);
        ensure!(!target.is_zero(), "Kryptex target is zero");
        let difficulty = params
            .get("diff")
            .and_then(Value::as_u64)
            .or_else(|| job_id.rsplit('_').next().and_then(|value| value.parse().ok()));
        let cert_version = params
            .get("cert_version")
            .and_then(Value::as_u64)
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX));
        Ok(Some(PoolJob {
            job_id: job_id.to_owned(),
            header,
            target,
            target_be,
            height: params.get("height").and_then(Value::as_u64),
            difficulty,
            cert_version,
        }))
    }
}

pub trait PoolProtocol: Send + Sync {
    fn name(&self) -> &'static str;
    fn handshake(&self, base_id: u64) -> Vec<Value>;
    fn authorize_id(&self, base_id: u64) -> u64;
    fn parse_job(&self, value: &Value) -> Result<Option<PoolJob>>;
    fn submit(&self, id: u64, share: &VerifiedShare) -> Result<Value>;
}

impl PoolProtocol for KryptexProtocol {
    fn name(&self) -> &'static str {
        "kryptex"
    }

    fn handshake(&self, base_id: u64) -> Vec<Value> {
        KryptexProtocol::handshake(self, base_id)
    }

    fn authorize_id(&self, base_id: u64) -> u64 {
        KryptexProtocol::authorize_id(self, base_id)
    }

    fn parse_job(&self, value: &Value) -> Result<Option<PoolJob>> {
        KryptexProtocol::parse_job(self, value)
    }

    fn submit(&self, id: u64, share: &VerifiedShare) -> Result<Value> {
        KryptexProtocol::submit(self, id, share)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedShare {
    pub job_id: String,
    pub proof_bytes: Vec<u8>,
    pub digest: [u8; 32],
    /// Integer work rate reported in Kryptex V2's `hs` submit field.
    pub hashrate: u64,
}

impl VerifiedShare {
    /// The only constructor for a share accepted by the network layer. It runs the official CPU
    /// verifier before serializing the proof; callers cannot submit raw GPU output directly.
    pub fn from_verified_tile(problem: &Problem, job: &PoolJob, tile: &TileResult) -> Result<Self> {
        Self::from_verified_tile_with_hashrate(problem, job, tile, 0)
    }

    pub fn from_verified_tile_with_hashrate(
        problem: &Problem,
        job: &PoolJob,
        tile: &TileResult,
        hashrate: u64,
    ) -> Result<Self> {
        ensure!(problem.header.to_bytes() == job.header, "share header does not match pool job");
        ensure!(!job.requires_update(), "pool advertised an unsupported certificate version");
        let proof = problem.verified_proof_against_target(tile, job.target)?;
        let proof_bytes = bincode::serialize(&proof).context("serializing verified PlainProof")?;
        Ok(Self {
            job_id: job.job_id.clone(),
            proof_bytes,
            digest: tile.digest,
            hashrate,
        })
    }
}

fn request(id: Option<u64>, method: &str, params: Value) -> Value {
    let mut value = serde_json::Map::new();
    value.insert("jsonrpc".to_owned(), Value::String("2.0".to_owned()));
    value.insert(
        "id".to_owned(),
        id.map_or(Value::Null, |id| Value::Number(id.into())),
    );
    value.insert("method".to_owned(), Value::String(method.to_owned()));
    value.insert("params".to_owned(), params);
    Value::Object(value)
}

pub fn reply(value: &Value, expected_id: u64) -> Result<bool> {
    ensure!(value.get("id").and_then(Value::as_u64) == Some(expected_id), "unexpected response id");
    if value.get("error").is_some_and(|error| !error.is_null()) {
        return Ok(false);
    }
    Ok(!matches!(value.get("result"), None | Some(Value::Null) | Some(Value::Bool(false))))
}

pub fn reject_reason(value: &Value) -> String {
    match value.get("error") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(map)) => map
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("pool rejected share")
            .to_owned(),
        Some(Value::Array(values)) => values
            .iter()
            .find_map(Value::as_str)
            .unwrap_or("pool rejected share")
            .to_owned(),
        _ => "pool rejected share".to_owned(),
    }
}
