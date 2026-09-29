//! Hex-line X25519/AES-GCM transport matching proxy/src/crypto.js and tunel.py.
use aes_gcm::{aead::Aead, Aes256Gcm, KeyInit, Nonce};
use anyhow::{ensure, Context, Result};
use hkdf::Hkdf;
use rand_core::OsRng;
use sha2::Sha256;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::Zeroizing;

use crate::{client::PoolStream, protocol::MAX_LINE_BYTES};

struct Cipher {
    key: Zeroizing<[u8; 32]>,
    counter: u64,
}

impl Cipher {
    fn new(shared: &[u8; 32], info: &[u8]) -> Result<Self> {
        let mut key = Zeroizing::new([0; 32]);
        Hkdf::<Sha256>::new(Some(b"csd-pool-e2e-v1"), shared)
            .expand(info, key.as_mut())
            .map_err(|_| anyhow::anyhow!("HKDF failed"))?;
        Ok(Self { key, counter: 0 })
    }

    fn encrypt(&mut self, plain: &[u8]) -> Result<Vec<u8>> {
        let next = self
            .counter
            .checked_add(1)
            .context("tunnel counter exhausted")?;
        let mut nonce = [0; 12];
        nonce[..4].copy_from_slice(b"CSD!");
        nonce[4..].copy_from_slice(&next.to_be_bytes());
        let cipher = Aes256Gcm::new_from_slice(self.key.as_ref()).unwrap();
        let ciphertext = cipher
            .encrypt(Nonce::from_slice(&nonce), plain)
            .map_err(|_| anyhow::anyhow!("tunnel encryption failed"))?;
        let mut wire = vec![12];
        wire.extend_from_slice(&nonce);
        wire.extend_from_slice(&ciphertext);
        self.counter = next;
        Ok(wire)
    }

    fn decrypt(&mut self, wire: &[u8]) -> Result<Vec<u8>> {
        ensure!(wire.len() >= 29 && wire[0] == 12, "invalid tunnel frame");
        ensure!(&wire[1..5] == b"CSD!", "invalid tunnel nonce");
        let counter = u64::from_be_bytes(wire[5..13].try_into().unwrap());
        ensure!(counter > self.counter, "replayed tunnel frame");
        let cipher = Aes256Gcm::new_from_slice(self.key.as_ref()).unwrap();
        let plain = cipher
            .decrypt(Nonce::from_slice(&wire[1..13]), &wire[13..])
            .map_err(|_| anyhow::anyhow!("tunnel authentication failed"))?;
        self.counter = counter;
        Ok(plain)
    }
}

async fn line<R: AsyncBufRead + Unpin>(reader: &mut R, limit: usize) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    reader
        .take((limit + 1) as u64)
        .read_until(b'\n', &mut data)
        .await?;
    ensure!(!data.is_empty(), "tunnel connection closed");
    ensure!(
        data.len() <= limit && data.last() == Some(&b'\n'),
        "oversized or truncated tunnel line"
    );
    data.pop();
    if data.last() == Some(&b'\r') {
        data.pop();
    }
    Ok(data)
}

pub(crate) async fn connect(tcp: TcpStream) -> Result<Box<dyn PoolStream>> {
    let secret = StaticSecret::random_from_rng(OsRng);
    let mut socket = BufReader::new(tcp);
    socket
        .write_all(format!("{}\n", hex::encode(PublicKey::from(&secret).as_bytes())).as_bytes())
        .await?;
    let hello = line(&mut socket, 66)
        .await
        .context("reading tunnel public key")?;
    let peer: [u8; 32] = hex::decode(hello)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("proxy public key must be 32 bytes"))?;
    let shared = secret.diffie_hellman(&PublicKey::from(peer));
    ensure!(shared.was_contributory(), "invalid proxy public key");
    let mut tx = Cipher::new(shared.as_bytes(), b"aes-256-gcm-key-c2s")?;
    let mut rx = Cipher::new(shared.as_bytes(), b"aes-256-gcm-key-s2c")?;
    let (app, bridge) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let (net_read, mut net_write) = tokio::io::split(socket);
        let (app_read, mut app_write) = tokio::io::split(bridge);
        let send = async {
            let mut reader = BufReader::new(app_read);
            loop {
                // Buffer a complete JSON line: the pool client may write JSON and LF separately.
                let plain = line(&mut reader, MAX_LINE_BYTES).await?;
                let wire = tx.encrypt(&plain)?;
                net_write.write_all(hex::encode(wire).as_bytes()).await?;
                net_write.write_all(b"\n").await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        };
        let receive = async {
            let mut reader = BufReader::new(net_read);
            loop {
                let encoded = line(&mut reader, 2 * (MAX_LINE_BYTES + 29) + 2).await?;
                let mut plain = rx.decrypt(&hex::decode(encoded)?)?;
                if !plain.ends_with(b"\n") {
                    plain.push(b'\n');
                }
                app_write.write_all(&plain).await?;
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        };
        // A lost upstream session must also close the miner stream and force a fresh handshake.
        let result = tokio::select! { result = send => result, result = receive => result };
        if let Err(error) = result {
            tracing::debug!(%error, "トンネルセッションが終了しました");
        }
    });
    Ok(Box::new(app))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires tools/test_tunnel.py, Node dependencies and Python websockets"]
    async fn real_proxy_tunnel() {
        tokio::time::timeout(std::time::Duration::from_secs(20), async {
            let addr = std::env::var("PEARL_TEST_TUNNEL").unwrap();
            for _ in 0..2 {
                let endpoint = crate::PoolEndpoint::parse(&addr).unwrap();
                let stream = crate::client::connect_stream(&endpoint).await.unwrap();
                let mut stream = BufReader::new(stream);
                for payload in [b"{\"id\":1}".to_vec(), vec![b'a'; 200_000]] {
                    // Fragment JSON and its LF across writes; the proxy must receive one line.
                    for chunk in payload.chunks(137) {
                        stream.write_all(chunk).await.unwrap();
                    }
                    stream.write_all(b"\n").await.unwrap();
                    let echoed = line(&mut stream, MAX_LINE_BYTES).await.unwrap();
                    assert_eq!(echoed, payload);
                }
                stream.write_all(b"close\n").await.unwrap();
                let mut byte = [0];
                assert_eq!(stream.read(&mut byte).await.unwrap(), 0);
            }
            let endpoint = crate::PoolEndpoint::parse(&addr).unwrap();
            let protocol = crate::KryptexProtocol::new(
                "test-wallet".into(),
                "test-worker".into(),
                String::new(),
                crate::KryptexMode::V2,
            );
            let mut session = crate::KryptexSession::connect(&endpoint, protocol)
                .await
                .unwrap();
            match session.next_event().await.unwrap() {
                crate::PoolEvent::Job(job) => assert_eq!(job.job_id, "integration_1"),
                other => panic!("expected job, got {other:?}"),
            }
        })
        .await
        .expect("tunnel test timed out");
    }

    #[test]
    fn rejects_replay_tampering_and_wrong_direction() {
        let mut tx = Cipher::new(&[7; 32], b"aes-256-gcm-key-c2s").unwrap();
        let mut rx = Cipher::new(&[7; 32], b"aes-256-gcm-key-c2s").unwrap();
        let wire = tx.encrypt(b"hello").unwrap();
        let mut bad = wire.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert!(rx.decrypt(&bad).is_err());
        assert_eq!(rx.decrypt(&wire).unwrap(), b"hello");
        assert!(rx.decrypt(&wire).is_err());
        let mut wrong = Cipher::new(&[7; 32], b"aes-256-gcm-key-s2c").unwrap();
        assert!(wrong.decrypt(&wire).is_err());
        assert!(rx.decrypt(&[]).is_err());
    }
}
