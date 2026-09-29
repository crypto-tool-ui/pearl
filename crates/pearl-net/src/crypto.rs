//! M7 miner-to-proxy channel.
//!
//! The wire contract is deliberately small and is mirrored by `tools/proxy/crypto/index.js`.
//! This module owns only the authenticated channel; Stratum payloads remain opaque bytes. All
//! secret material is held in types that zeroize on drop, and frame parsing is fail-closed.

use aes_gcm::{
    aead::{Aead, Payload},
    Aes256Gcm, KeyInit, Nonce,
};
use hkdf::Hkdf;
use rand_core::{OsRng, RngCore};
use sha2::Sha256;
use thiserror::Error;
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

pub const PROTOCOL_VERSION: u8 = 1;
pub const MAX_FRAME_BYTES: usize = 1_048_576;
pub const MAX_COUNTER: u64 = u64::MAX - 1;
pub const CLIENT_TO_PROXY: u8 = 1;
pub const PROXY_TO_CLIENT: u8 = 2;

const MAGIC: &[u8; 4] = b"PRL7";
const CLIENT_HELLO: u8 = 1;
const SERVER_HELLO: u8 = 2;
const DATA_FRAME: u8 = 16;
const HKDF_SALT: &[u8] = b"pearl-e2e-v1-salt";
const HKDF_CLIENT_INFO: &[u8] = b"pearl-e2e-v1 client-to-proxy";
const HKDF_PROXY_INFO: &[u8] = b"pearl-e2e-v1 proxy-to-client";
const AAD_PREFIX: &[u8] = b"pearl-e2e-v1 aad";

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("frame is truncated")]
    Truncated,
    #[error("frame length is invalid")]
    InvalidLength,
    #[error("frame exceeds the 1048576 byte limit")]
    FrameTooLarge,
    #[error("unknown protocol magic or version")]
    InvalidHeader,
    #[error("unexpected handshake message type")]
    UnexpectedHandshake,
    #[error("handshake session id does not match")]
    SessionMismatch,
    #[error("server identity key does not match the configured pin")]
    PinnedKeyMismatch,
    #[error("direction byte is invalid")]
    InvalidDirection,
    #[error("replayed or out-of-order frame: expected counter {expected}, got {actual}")]
    CounterMismatch { expected: u64, actual: u64 },
    #[error("counter limit reached; a new X25519 handshake is required")]
    RekeyRequired,
    #[error("AES-GCM authentication failed")]
    AuthenticationFailed,
    #[error("HKDF expansion failed")]
    KdfFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Client,
    Proxy,
}

pub struct ClientHandshake {
    session_id: [u8; 16],
    client_secret: StaticSecret,
    client_public: [u8; 32],
    pinned_server_static: Option<[u8; 32]>,
}

impl ClientHandshake {
    pub fn random(pinned_server_static: Option<[u8; 32]>) -> Self {
        Self::new(random_session_id(), StaticSecret::random_from_rng(OsRng), pinned_server_static)
    }

    pub fn new(
        session_id: [u8; 16],
        client_secret: StaticSecret,
        pinned_server_static: Option<[u8; 32]>,
    ) -> Self {
        let client_public = PublicKey::from(&client_secret).to_bytes();
        Self {
            session_id,
            client_secret,
            client_public,
            pinned_server_static,
        }
    }

    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    pub fn client_public_key(&self) -> [u8; 32] {
        self.client_public
    }

    pub fn hello(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(4 + 1 + 1 + 16 + 32);
        body.extend_from_slice(MAGIC);
        body.extend_from_slice(&[PROTOCOL_VERSION, CLIENT_HELLO]);
        body.extend_from_slice(&self.session_id);
        body.extend_from_slice(&self.client_public);
        prefix_frame(&body)
    }

    pub fn finish(self, server_hello: &[u8]) -> Result<SessionKeys, CryptoError> {
        let body = parse_prefixed_frame(server_hello)?;
        if body.len() != 4 + 1 + 1 + 16 + 32 + 32 {
            return Err(CryptoError::InvalidLength);
        }
        check_header(body, SERVER_HELLO)?;
        if body[6..22] != self.session_id {
            return Err(CryptoError::SessionMismatch);
        }
        let mut server_ephemeral = [0u8; 32];
        server_ephemeral.copy_from_slice(&body[22..54]);
        let mut server_static = [0u8; 32];
        server_static.copy_from_slice(&body[54..86]);
        if self.pinned_server_static.is_some_and(|pin| pin != server_static) {
            return Err(CryptoError::PinnedKeyMismatch);
        }

        let ee = self
            .client_secret
            .diffie_hellman(&PublicKey::from(server_ephemeral));
        let es = self
            .client_secret
            .diffie_hellman(&PublicKey::from(server_static));
        Ok(SessionKeys::derive(
            Role::Client,
            self.session_id,
            ee.as_bytes(),
            es.as_bytes(),
        )?)
    }
}

pub struct ServerHandshake;

impl ServerHandshake {
    pub fn accept(
        client_hello: &[u8],
        server_identity_secret: StaticSecret,
        server_ephemeral_secret: StaticSecret,
    ) -> Result<(Vec<u8>, SessionKeys), CryptoError> {
        let body = parse_prefixed_frame(client_hello)?;
        if body.len() != 4 + 1 + 1 + 16 + 32 {
            return Err(CryptoError::InvalidLength);
        }
        check_header(body, CLIENT_HELLO)?;
        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&body[6..22]);
        let mut client_public = [0u8; 32];
        client_public.copy_from_slice(&body[22..54]);
        let client_public = PublicKey::from(client_public);
        let server_ephemeral_public = PublicKey::from(&server_ephemeral_secret).to_bytes();
        let server_static_public = PublicKey::from(&server_identity_secret).to_bytes();

        let ee = server_ephemeral_secret.diffie_hellman(&client_public);
        let es = server_identity_secret.diffie_hellman(&client_public);
        let keys = SessionKeys::derive(
            Role::Proxy,
            session_id,
            ee.as_bytes(),
            es.as_bytes(),
        )?;

        let mut response = Vec::with_capacity(4 + 1 + 1 + 16 + 32 + 32);
        response.extend_from_slice(MAGIC);
        response.extend_from_slice(&[PROTOCOL_VERSION, SERVER_HELLO]);
        response.extend_from_slice(&session_id);
        response.extend_from_slice(&server_ephemeral_public);
        response.extend_from_slice(&server_static_public);
        Ok((prefix_frame(&response), keys))
    }
}

#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SessionKeys {
    #[zeroize(skip)]
    session_id: [u8; 16],
    send_key: [u8; 32],
    recv_key: [u8; 32],
    #[zeroize(skip)]
    role: Role,
    #[zeroize(skip)]
    send_counter: u64,
    #[zeroize(skip)]
    recv_counter: u64,
}

impl SessionKeys {
    fn derive(
        role: Role,
        session_id: [u8; 16],
        ephemeral_shared: &[u8; 32],
        identity_shared: &[u8; 32],
    ) -> Result<Self, CryptoError> {
        let mut ikm = Zeroizing::new(Vec::with_capacity(64));
        ikm.extend_from_slice(ephemeral_shared);
        ikm.extend_from_slice(identity_shared);
        let hk = Hkdf::<Sha256>::new(Some(HKDF_SALT), &ikm);
        let mut client_key = [0u8; 32];
        let mut proxy_key = [0u8; 32];
        let mut client_info = Vec::with_capacity(HKDF_CLIENT_INFO.len() + 16);
        client_info.extend_from_slice(HKDF_CLIENT_INFO);
        client_info.extend_from_slice(&session_id);
        let mut proxy_info = Vec::with_capacity(HKDF_PROXY_INFO.len() + 16);
        proxy_info.extend_from_slice(HKDF_PROXY_INFO);
        proxy_info.extend_from_slice(&session_id);
        hk.expand(&client_info, &mut client_key)
            .map_err(|_| CryptoError::KdfFailed)?;
        hk.expand(&proxy_info, &mut proxy_key)
            .map_err(|_| CryptoError::KdfFailed)?;
        let (send_key, recv_key) = match role {
            Role::Client => (client_key, proxy_key),
            Role::Proxy => (proxy_key, client_key),
        };
        Ok(Self {
            session_id,
            send_key,
            recv_key,
            role,
            send_counter: 0,
            recv_counter: 0,
        })
    }

    pub fn session_id(&self) -> [u8; 16] {
        self.session_id
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if plaintext.len() > MAX_FRAME_BYTES {
            return Err(CryptoError::FrameTooLarge);
        }
        let counter = self.send_counter;
        if counter >= MAX_COUNTER {
            return Err(CryptoError::RekeyRequired);
        }
        let direction = match self.role {
            Role::Client => CLIENT_TO_PROXY,
            Role::Proxy => PROXY_TO_CLIENT,
        };
        let aad = associated_data(&self.session_id, direction, counter);
        let nonce_bytes = nonce(direction, counter);
        let cipher = Aes256Gcm::new_from_slice(&self.send_key)
            .map_err(|_| CryptoError::AuthenticationFailed)?;
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce_bytes),
                Payload {
                    msg: plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| CryptoError::AuthenticationFailed)?;
        let mut body = Vec::with_capacity(4 + 1 + 1 + 1 + 8 + ciphertext.len());
        body.extend_from_slice(MAGIC);
        body.extend_from_slice(&[PROTOCOL_VERSION, DATA_FRAME, direction]);
        body.extend_from_slice(&counter.to_be_bytes());
        body.extend_from_slice(&ciphertext);
        self.send_counter = self
            .send_counter
            .checked_add(1)
            .ok_or(CryptoError::RekeyRequired)?;
        Ok(prefix_frame(&body))
    }

    pub fn decrypt(&mut self, frame: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let body = parse_prefixed_frame(frame)?;
        if body.len() < 4 + 1 + 1 + 1 + 8 + 16 {
            return Err(CryptoError::Truncated);
        }
        check_header(body, DATA_FRAME)?;
        let expected_direction = match self.role {
            Role::Client => PROXY_TO_CLIENT,
            Role::Proxy => CLIENT_TO_PROXY,
        };
        if body[6] != expected_direction {
            return Err(CryptoError::InvalidDirection);
        }
        let mut counter_bytes = [0u8; 8];
        counter_bytes.copy_from_slice(&body[7..15]);
        let counter = u64::from_be_bytes(counter_bytes);
        if counter >= MAX_COUNTER {
            return Err(CryptoError::RekeyRequired);
        }
        if counter != self.recv_counter {
            return Err(CryptoError::CounterMismatch {
                expected: self.recv_counter,
                actual: counter,
            });
        }
        let aad = associated_data(&self.session_id, expected_direction, counter);
        let nonce_bytes = nonce(expected_direction, counter);
        let cipher = Aes256Gcm::new_from_slice(&self.recv_key)
            .map_err(|_| CryptoError::AuthenticationFailed)?;
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(&nonce_bytes),
                Payload {
                    msg: &body[15..],
                    aad: &aad,
                },
            )
            .map_err(|_| CryptoError::AuthenticationFailed)?;
        self.recv_counter = self
            .recv_counter
            .checked_add(1)
            .ok_or(CryptoError::RekeyRequired)?;
        Ok(plaintext)
    }
}

fn random_session_id() -> [u8; 16] {
    let mut session_id = [0u8; 16];
    OsRng.fill_bytes(&mut session_id);
    session_id
}

fn prefix_frame(body: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(4 + body.len());
    result.extend_from_slice(&(body.len() as u32).to_be_bytes());
    result.extend_from_slice(body);
    result
}

fn parse_prefixed_frame(frame: &[u8]) -> Result<&[u8], CryptoError> {
    if frame.len() < 4 {
        return Err(CryptoError::Truncated);
    }
    let declared = u32::from_be_bytes(frame[..4].try_into().expect("length checked")) as usize;
    if declared > MAX_FRAME_BYTES + 64 {
        return Err(CryptoError::FrameTooLarge);
    }
    if declared != frame.len() - 4 {
        return Err(CryptoError::InvalidLength);
    }
    Ok(&frame[4..])
}

fn check_header(body: &[u8], message_type: u8) -> Result<(), CryptoError> {
    if body.len() < 6 || &body[..4] != MAGIC || body[4] != PROTOCOL_VERSION {
        return Err(CryptoError::InvalidHeader);
    }
    if body[5] != message_type {
        return Err(CryptoError::UnexpectedHandshake);
    }
    Ok(())
}

fn nonce(direction: u8, counter: u64) -> [u8; 12] {
    let mut result = [0u8; 12];
    result[0] = direction;
    result[1..9].copy_from_slice(&counter.to_be_bytes());
    result
}

fn associated_data(session_id: &[u8; 16], direction: u8, counter: u64) -> Vec<u8> {
    let mut aad = Vec::with_capacity(AAD_PREFIX.len() + 16 + 1 + 8);
    aad.extend_from_slice(AAD_PREFIX);
    aad.extend_from_slice(session_id);
    aad.push(direction);
    aad.extend_from_slice(&counter.to_be_bytes());
    aad
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Vector {
        session_id: String,
        client_ephemeral_private: String,
        server_ephemeral_private: String,
        server_identity_private: String,
        client_hello: String,
        server_hello: String,
        plaintext: String,
        client_to_proxy_frame: String,
    }

    fn bytes(value: &str) -> Vec<u8> {
        hex::decode(value).expect("valid crypto vector hex")
    }

    fn array<const N: usize>(value: &str) -> [u8; N] {
        bytes(value).try_into().expect("crypto vector has the expected length")
    }

    fn secret(value: &str) -> StaticSecret {
        StaticSecret::from(array(value))
    }

    fn vector() -> Vector {
        serde_json::from_str(include_str!("../../../tests/crypto-vectors/v1.json"))
            .expect("valid crypto vector JSON")
    }

    #[test]
    fn fixed_node_vector_matches() {
        let vector = vector();
        let client = ClientHandshake::new(
            array(&vector.session_id),
            secret(&vector.client_ephemeral_private),
            None,
        );
        assert_eq!(client.hello(), bytes(&vector.client_hello));
        let client_hello = client.hello();
        let (server_hello, mut proxy) = ServerHandshake::accept(
            &client_hello,
            secret(&vector.server_identity_private),
            secret(&vector.server_ephemeral_private),
        )
        .expect("server handshake");
        assert_eq!(server_hello, bytes(&vector.server_hello));
        let mut client = client.finish(&server_hello).expect("client handshake");
        let frame = client.encrypt(&bytes(&vector.plaintext)).expect("encrypt vector");
        assert_eq!(frame, bytes(&vector.client_to_proxy_frame));
        assert_eq!(proxy.decrypt(&frame).expect("decrypt vector"), bytes(&vector.plaintext));
    }

    #[test]
    fn tamper_replay_pin_and_truncation_fail_closed() {
        let vector = vector();
        let client = ClientHandshake::new(
            array(&vector.session_id),
            secret(&vector.client_ephemeral_private),
            None,
        );
        let client_hello = client.hello();
        let (server_hello, mut proxy) = ServerHandshake::accept(
            &client_hello,
            secret(&vector.server_identity_private),
            secret(&vector.server_ephemeral_private),
        )
        .expect("server handshake");
        let mut client = client.finish(&server_hello).expect("client handshake");
        let frame = client.encrypt(&bytes(&vector.plaintext)).expect("encrypt vector");

        let mut tampered = frame.clone();
        *tampered.last_mut().expect("tag byte") ^= 1;
        assert!(matches!(proxy.decrypt(&tampered), Err(CryptoError::AuthenticationFailed)));
        assert!(matches!(proxy.decrypt(&frame[..frame.len() - 1]), Err(CryptoError::InvalidLength)));

        let (_, mut replay_proxy) = ServerHandshake::accept(
            &client_hello,
            secret(&vector.server_identity_private),
            secret(&vector.server_ephemeral_private),
        )
        .expect("second server handshake");
        replay_proxy.decrypt(&frame).expect("first frame");
        assert!(matches!(replay_proxy.decrypt(&frame), Err(CryptoError::CounterMismatch { .. })));

        let (_, mut wrong_proxy) = ServerHandshake::accept(
            &client_hello,
            StaticSecret::from([0x77; 32]),
            StaticSecret::from([0x88; 32]),
        )
        .expect("wrong-key handshake");
        assert!(matches!(wrong_proxy.decrypt(&frame), Err(CryptoError::AuthenticationFailed)));

        let pinned = ClientHandshake::new(
            array(&vector.session_id),
            secret(&vector.client_ephemeral_private),
            Some([0x66; 32]),
        );
        assert!(matches!(pinned.finish(&server_hello), Err(CryptoError::PinnedKeyMismatch)));

        client.send_counter = MAX_COUNTER;
        assert!(matches!(client.encrypt(b"rollover"), Err(CryptoError::RekeyRequired)));
    }
}
