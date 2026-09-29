//! Pearl network transport.
//!
//! M6 provides Kryptex Stratum and local gateway RPC. M7 adds the documented X25519/HKDF/AES-GCM
//! channel used between the miner and the local proxy.

pub mod client;
pub mod crypto;
pub mod gateway;
pub mod protocol;

pub use client::{
    KryptexHandle, KryptexPoolSet, KryptexSession, PoolEvent, RejectKind, ShareStats, SubmitResult,
};
pub use crypto::{
    ClientHandshake, CryptoError, Role, ServerHandshake, SessionKeys, MAX_COUNTER,
    MAX_FRAME_BYTES,
};
pub use gateway::GatewayRpcClient;
pub use protocol::{
    KryptexMode, KryptexProtocol, PoolEndpoint, PoolJob, PoolProtocol, VerifiedShare,
    SUPPORTED_CERT_VERSION,
};

mod tunnel;
