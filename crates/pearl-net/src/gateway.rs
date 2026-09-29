//! Official Pearl gateway JSON-RPC client for local/testnet solo operation.
//!
//! The gateway path handles node `getblocktemplate` and `submitblock` only. It is deliberately
//! separate from Kryptex Stratum and never submits a pool share.

use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rustls::{ClientConfig, RootCertStore};
use rustls_pki_types::ServerName;
use serde_json::{json, Value};
use tokio::io::{split, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

use crate::client::{connect_stream, PoolStream};
use crate::protocol::PoolEndpoint;

#[derive(Clone)]
pub struct GatewayRpcClient {
    endpoint: PoolEndpoint,
    username: String,
    password: String,
}

impl GatewayRpcClient {
    pub fn new(url: &str, username: String, password: String) -> Result<Self> {
        let (tls, rest, default_port) = if let Some(value) = url.strip_prefix("https://") {
            (true, value, 44107)
        } else if let Some(value) = url.strip_prefix("http://") {
            (false, value, 44107)
        } else {
            bail!("gateway URL must start with http:// or https://")
        };
        let (host, port) = match rest.rsplit_once(':') {
            Some((host, port)) if !port.contains('/') => (host, port.parse::<u16>().context("invalid gateway port")?),
            _ => (rest, default_port),
        };
        let host = host.trim_end_matches('/');
        ensure!(!host.is_empty(), "gateway host is empty");
        Ok(Self {
            endpoint: PoolEndpoint {
                host: host.to_owned(),
                port,
                tls,
                e2e: false,
                legacy_prl7: false,
            },
            username,
            password,
        })
    }

    pub async fn get_block_template(&self) -> Result<Value> {
        self.call("getblocktemplate", json!([])).await
    }

    pub async fn submit_block(&self, serialized_block_hex: &str) -> Result<Value> {
        ensure!(!serialized_block_hex.is_empty(), "serialized block is empty");
        self.call("submitblock", json!([serialized_block_hex])).await
    }

    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let stream = if self.endpoint.tls {
            connect_tls(&self.endpoint).await?
        } else {
            connect_stream(&self.endpoint).await?
        };
        let (mut reader, mut writer) = split(stream);
        let body = serde_json::to_vec(&json!({
            "jsonrpc": "1.0",
            "id": "pearl-miner",
            "method": method,
            "params": params,
        }))?;
        let auth = STANDARD.encode(format!("{}:{}", self.username, self.password));
        let request = format!(
            "POST / HTTP/1.1\r\nHost: {}\r\nAuthorization: Basic {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.endpoint.host,
            auth,
            body.len()
        );
        writer.write_all(request.as_bytes()).await.context("writing gateway headers")?;
        writer.write_all(&body).await.context("writing gateway body")?;
        writer.shutdown().await.context("closing gateway request")?;
        let mut response = Vec::new();
        reader.read_to_end(&mut response).await.context("reading gateway response")?;
        let split_at = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or_else(|| anyhow::anyhow!("gateway response has no HTTP header terminator"))?;
        let header = std::str::from_utf8(&response[..split_at]).context("gateway headers are not UTF-8")?;
        ensure!(header.starts_with("HTTP/1.1 200") || header.starts_with("HTTP/1.0 200"), "gateway returned HTTP error: {header}");
        let value: Value = serde_json::from_slice(&response[split_at + 4..]).context("decoding gateway JSON")?;
        if let Some(error) = value.get("error") {
            if !error.is_null() {
                bail!("gateway RPC {method} failed: {error}");
            }
        }
        Ok(value.get("result").cloned().unwrap_or(Value::Null))
    }
}

async fn connect_tls(endpoint: &PoolEndpoint) -> Result<Box<dyn PoolStream>> {
    let tcp = TcpStream::connect((endpoint.host.as_str(), endpoint.port))
        .await
        .with_context(|| format!("connecting to gateway {}", endpoint.address()))?;
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));
    let name = ServerName::try_from(endpoint.host.clone())
        .map_err(|_| anyhow::anyhow!("invalid gateway TLS server name"))?;
    let stream = connector.connect(name, tcp).await.context("gateway TLS handshake")?;
    Ok(Box::new(stream))
}
