# csd-proxy

WebSocket ↔ TCP bridge proxy. Sits between the encrypted miner and the
plain-text TCP Stratum backend.

```
Miner ──(WebSocket + E2E encryption)──▶ Proxy ──(TCP plain)──▶ Pool
```

## WebSocket frame format

Every application WebSocket frame is a UTF-8 string; the proxy does not use a
prefix or JSON envelope over WebSocket. Frame type is determined by connection
state: the first frame is the handshake and all later frames are encrypted data.

```
<32-byte-X25519-public-key-in-hex>  # first frame only
<AES-256-GCM-wire-frame-in-hex>     # every later frame
```

The decrypted `e2e_data` content remains the pool's line-delimited JSON-RPC,
which is forwarded unchanged on the TCP side.

## Quick start

```bash
cd proxy
npm install

# 1. Generate static keypair
node src/generate-keypair.js
# → copy PROXY_PRIVATE_KEY into .env

# 2. Start proxy
node src/server.js
```

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `PROXY_PRIVATE_KEY` | (generated if omitted) | Static X25519 private key (hex) |
| `WS_PORT` | `8443` | WebSocket listen port |
| `WS_HOST` | `0.0.0.0` | WebSocket bind address |
| `TCP_TARGET_HOST` | `csd-us-east.lproute.com` | Backend TCP host |
| `TCP_TARGET_PORT` | `8760` | Backend TCP port |

## Setup

1. Generate keypair: `node src/generate-keypair.js`
2. Set `PROXY_PRIVATE_KEY` in proxy env
3. Start proxy: `node src/server.js`
4. Run `tunel.py` with `WS_URL=ws://localhost:8443 LISTEN_HOST=127.0.0.1 LISTEN_PORT=9000`,
   then set the Pearl miner STATEAA endpoint to `localhost:9000` (see [root README](../README.md)).

The miner automatically discovers the proxy's public key during the
E2E handshake — no manual key configuration needed on the miner side.
