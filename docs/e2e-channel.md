# Legacy PRL7 miner ↔ proxy encrypted channel

This M7 channel is a new, legitimate protocol implemented independently in Rust and Node.js. The
Node implementation is [`tools/proxy/crypto/index.js`](../tools/proxy/crypto/index.js). It has no
persistence, shell execution, wallet handling, or process hooks.

The channel protects opaque payload bytes. The payload is unchanged Stratum JSON or the gateway
payload selected by the caller. `tools/proxy/server.js` forwards decrypted payload bytes to the
configured Kryptex TCP/TLS endpoint and encrypts newline-delimited pool responses back to the miner.
In miner configuration, `prl7://host:port` selects this PRL7 transport; explicit
`stratum+ssl://` or `stratum+tcp://` means a direct pool transport.

## Cryptographic parameters

- Key agreement: X25519, RFC 7748. Public keys are 32-byte raw Montgomery-u values. Private keys
  are 32-byte raw scalar values; Node wraps them in PKCS#8 DER only while calling its built-in
  `crypto` implementation.
- Key derivation: HKDF-SHA256 with salt `pearl-e2e-v1-salt`.
- The input key material is `X25519(client_ephemeral, proxy_ephemeral)` followed by
  `X25519(client_ephemeral, proxy_static)`.
- Two 32-byte keys are expanded with the session ID appended to the info string:
  `pearl-e2e-v1 client-to-proxy || session_id` and
  `pearl-e2e-v1 proxy-to-client || session_id`.
- Encryption: AES-256-GCM with a 16-byte authentication tag.
- Secrets are zeroized on the Rust side with `zeroize`; Node keeps secrets only in local
  `Buffer`/`KeyObject` values for the active session.

## Handshake frames

Every frame starts with a four-byte unsigned big-endian length prefix. The prefix counts the body
only and is not encrypted. A body larger than 1 MiB plus the fixed protocol overhead is rejected.

The common header is `magic[4] || version[1] || type[1]` where `magic` is ASCII `PRL7` and
`version` is `1`.

The client sends type `1`:

```text
u32_be body_len
"PRL7"[4] | version[1] | type=1[1] | session_id[16] | client_ephemeral_public[32]
```

The proxy sends type `2`:

```text
u32_be body_len
"PRL7"[4] | version[1] | type=2[1] | session_id[16]
  | proxy_ephemeral_public[32] | proxy_static_public[32]
```

The client checks the session ID and, when configured, compares `proxy_static_public` byte-for-byte
with its pinned 32-byte key before deriving application keys. A pin mismatch aborts the handshake.
The proxy static private key is never sent.

## Encrypted data frames

Type `16` has this body:

```text
"PRL7"[4] | version[1] | type=16[1] | direction[1] | counter[8] | ciphertext_and_tag[*]
```

`direction=1` is client to proxy and `direction=2` is proxy to client. `counter` is unsigned
big-endian and starts at zero independently in each direction. The GCM nonce is exactly
`direction[1] || counter[8] || 0x000000[3]`.

The AAD is exactly:

```text
"pearl-e2e-v1 aad"[16] || session_id[16] || direction[1] || counter[8]
```

The length prefix covers the data body. Empty payloads are valid. Plaintext above 1 MiB, truncated
frames, invalid lengths, a wrong direction, an authentication failure, and an unknown header fail
closed.

## Replay, ordering, and rekeying

Each receiver accepts only the exact next counter. Replays and reordering are rejected before
decryption. A failed authentication does not advance the counter. The implementation refuses to
send or receive at `2^64 - 2`; the caller must close the session and perform a fresh X25519
handshake before that limit. There is no in-band rekey message, so a rekey cannot be confused with
application data.

The Rust API returns `RekeyRequired`, `CounterMismatch`, `PinnedKeyMismatch`, or
`AuthenticationFailed`; the Node API returns a `CryptoError` with the corresponding code. Keys,
ephemerals, and plaintext are never written to logs.

## Verification vectors and tests

Cross-language vectors live in `tests/crypto-vectors/`. The Node test suite is run with:

```sh
cd tools/proxy
npm test
```

The Rust suite exercises the same fixed handshake/frame bytes and checks tamper, replay, wrong-key,
wrong-pin, truncation, and counter-limit failures. `relay.test.js` uses a local fake Stratum
endpoint and verifies byte-preserving forwarding in both directions. Kryptex traffic remains an
operator-controlled M6 acceptance step.

## proxy/ + tunel.py transport

Bare `host:port` and `tunnel://host:port` now select the TCP listener of `tunel.py`.
The handshake is a 32-byte X25519 public key in hex followed by LF in each direction.
HKDF-SHA256 uses salt `csd-pool-e2e-v1` and directional info
`aes-256-gcm-key-c2s` / `aes-256-gcm-key-s2c` with the ECDH shared secret.
Each subsequent line is hex of `0x0c || nonce[12] || ciphertext || GCM_tag[16]`.
Nonce is `CSD! || counter_u64_be`, starting at one; no AAD is used.
The Python tunnel maps one TCP line to one WebSocket text message and back.
The Rust adapter assembles complete Stratum lines before encryption and restores
LF on decrypted proxy responses. A lost WebSocket session closes the TCP client;
reconnection requires a new handshake. See `proxy/src/crypto.js` for the peer.
