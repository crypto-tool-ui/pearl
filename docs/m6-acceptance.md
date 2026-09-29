# M6 Kryptex and gateway acceptance

The M6 client is in `crates/pearl-net`. `KryptexProtocol` implements the captured PRL dialect:

- TLS `stratum+ssl://...:8048` is supported with public-CA verification.
- v1 sends `mining.subscribe` followed by array `mining.authorize` and submits
  `plain_proof = base64(bincode(PlainProof))`.
- v2 is explicit configuration: object authorization and gzip-compressed bincode in the same
  `plain_proof` field.
- `mining.notify` requires a 76-byte header, a nonzero target, and `cert_version: 3`; V4 or missing
  versions pause with an update-required error.
- The pool target is preserved as sent. A `VerifiedShare` can only be created after the official
  CPU verifier passes, and stale job IDs are rejected locally before the socket write.

`KryptexPoolSet` supports one to three endpoints with reconnect/failover, keepalive, stale-job
handling, and per-kind accepted/rejected statistics. `GatewayRpcClient` provides the official
Pearl node `getblocktemplate` and `submitblock` JSON-RPC path for local/testnet solo operation.

Acceptance on a controlled local gateway, SimNet/testnet, or the operator's Kryptex test account:

```text
cargo test -p pearl-net
```

The E2E run must show at least one locally verified accepted share, zero locally invalid submits,
stale-share refusal, endpoint failover, reconnect, and gateway template/submit behavior. No real
pool traffic is permitted until this controlled run is complete.
