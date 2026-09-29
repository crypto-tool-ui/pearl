use pearl_net::{KryptexMode, KryptexProtocol, PoolEndpoint, VerifiedShare};
use serde_json::json;
use std::io::Read;

#[test]
fn kryptex_v1_frames_match_captured_dialect() {
    let protocol = KryptexProtocol::new(
        "prl1wallet".to_owned(),
        "rig0".to_owned(),
        "d=2097152".to_owned(),
        KryptexMode::V1,
    );
    let frames = protocol.handshake(1);
    assert_eq!(frames[0]["method"], "mining.subscribe");
    assert_eq!(frames[0]["params"][0], protocol.agent);
    assert_eq!(frames[1]["method"], "mining.authorize");
    assert_eq!(frames[1]["params"][0], "prl1wallet.rig0");
    assert_eq!(protocol.authorize_id(1), 2);
}

#[test]
fn kryptex_notify_preserves_pool_target_and_certificate() {
    let protocol = KryptexProtocol::new("w".into(), "r".into(), "x".into(), KryptexMode::V1);
    let header = "00".repeat(76);
    let job = protocol
        .parse_job(&json!({
            "id": null,
            "method": "mining.notify",
            "params": {
                "job_id": "a7460708_2097152",
                "header": header,
                "target": "00000000000007ffffffffffffffffffffffffffffffffffffffffffffffffff",
                "height": 119376,
                "cert_version": 3
            }
        }))
        .unwrap()
        .unwrap();
    assert_eq!(job.header.len(), 76);
    assert_eq!(job.target_be[0], 0);
    assert_eq!(hex::encode(job.target_be),
        "00000000000007ffffffffffffffffffffffffffffffffffffffffffffffffff");
    assert_eq!(job.cert_version, Some(3));
    assert!(!job.requires_update());
}

#[test]
fn endpoint_and_submit_are_tls_safe_defaults() {
    let endpoint = PoolEndpoint::parse("stratum+ssl://prl-br.kryptex.network:8048").unwrap();
    assert!(endpoint.tls);
    assert_eq!(endpoint.port, 8048);
    let protocol = KryptexProtocol::new("w".into(), "r".into(), "x".into(), KryptexMode::V1);
    let share = VerifiedShare {
        job_id: "job_1".into(),
        proof_bytes: vec![1, 2, 3],
        digest: [0; 32],
        hashrate: 0,
    };
    let submit = protocol.submit(4, &share).unwrap();
    assert_eq!(submit["method"], "mining.submit");
    assert_eq!(submit["params"]["job_id"], "job_1");
    assert_eq!(submit["params"]["plain_proof"], "AQID");
}

#[test]
fn kryptex_v2_matches_b200_wire_contract() {
    let protocol = KryptexProtocol::new(
        "prl1wallet".into(),
        "b200".into(),
        "secret-is-not-sent-in-v2".into(),
        KryptexMode::V2,
    );
    let handshake = protocol.handshake(1);
    assert_eq!(handshake.len(), 1);
    assert_eq!(handshake[0]["params"]["wallet"], "prl1wallet.b200");
    assert_eq!(handshake[0]["params"]["type"], "v2");
    assert!(handshake[0]["params"].get("password").is_none());

    let share = VerifiedShare {
        job_id: "job_1".into(),
        proof_bytes: vec![1, 2, 3, 0],
        digest: [0; 32],
        hashrate: 1_440_311_612_472_699,
    };
    let submit = protocol.submit(4, &share).unwrap();
    assert_eq!(submit["params"]["hs"], share.hashrate);
    assert!(submit["params"].get("worker").is_none());
    let encoded = submit["params"]["plain_proof"].as_str().unwrap();
    let compressed = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        encoded,
    )
    .unwrap();
    let mut decoder = flate2::read::GzDecoder::new(compressed.as_slice());
    let mut legacy = Vec::new();
    decoder.read_to_end(&mut legacy).unwrap();
    assert_eq!(legacy, [1, 2, 3]);
}

#[test]
fn endpoint_selects_tunnel_and_explicit_legacy_protocol() {
    for url in ["localhost:9000", "tunnel://localhost:9000"] {
        let endpoint = PoolEndpoint::parse(url).unwrap();
        assert!(endpoint.e2e);
        assert!(!endpoint.legacy_prl7);
        assert_eq!(endpoint.address(), "localhost:9000");
    }
    assert!(PoolEndpoint::parse("prl7://localhost:3000").unwrap().legacy_prl7);
    assert!(!PoolEndpoint::parse("stratum+tcp://localhost:8760").unwrap().e2e);
    assert!(PoolEndpoint::parse("ws://localhost:8443").is_err());
}
