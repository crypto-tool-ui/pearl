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
    assert_eq!(frames[1]["params"][1], "d=2097152");
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
        "d=2097152".into(),
        KryptexMode::V2,
    );
    let handshake = protocol.handshake(1);
    assert_eq!(handshake.len(), 1);
    assert_eq!(handshake[0]["params"]["wallet"], "prl1wallet.b200");
    assert_eq!(handshake[0]["params"]["type"], "v2");
    assert_eq!(handshake[0]["params"]["password"], "d=2097152");

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

#[test]
fn production_pattern_share_survives_v2_encoding_and_official_verification() {
    use pearl_ref::{Problem, MiningConfiguration, PeriodicPattern, MMAType,
        KRYPTEX_ROWS, KRYPTEX_COLS, SeedDerivation};
    use primitive_types::U256;
    let header = pearl_ref::test_problem().unwrap().header;
    let k = 4096;
    let config = MiningConfiguration {
        common_dim: k, rank: 128, mma_type: MMAType::Int7xInt7ToInt32, moe: None,
        rows_pattern: PeriodicPattern::from_list(&KRYPTEX_ROWS).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&KRYPTEX_COLS).unwrap(),
    };
    let a = (0..64 * k).map(|i| (((i * 13 + 7) % 129) as i32 - 64) as i8).collect();
    let bt = (0..64 * k).map(|i| (((i * 17 + 3) % 129) as i32 - 64) as i8).collect();
    let problem = Problem::new(header, config, 64, 64, a, bt).unwrap();
    let tile = problem.tile(0, 0).unwrap();
    // Synthetic easy target for a deterministic CPU test; not a live pool target.
    let target = U256::MAX >> 19;
    let mut target_be = [0; 32];
    target.to_big_endian(&mut target_be);
    let mut job = pearl_net::PoolJob { job_id: "cpu-proof".into(), header: header.to_bytes(),
        target, target_be, height: None, difficulty: None, cert_version: Some(3) };
    assert!(U256::from_little_endian(&tile.digest) <= pearl_ref::share_bound(target, &problem.config).unwrap());
    let share = VerifiedShare::from_verified_tile_with_hashrate(&problem, &job, &tile, 200_000_000_000).unwrap();
    let protocol = KryptexProtocol::new("test-wallet".into(), "test-worker".into(), "d=2097152".into(), KryptexMode::V2);
    let wire = protocol.submit(2, &share).unwrap();
    let compressed = base64::Engine::decode(&base64::engine::general_purpose::STANDARD,
        wire["params"]["plain_proof"].as_str().unwrap()).unwrap();
    let mut restored = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice()).read_to_end(&mut restored).unwrap();
    restored.push(0); // Restore dense moe=None after decoding the pool's V2 wire representation.
    assert_eq!(restored, share.proof_bytes);
    let proof = bincode::deserialize(&restored).unwrap();
    pearl_ref::verify_plain_proof(&header, &proof, Some(pearl_ref::compact_from_target(target)), SeedDerivation::Salted).unwrap();
    job.target = U256::one();
    assert!(VerifiedShare::from_verified_tile(&problem, &job, &tile).is_err());
}
