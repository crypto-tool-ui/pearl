use pearl_ref::{test_problem, verify_plain_proof, Problem, SeedDerivation};
use serde_json::Value;

fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn tile_value(problem: &Problem, t_rows: u32, t_cols: u32) -> Value {
    let tile = problem.tile(t_rows, t_cols).unwrap();
    serde_json::json!({
        "t_rows": tile.t_rows,
        "t_cols": tile.t_cols,
        "rows": tile.rows,
        "cols": tile.cols,
        "transcript": tile.transcript,
        "digest": hash_hex(&tile.digest),
    })
}

#[test]
fn v3_dense_golden_vector_is_bit_exact() {
    let expected: Value =
        serde_json::from_str(include_str!("../../../tests/golden/v3_dense.json")).unwrap();
    let problem = test_problem().unwrap();
    let commitment = problem.commitment();

    assert_eq!(
        hex::encode(problem.header.to_bytes()),
        expected["inputs"]["header_hex"]
    );
    assert_eq!(
        hex::encode(problem.config.to_bytes()),
        expected["inputs"]["config_hex"]
    );
    assert_eq!(
        hash_hex(&commitment.job_key),
        expected["commitment"]["job_key"]
    );
    assert_eq!(
        hash_hex(&commitment.root_a),
        expected["commitment"]["root_a"]
    );
    assert_eq!(
        hash_hex(&commitment.root_b),
        expected["commitment"]["root_b"]
    );
    assert_eq!(
        hash_hex(&commitment.bound_a),
        expected["commitment"]["bound_a"]
    );
    assert_eq!(
        hash_hex(&commitment.bound_b),
        expected["commitment"]["bound_b"]
    );
    assert_eq!(
        hash_hex(&commitment.b_noise_seed),
        expected["commitment"]["b_noise_seed"]
    );
    assert_eq!(
        hash_hex(&commitment.a_noise_seed),
        expected["commitment"]["a_noise_seed"]
    );

    let expected_tiles = expected["tiles"].as_array().unwrap();
    for expected_tile in expected_tiles {
        let t_rows = expected_tile["t_rows"].as_u64().unwrap() as u32;
        let t_cols = expected_tile["t_cols"].as_u64().unwrap() as u32;
        assert_eq!(tile_value(&problem, t_rows, t_cols), *expected_tile);
    }

    let valid_tile = problem.tile(0, 0).unwrap();
    problem.verify_tile(&valid_tile, None).unwrap();

    let mut bad_header = problem.header;
    bad_header.timestamp ^= 1;
    let proof = problem.proof_for_tile(&valid_tile).unwrap();
    assert!(verify_plain_proof(&bad_header, &proof, None, SeedDerivation::Salted).is_err());

    let mut bad_proof = proof;
    bad_proof.noise_rank = 32;
    assert!(verify_plain_proof(&problem.header, &bad_proof, None, SeedDerivation::Salted).is_err());

    assert!(expected["candidate_checks"]["valid_tile_0"]
        .as_bool()
        .unwrap());
    assert!(!expected["candidate_checks"]["header_timestamp_bit_flip"]
        .as_bool()
        .unwrap());
    assert!(!expected["candidate_checks"]["proof_noise_rank_32"]
        .as_bool()
        .unwrap());
    assert!(expected["inputs"]["nonce_encoding"].is_null());
}
