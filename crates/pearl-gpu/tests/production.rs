#![cfg(feature = "gpu")]

use pearl_gpu::{production_test_problem, select_production_kernel};
use std::time::Instant;

#[test]
fn production_correctness_and_resident_benchmark() {
    let device = std::env::var("PEARL_GPU_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .expect("PEARL_GPU_DEVICE must be an ordinal");
    let kernel = match std::panic::catch_unwind(|| select_production_kernel(device)) {
        Ok(Ok(kernel)) => kernel,
        error => {
            if std::env::var_os("PEARL_REQUIRE_GPU_ACCEPTANCE").is_some() {
                panic!("production acceptance requires compiled cubins and a supported CUDA device: {}",
                    match error { Ok(Err(e)) => format!("{e:#}"), _ => "CUDA driver unavailable".into() });
            }
            eprintln!("SKIP: production CUDA device/cubin unavailable (not hardware acceptance)");
            return;
        }
    };
    kernel
        .mining_selftest()
        .expect("CPU/GPU production equivalence");
    if kernel.device_matrix_preparation() {
        let p = pearl_ref::Problem::from_kryptex_header_attempt_deferred(
            pearl_ref::test_problem().unwrap().header, 7).unwrap();
        assert!(!p.a.is_materialized() && !p.bt.is_materialized());
        kernel.prepare_job(&p).expect("resident generated production matrices");
        assert!(!p.a.is_materialized() && !p.bt.is_materialized(),
            "GPU preparation unexpectedly materialized CPU matrices");
        let tiles = kernel.scan_groups(&p, 0, 1, [255; 32]).unwrap();
        assert_eq!(tiles.len(), 128);
        assert_eq!(tiles[0], p.tile(tiles[0].t_rows, tiles[0].t_cols).unwrap());
    }
    let problem = production_test_problem(8192, 8192, 4096).unwrap();
    let preparation = Instant::now();
    kernel.prepare_job(&problem).unwrap();
    let prep_seconds = preparation.elapsed().as_secs_f64();
    let (groups, batch) = kernel.mining_groups(&problem).unwrap().unwrap();
    // Out-of-bounds and stale resident jobs must fail before launching CUDA.
    assert!(kernel.scan_groups(&problem, groups, 1, [0; 32]).is_err());
    assert!(kernel.scan_groups(&problem, 0, 0, [0; 32]).is_err());
    let other = production_test_problem(128, 128, 256).unwrap();
    assert!(kernel.scan_groups(&other, 0, 1, [0; 32]).is_err());
    // Warm up the same resident path used in mining. Zero target avoids CPU proof costs.
    kernel
        .scan_groups(&problem, 0, batch.min(groups), [0; 32])
        .unwrap();
    let start = Instant::now();
    let mut scanned = 0u64;
    let mut launches = 0u64;
    for _ in 0..3 {
        for begin in (0..groups).step_by(batch as usize) {
            let count = batch.min(groups - begin);
            assert!(kernel
                .scan_groups(&problem, begin, count, [0; 32])
                .unwrap()
                .is_empty());
            scanned += u64::from(count) * 128;
            launches += 1;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("production {:?}: prep={prep_seconds:.3}s scan={seconds:.3}s launches={launches} \
        tiles/s={:.2} TH/s={:.3} (MAC/s, includes launch/target/count-readback; excludes job prep and proofs)",
        kernel.metadata().architecture, scanned as f64 / seconds,
        scanned as f64 * 128.0 * 4096.0 / seconds / 1e12);
}
