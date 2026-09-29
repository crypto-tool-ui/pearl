#![cfg(feature = "gpu")]

use pearl_gpu::m5::{GpuDeviceWorker, GpuJobSetup, TensorHashCudaKernel};
use pearl_ref::test_problem;

fn gpu_ordinal() -> usize {
    std::env::var("PEARL_GPU_DEVICE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn require_m5_gpu() -> Option<TensorHashCudaKernel> {
    match TensorHashCudaKernel::new(gpu_ordinal()) {
        Ok(kernel) => Some(kernel),
        Err(error) if std::env::var_os("PEARL_REQUIRE_GPU_ACCEPTANCE").is_some() => {
            panic!("M5 GPU prerequisite missing: {error:#}")
        }
        Err(error) => {
            eprintln!("skipping M5 GPU test: {error:#}");
            None
        }
    }
}

#[test]
fn m5_tensor_hash_matches_cpu_commitment() {
    let problem = test_problem().expect("deterministic test problem");
    let Some(tensor_hash) = require_m5_gpu() else { return };
    let gpu_commitment = GpuJobSetup {
        tensor_hash: &tensor_hash,
    }
    .prepare(&problem)
    .expect("GPU job setup");
    assert_eq!(gpu_commitment, problem.commitment());
}

#[test]
fn m5_double_buffer_rebuilds_next_job() {
    let problem = test_problem().expect("deterministic test problem");
    let mut worker = match GpuDeviceWorker::new(gpu_ordinal()) {
        Ok(worker) => worker,
        Err(error) if std::env::var_os("PEARL_REQUIRE_GPU_ACCEPTANCE").is_some() => {
            panic!("M5 GPU prerequisite missing: {error:#}")
        }
        Err(error) => {
            eprintln!("skipping M5 GPU test: {error:#}");
            return;
        }
    };
    let first = worker.build_next_job(&problem).expect("first buffered job");
    worker.start_ready_job(first).expect("start first buffered job");
    let second = worker.build_next_job(&problem).expect("second buffered job");
    assert_ne!(first, second);
    worker.finish_current_job();
    worker.start_ready_job(second).expect("start second buffered job");
}
