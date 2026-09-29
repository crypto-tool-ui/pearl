#![cfg(feature = "gpu")]

use pearl_gpu::{GemmKernel, MmaSm120CudaKernel, NaiveCudaKernel, WgmmaSm90CudaKernel};
use pearl_ref::{
    test_problem, IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, Problem,
};
use std::time::Instant;

fn gpu_ordinal() -> usize {
    match std::env::var("PEARL_GPU_DEVICE") {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("PEARL_GPU_DEVICE is not a valid ordinal: {value}")),
        Err(_) => 0,
    }
}

fn skip_gpu(reason: &str) {
    if std::env::var_os("PEARL_REQUIRE_GPU_ACCEPTANCE").is_some() {
        panic!("GPU acceptance prerequisite missing: {reason}");
    }
    eprintln!("skipping GPU test: {reason}");
}

#[test]
fn naive_cuda_matches_cpu_golden_tile() {
    if option_env!("PEARL_NAIVE_CUBIN_SM90").is_none()
        && option_env!("PEARL_NAIVE_CUBIN_SM120").is_none()
    {
        skip_gpu("nvcc was unavailable when pearl-gpu was built");
        return;
    }

    let problem = test_problem().unwrap();
    let expected = problem.tile(0, 0).unwrap();
    let ordinal = gpu_ordinal();
    let kernel_result = std::panic::catch_unwind(|| NaiveCudaKernel::new(ordinal));
    let kernel = match kernel_result {
        Err(_) => {
            skip_gpu("CUDA driver was unavailable");
            return;
        }
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("{error:#}"));
            return;
        }
    };
    let actual = kernel.compute_tile(&problem, &expected).unwrap();
    assert_eq!(actual, expected);

    let mma_result = std::panic::catch_unwind(|| MmaSm120CudaKernel::new(ordinal));
    let mma = match mma_result {
        Err(_) => {
            skip_gpu("sm_120 CUDA driver was unavailable");
            return;
        }
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_120 kernel unavailable: {error:#}"));
            return;
        }
    };
    let actual = mma.compute_tile(&problem, &expected).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn sm120_mma_matches_naive_on_10k_randomized_tiles() {
    if option_env!("PEARL_NAIVE_CUBIN_SM120").is_none()
        || option_env!("PEARL_SM120_MMA_CUBIN").is_none()
    {
        skip_gpu("sm_120 CUDA cubins were unavailable");
        return;
    }

    let ordinal = gpu_ordinal();
    let naive = match std::panic::catch_unwind(|| NaiveCudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_120 naive oracle unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_120 naive CUDA driver was unavailable");
            return;
        }
    };
    let mma = match std::panic::catch_unwind(|| MmaSm120CudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_120 MMA kernel unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_120 MMA CUDA driver was unavailable");
            return;
        }
    };

    // Use a fresh committed problem for every iteration. Alternate the official 4x8/rank32
    // shape with a full 16x16/rank128 shape so the acceptance covers both the deployed tile
    // geometry and the two-stage pipeline steady state.
    let k = 1024usize;
    let mut state = 0x9e37_79b9_u32;
    let mut next_value = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        ((state % 129) as i32 - 64) as i8
    };
    for iteration in 0..10_000 {
        let (m, n, rank, rows_pattern, cols_pattern) = if iteration % 2 == 0 {
            (
                73usize,
                42usize,
                32u16,
                PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
                PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41]).unwrap(),
            )
        } else {
            (
                16usize,
                16usize,
                128u16,
                PeriodicPattern::from_list(&[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                ])
                .unwrap(),
                PeriodicPattern::from_list(&[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                ])
                .unwrap(),
            )
        };
        let a = (0..m * k).map(|_| next_value()).collect();
        let bt = (0..n * k).map(|_| next_value()).collect();
        let problem = Problem::new(
            IncompleteBlockHeader {
                version: 0x2000_0000,
                prev_block: [0x31; 32],
                merkle_root: [0x42; 32],
                timestamp: 0x7777_0001u32.wrapping_add(iteration as u32),
                nbits: 0x207f_ffff,
            },
            MiningConfiguration {
                common_dim: k as u32,
                rank,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern,
                cols_pattern,
                moe: None,
            },
            m as u32,
            n as u32,
            a,
            bt,
        )
        .unwrap();
        let tile = problem.tile(0, 0).unwrap();
        let expected = naive.compute_tile(&problem, &tile).unwrap();
        let actual = mma.compute_tile(&problem, &tile).unwrap();
        assert_eq!(
            actual, expected,
            "randomized input iteration {iteration}"
        );
    }
}

#[test]
fn sm120_mma_benchmark() {
    if option_env!("PEARL_SM120_MMA_CUBIN").is_none() {
        skip_gpu("sm_120 benchmark cubin was unavailable");
        return;
    }

    let ordinal = gpu_ordinal();
    let mma = match std::panic::catch_unwind(|| MmaSm120CudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_120 benchmark kernel unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_120 benchmark CUDA driver was unavailable");
            return;
        }
    };
    let problem = test_problem().unwrap();
    let tile = problem.tile(0, 0).unwrap();
    let iterations = 256usize;
    let device_name = mma
        .device_name()
        .unwrap_or_else(|error| format!("unknown ({error:#})"));

    for _ in 0..8 {
        mma.compute_tile(&problem, &tile).unwrap();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        mma.compute_tile(&problem, &tile).unwrap();
    }
    let seconds = start.elapsed().as_secs_f64();
    let tiles_per_second = iterations as f64 / seconds;
    let operations = tile.rows.len() as f64
        * tile.cols.len() as f64
        * problem.config.dot_product_length() as f64
        * 2.0
        * iterations as f64;
    let effective_ths = operations / seconds / 1.0e12;
    println!(
        "sm_120 benchmark: {:.2} tiles/s, {:.3} effective TH/s ({} tiles, k={}, device={})",
        tiles_per_second,
        effective_ths,
        iterations,
        problem.config.dot_product_length(),
        device_name
    );
}

#[test]
fn sm90_wgmma_matches_cpu_golden_tile() {
    if option_env!("PEARL_NAIVE_CUBIN_SM90").is_none()
        || option_env!("PEARL_SM90_WGMMA_CUBIN").is_none()
    {
        skip_gpu("sm_90 CUDA cubins were unavailable");
        return;
    }
    let ordinal = gpu_ordinal();
    let wgmma = match std::panic::catch_unwind(|| WgmmaSm90CudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_90 WGMMA kernel unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_90 WGMMA CUDA driver was unavailable");
            return;
        }
    };
    let problem = test_problem().unwrap();
    let expected = problem.tile(0, 0).unwrap();
    assert_eq!(wgmma.compute_tile(&problem, &expected).unwrap(), expected);
}

#[test]
fn sm90_wgmma_matches_naive_on_10k_randomized_tiles() {
    if option_env!("PEARL_NAIVE_CUBIN_SM90").is_none()
        || option_env!("PEARL_SM90_WGMMA_CUBIN").is_none()
    {
        skip_gpu("sm_90 CUDA cubins were unavailable");
        return;
    }
    let ordinal = gpu_ordinal();
    let naive = match std::panic::catch_unwind(|| NaiveCudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_90 naive oracle unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_90 naive CUDA driver was unavailable");
            return;
        }
    };
    let wgmma = match std::panic::catch_unwind(|| WgmmaSm90CudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_90 WGMMA kernel unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_90 WGMMA CUDA driver was unavailable");
            return;
        }
    };

    let k = 1024usize;
    let mut state = 0x243f_6a88_u32;
    let mut next_value = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        ((state % 129) as i32 - 64) as i8
    };
    for iteration in 0..10_000 {
        let (m, n, rank, rows_pattern, cols_pattern) = if iteration % 2 == 0 {
            let indices: Vec<u32> = (0..64).collect();
            (
                64usize,
                64usize,
                32u16,
                PeriodicPattern::from_list(&indices).unwrap(),
                PeriodicPattern::from_list(&indices).unwrap(),
            )
        } else {
            let indices: Vec<u32> = (0..16).collect();
            (
                16usize,
                16usize,
                128u16,
                PeriodicPattern::from_list(&indices).unwrap(),
                PeriodicPattern::from_list(&indices).unwrap(),
            )
        };
        let a = (0..m * k).map(|_| next_value()).collect();
        let bt = (0..n * k).map(|_| next_value()).collect();
        let problem = Problem::new(
            IncompleteBlockHeader {
                version: 0x2000_0000,
                prev_block: [0x51; 32],
                merkle_root: [0x62; 32],
                timestamp: 0x8888_0001u32.wrapping_add(iteration as u32),
                nbits: 0x207f_ffff,
            },
            MiningConfiguration {
                common_dim: k as u32,
                rank,
                mma_type: MMAType::Int7xInt7ToInt32,
                rows_pattern,
                cols_pattern,
                moe: None,
            },
            m as u32,
            n as u32,
            a,
            bt,
        )
        .unwrap();
        let tile = problem.tile(0, 0).unwrap();
        let expected = naive.compute_tile(&problem, &tile).unwrap();
        let actual = wgmma.compute_tile(&problem, &tile).unwrap();
        assert_eq!(actual, expected, "randomized input iteration {iteration}");
    }
}

#[test]
fn sm90_wgmma_benchmark() {
    if option_env!("PEARL_SM90_WGMMA_CUBIN").is_none() {
        skip_gpu("sm_90 WGMMA benchmark cubin was unavailable");
        return;
    }
    let ordinal = gpu_ordinal();
    let wgmma = match std::panic::catch_unwind(|| WgmmaSm90CudaKernel::new(ordinal)) {
        Ok(Ok(kernel)) => kernel,
        Ok(Err(error)) => {
            skip_gpu(&format!("sm_90 WGMMA benchmark kernel unavailable: {error:#}"));
            return;
        }
        Err(_) => {
            skip_gpu("sm_90 WGMMA benchmark CUDA driver was unavailable");
            return;
        }
    };
    let problem = test_problem().unwrap();
    let tile = problem.tile(0, 0).unwrap();
    let iterations = 256usize;
    let device_name = wgmma
        .device_name()
        .unwrap_or_else(|error| format!("unknown ({error:#})"));
    for _ in 0..8 {
        wgmma.compute_tile(&problem, &tile).unwrap();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        wgmma.compute_tile(&problem, &tile).unwrap();
    }
    let seconds = start.elapsed().as_secs_f64();
    let tiles_per_second = iterations as f64 / seconds;
    let operations = tile.rows.len() as f64
        * tile.cols.len() as f64
        * problem.config.dot_product_length() as f64
        * 2.0
        * iterations as f64;
    let effective_ths = operations / seconds / 1.0e12;
    println!(
        "sm_90 benchmark: {:.2} tiles/s, {:.3} effective TH/s ({} tiles, k={}, device={})",
        tiles_per_second,
        effective_ths,
        iterations,
        problem.config.dot_product_length(),
        device_name
    );
}
