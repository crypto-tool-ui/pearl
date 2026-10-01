//! Run on an already provisioned GPU host; includes host preparation and transfers.
use anyhow::{bail, ensure, Result};
use pearl_gpu::{GemmKernel, MmaSm120CudaKernel, MmaSm75CudaKernel, WgmmaSm90CudaKernel};
use pearl_ref::{test_problem, PeriodicPattern, Problem};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let device = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let kernel: Box<dyn GemmKernel> = match args.get(1).map(String::as_str) {
        Some("sm75") => Box::new(MmaSm75CudaKernel::new(device)?),
        Some("sm120") => Box::new(MmaSm120CudaKernel::new(device)?),
        Some("sm90") => Box::new(WgmmaSm90CudaKernel::new(device)?),
        _ => bail!("usage: throughput sm75|sm90|sm120 [device]"),
    };
    let seed = test_problem()?;
    let mut config = seed.config;
    config.common_dim = 4096;
    config.rank = 128;
    config.rows_pattern = PeriodicPattern::from_list(&[0, 8, 16, 24, 32, 40, 48, 56])?;
    config.cols_pattern = PeriodicPattern::from_list(&[0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57])?;
    // Production hash tile geometry with a small committed matrix, not a full mining attempt.
    let problem = Problem::new(seed.header, config, 128, 128,
        (0..128 * 4096).map(|i| ((i * 13 + 7) % 129 - 64) as i8).collect(),
        (0..128 * 4096).map(|i| ((i * 17 + 3) % 129 - 64) as i8).collect())?;
    let (rows, cols) = problem.tile_offsets();
    let requests: Vec<_> = rows.iter().flat_map(|&r| cols.iter().map(move |&c| (r, c)))
        .take(kernel.max_batch_size())
        .map(|(r, c)| problem.empty_tile(r, c)).collect::<Result<_>>()?;
    let expected = requests.iter().map(|t| problem.tile(t.t_rows, t.t_cols))
        .collect::<Result<Vec<_>>>()?;
    ensure!(kernel.compute_tiles(&problem, &requests)? == expected, "GPU differs from CPU reference");
    for batch_size in [1, requests.len()] {
        if batch_size == 1 && requests.len() == 1 {
            eprintln!("sm90 currently uses single-tile dispatch");
        }
        let batch = &requests[..batch_size];
        for _ in 0..3 { kernel.compute_tiles(&problem, batch)?; }
        let started = Instant::now();
        let mut iterations = 0u64;
        while started.elapsed() < Duration::from_secs(10) {
            kernel.compute_tiles(&problem, batch)?;
            iterations += 1;
        }
        let seconds = started.elapsed().as_secs_f64();
        let macs = iterations as f64 * batch_size as f64 * 8.0 * 16.0 * 4096.0;
        println!("batch={batch_size} iterations={iterations} elapsed={seconds:.3}s wall_MAC/s={:.3e} TH/s={:.9}", macs / seconds, macs / seconds / 1e12);
        if requests.len() == 1 { break; }
    }
    Ok(())
}
