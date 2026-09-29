//! Kryptex job admission and the correctness-first tile loop.

use anyhow::{ensure, Context, Result};
use pearl_gpu::GemmKernel;
use pearl_net::{PoolJob, VerifiedShare};
use primitive_types::U256;
use pearl_ref::{share_bound, IncompleteBlockHeader, Problem};
use std::time::Instant;
use tracing::{debug, info, warn};

/// Builds the committed V3 work unit from the exact 76-byte header sent by Kryptex.
pub fn problem_for_job(job: &PoolJob) -> Result<Problem> {
    ensure!(!job.requires_update(), "update required: unsupported certificate version");
    let header = IncompleteBlockHeader::from_bytes(&job.header)
        .context("decoding Kryptex IncompleteBlockHeader")?;
    ensure!(header.nbits != 0, "Kryptex job has zero block target");
    Problem::from_kryptex_header(header)
}

/// Scans the fixed Kryptex tile partition with the selected production kernel.
///
/// A tile is considered a candidate only after the GPU returns its fused digest. The CPU
/// reference then reconstructs the official Merkle proof and verifier before the caller can submit
/// it. The loop checks the current job boundary between tiles so a supervisor can cancel work on a
/// future job update.
pub fn find_verified_share(
    problem: &Problem,
    job: &PoolJob,
    kernel: &dyn GemmKernel,
    should_cancel: impl Fn() -> bool,
) -> Result<Option<VerifiedShare>> {
    ensure!(!job.requires_update(), "update required: unsupported certificate version");
    ensure!(problem.header.to_bytes() == job.header, "problem header does not match Kryptex job");
    let bound = share_bound(job.target, &problem.config)
        .filter(|bound| !bound.is_zero())
        .ok_or_else(|| anyhow::anyhow!("Kryptex share target is unusable for this configuration"))?;
    let (row_offsets, col_offsets) = problem.tile_offsets();
    let total = row_offsets.len().saturating_mul(col_offsets.len());
    info!(job_id = %job.job_id, tiles = total, "starting Kryptex job");
    let mut scanned = 0usize;
    let started = Instant::now();

    for t_rows in row_offsets {
        for t_cols in &col_offsets {
            if should_cancel() {
                warn!(job_id = %job.job_id, scanned, "Kryptex job cancelled as stale");
                return Ok(None);
            }
            let request = problem.empty_tile(t_rows, *t_cols)?;
            let tile = kernel
                .compute_tile(problem, &request)
                .with_context(|| format!("computing Kryptex tile ({t_rows},{t_cols})"))?;
            scanned += 1;
            if U256::from_little_endian(&tile.digest) > bound {
                continue;
            }
            debug!(job_id = %job.job_id, t_rows, t_cols, scanned, "GPU candidate reached share bound");
            let elapsed = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
            let operations = (scanned as u128)
                .saturating_mul(problem.m as u128)
                .saturating_mul(problem.n as u128)
                .saturating_mul(problem.k() as u128);
            let hashrate = (operations as f64 / elapsed)
                .min(u64::MAX as f64)
                .max(0.0) as u64;
            let share = VerifiedShare::from_verified_tile_with_hashrate(
                problem,
                job,
                &tile,
                hashrate,
            )
                .context("CPU reference rejected GPU candidate")?;
            info!(job_id = %job.job_id, t_rows, t_cols, scanned, "CPU-verified Kryptex share ready");
            return Ok(Some(share));
        }
    }
    info!(job_id = %job.job_id, scanned, "Kryptex job exhausted without a share");
    Ok(None)
}
