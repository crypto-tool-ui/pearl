//! Kryptex job admission and the correctness-first tile loop.

use anyhow::{ensure, Context, Result};
use pearl_gpu::GemmKernel;
use pearl_net::PoolJob;
use pearl_ref::{share_bound, IncompleteBlockHeader, Problem, TileResult};
use primitive_types::U256;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Builds the committed V3 work unit from the exact 76-byte header sent by Kryptex.
pub fn problem_for_job(job: &PoolJob, attempt: u64) -> Result<Problem> {
    ensure!(
        !job.requires_update(),
        "update required: unsupported certificate version"
    );
    let header = IncompleteBlockHeader::from_bytes(&job.header)
        .context("decoding Kryptex IncompleteBlockHeader")?;
    ensure!(header.nbits != 0, "Kryptex job has zero block target");
    Problem::from_kryptex_header_attempt(header, attempt)
}

/// Scans the fixed Kryptex tile partition with the selected production kernel.
///
/// GPU target hits go to a bounded CPU proof worker; scanning continues after every hit.
/// The caller must verify every candidate before network submission.
pub fn scan_job(
    problem: &Problem,
    job: &PoolJob,
    kernel: &dyn GemmKernel,
    should_cancel: impl Fn() -> bool,
    started: Instant,
    mut on_candidate: impl FnMut(TileResult, u64) -> Result<()>,
) -> Result<()> {
    ensure!(
        !job.requires_update(),
        "update required: unsupported certificate version"
    );
    ensure!(
        problem.header.to_bytes() == job.header,
        "problem header does not match Kryptex job"
    );
    let bound = share_bound(job.target, &problem.config)
        .filter(|bound| !bound.is_zero())
        .ok_or_else(|| {
            anyhow::anyhow!("Kryptex share target is unusable for this configuration")
        })?;
    let (row_offsets, col_offsets) = problem.tile_offsets();
    let total = row_offsets.len().saturating_mul(col_offsets.len());
    let mut scanned = 0usize;
    let grouped = kernel.mining_groups(problem)?;
    let batch_tiles = grouped.map_or(kernel.batch_size(), |(_, groups)| groups as usize * 128);
    info!(
        "マイニング開始 | ジョブ={} | 総タイル数={} | バッチ={}",
        job.job_id, total, batch_tiles
    );
    let mut group_cursor = 0;
    let mut bound_bytes = [0; 32];
    bound.to_little_endian(&mut bound_bytes);

    let mut offsets = row_offsets
        .into_iter()
        .flat_map(|row| col_offsets.iter().copied().map(move |col| (row, col)));
    let mut last_report = Instant::now();
    let mut last_scanned = 0usize;
    loop {
        if should_cancel() {
            warn!(job_id = %job.job_id, scanned, "新しいジョブを受信したため、古いジョブを中止しました");
            return Ok(());
        }
        let (tiles, completed, per_tile) = if let Some((groups, batch)) = grouped {
            if group_cursor == groups {
                break;
            }
            let count = batch.min(groups - group_cursor);
            let hits = kernel.scan_groups(problem, group_cursor, count, bound_bytes)?;
            group_cursor += count;
            (hits, count as usize * 128, 128.0 * problem.k() as f64)
        } else {
            let requests = offsets
                .by_ref()
                .take(kernel.batch_size())
                .map(|(row, col)| problem.empty_tile(row, col))
                .collect::<Result<Vec<_>>>()?;
            if requests.is_empty() {
                break;
            }
            let tiles = kernel
                .compute_tiles(problem, &requests)
                .context("computing Kryptex tile batch")?;
            ensure!(
                tiles.len() == requests.len(),
                "GPU returned an incomplete tile batch"
            );
            let work =
                requests[0].rows.len() as f64 * requests[0].cols.len() as f64 * problem.k() as f64;
            (tiles, requests.len(), work)
        };
        if should_cancel() {
            return Ok(());
        }
        scanned += completed;
        let elapsed = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
        // Count the logical cells actually computed, not the dimensions of the entire job.
        let hashrate = (scanned as f64 * per_tile / elapsed).min(u64::MAX as f64) as u64;
        if last_report.elapsed().as_secs() >= 10 {
            let current_tiles =
                (scanned - last_scanned) as f64 / last_report.elapsed().as_secs_f64();
            info!("採掘状況 | ジョブ={} | ハッシュレート={} | 平均={} | タイル/秒={:.2} | 処理済み={} | バッチ={}",
                job.job_id, format_hashrate(current_tiles * per_tile), format_hashrate(hashrate as f64),
                current_tiles, scanned, completed);
            last_scanned = scanned;
            last_report = Instant::now();
        }
        for tile in tiles {
            if should_cancel() {
                return Ok(());
            }
            if U256::from_little_endian(&tile.digest) > bound {
                continue;
            }
            debug!(job_id = %job.job_id, t_rows = tile.t_rows, t_cols = tile.t_cols, scanned,
                "GPU候補がシェアの目標値を満たしました");
            on_candidate(tile, hashrate)?;
        }
    }
    info!(job_id = %job.job_id, scanned, "全タイルの処理が完了しました");
    Ok(())
}

// Keep the pool's existing logical h*w*k rate convention; do not count an MMA
// multiply and add twice or confuse tile hashes/s with this work-normalized rate.
fn format_hashrate(rate: f64) -> String {
    for (scale, unit) in [(1e12, "TH/s"), (1e9, "GH/s"), (1e6, "MH/s"), (1e3, "kH/s")] {
        if rate >= scale {
            return format!("{:.3} {unit}", rate / scale);
        }
    }
    format!("{rate:.3} H/s")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    struct GroupMock {
        calls: RefCell<Vec<(u32, u32)>>,
    }
    impl GemmKernel for GroupMock {
        fn metadata(&self) -> pearl_gpu::KernelMetadata {
            pearl_gpu::KernelMetadata {
                architecture: pearl_gpu::KernelArchitecture::Sm75Mma,
                precision: pearl_gpu::KernelPrecision::Int7Int7Int32,
                certificate_version: 3,
            }
        }
        fn compute_tile(&self, _: &Problem, _: &TileResult) -> Result<TileResult> {
            panic!("production scan must not call the per-tile fallback")
        }
        fn mining_groups(&self, _: &Problem) -> Result<Option<(u32, u32)>> {
            Ok(Some((3, 2)))
        }
        fn scan_groups(
            &self,
            p: &Problem,
            begin: u32,
            count: u32,
            _: [u8; 32],
        ) -> Result<Vec<TileResult>> {
            self.calls.borrow_mut().push((begin, count));
            // Two hits in the first batch, one in the tail; zero digest passes any nonzero bound.
            (begin..begin + count)
                .map(|g| p.empty_tile(0, g * 128))
                .collect()
        }
    }
    fn fixture() -> (Problem, PoolJob) {
        let p = pearl_gpu::production_test_problem(128, 384, 4096).unwrap();
        let job = PoolJob {
            job_id: "scan-regression".into(),
            header: p.header.to_bytes(),
            target: U256::MAX >> 64,
            target_be: [0; 32],
            height: None,
            difficulty: None,
            cert_version: Some(3),
        };
        (p, job)
    }

    #[test]
    fn grouped_scan_continues_after_hits_and_covers_partial_batch() {
        let (p, job) = fixture();
        let kernel = GroupMock {
            calls: RefCell::new(Vec::new()),
        };
        let mut hits = Vec::new();
        scan_job(
            &p,
            &job,
            &kernel,
            || false,
            Instant::now(),
            |tile, _| {
                hits.push(tile.t_cols);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*kernel.calls.borrow(), [(0, 2), (2, 1)]);
        assert_eq!(hits, [0, 128, 256]);
    }

    #[test]
    fn grouped_scan_cancels_between_candidates() {
        let (p, job) = fixture();
        let kernel = GroupMock {
            calls: RefCell::new(Vec::new()),
        };
        let cancelled = Cell::new(false);
        let hits = Cell::new(0);
        scan_job(
            &p,
            &job,
            &kernel,
            || cancelled.get(),
            Instant::now(),
            |_, _| {
                hits.set(hits.get() + 1);
                cancelled.set(true);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(hits.get(), 1);
        assert_eq!(*kernel.calls.borrow(), [(0, 2)]);
    }

    #[test]
    fn proof_queue_failure_stops_scanning() {
        let (p, job) = fixture();
        let kernel = GroupMock {
            calls: RefCell::new(Vec::new()),
        };
        assert!(scan_job(
            &p,
            &job,
            &kernel,
            || false,
            Instant::now(),
            |_, _| { anyhow::bail!("proof worker failed") }
        )
        .is_err());
        assert_eq!(*kernel.calls.borrow(), [(0, 2)]);
    }
    #[test]
    fn hashrate_units_match_the_reported_work_rate() {
        assert_eq!(format_hashrate(934506703.0), "934.507 MH/s");
        assert_eq!(format_hashrate(1e9), "1.000 GH/s");
        assert_eq!(format_hashrate(1e12), "1.000 TH/s");
        assert_eq!(format_hashrate(0.0), "0.000 H/s");
    }
}
