//! T4/Turing host adapter for the dedicated m8n8k16 Tensor Core kernel.
use crate::{
    load_cuda_kernel, GemmKernel, KernelArchitecture, KernelMetadata, KernelPrecision,
    LoadedCudaKernel,
};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{result, sys, CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use pearl_ref::{compute_jackpot_hash, MMAType, Problem, TileResult};
use std::{cell::RefCell, ffi::c_void, sync::Arc};
const SM75_MMA_M: usize = 8;
const SM75_MMA_N: usize = 8;
const SM75_MMA_K: usize = 16;
const SM75_MAX_WARPS: usize = 32;

pub struct MmaSm75CudaKernel {
    context: Arc<CudaContext>,
    kernel: LoadedCudaKernel,
    max_threads_per_block: usize,
    max_block_dim_x: usize,
    max_dynamic_shared_memory: usize,
    resident: RefCell<Option<ResidentJob>>,
    cubin: &'static [u8],
}

impl MmaSm75CudaKernel {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("opening CUDA device {device_ordinal}"))?;
        Self::from_context(context)
    }

    pub fn device_name(&self) -> Result<String> {
        self.context.name().context("querying CUDA device name")
    }

    pub(crate) fn from_context(context: Arc<CudaContext>) -> Result<Self> {
        let capability = context
            .compute_capability()
            .context("querying CUDA compute capability")?;
        if capability != (7, 5) {
            bail!(
                "sm_75 MMA kernel requires compute capability 7.5, found {}.{}",
                capability.0,
                capability.1
            );
        }
        let cubin = embedded_cubin!("PEARL_SM75_MMA_CUBIN")
            .ok_or_else(|| anyhow::anyhow!("sm_75 MMA cubin was not built; install CUDA nvcc"))?;
        Self::with_cubin(context, cubin)
    }

    fn with_cubin(context: Arc<CudaContext>, cubin: &'static [u8]) -> Result<Self> {
        let warp_size = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            .context("querying CUDA warp size")?;
        ensure!(
            warp_size == 32,
            "sm_75 MMA kernel requires 32-wide warps, found {warp_size}"
        );
        let max_threads_per_block = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .context("querying CUDA block-thread limit")?;
        let max_threads_per_block = usize::try_from(max_threads_per_block)
            .context("CUDA block-thread limit was negative")?;
        let max_block_dim_x = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
            .context("querying CUDA block X-dimension limit")?;
        let max_block_dim_x = usize::try_from(max_block_dim_x)
            .context("CUDA block X-dimension limit was negative")?;
        let max_dynamic_shared_memory = context
            .attribute(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .context("querying sm_75 opt-in shared-memory limit")?;
        let max_dynamic_shared_memory = usize::try_from(max_dynamic_shared_memory)
            .context("sm_75 shared-memory limit was negative")?;
        let kernel = load_cuda_kernel(&context, cubin, "pearl_sm75_mma_tile")?;
        Ok(Self {
            context,
            kernel,
            max_threads_per_block,
            max_block_dim_x,
            max_dynamic_shared_memory,
            resident: RefCell::new(None),
            cubin,
        })
    }

    fn padded_operands(
        problem: &Problem,
        tile: &TileResult,
        operands: (Vec<i8>, Vec<i8>),
    ) -> Result<(Vec<i8>, Vec<i8>, u32, u32)> {
        ensure!(
            problem.config.mma_type == MMAType::Int7xInt7ToInt32,
            "sm_75 kernel only supports the official V3 INT8 representation"
        );
        ensure!(
            problem.config.moe.is_none(),
            "sm_75 dense kernel does not support grouped GEMM"
        );
        let (a, bt) = operands;
        let h = tile.rows.len();
        let w = tile.cols.len();
        let k = problem.k();
        ensure!(h > 0 && w > 0, "sm_75 kernel requires a non-empty tile");
        ensure!(
            h <= u32::MAX as usize,
            "sm_75 row count exceeds CUDA parameter width"
        );
        ensure!(
            w <= u32::MAX as usize,
            "sm_75 column count exceeds CUDA parameter width"
        );
        let padded_h = h
            .checked_add(SM75_MMA_M - 1)
            .context("sm_75 padded row count overflow")?
            / SM75_MMA_M;
        let padded_h = padded_h
            .checked_mul(SM75_MMA_M)
            .context("sm_75 padded row count overflow")?;
        let padded_w = w
            .checked_add(SM75_MMA_N - 1)
            .context("sm_75 padded column count overflow")?
            / SM75_MMA_N;
        let padded_w = padded_w
            .checked_mul(SM75_MMA_N)
            .context("sm_75 padded column count overflow")?;
        ensure!(
            padded_h <= u32::MAX as usize,
            "sm_75 padded rows exceed CUDA parameter width"
        );
        ensure!(
            padded_w <= u32::MAX as usize,
            "sm_75 padded columns exceed CUDA parameter width"
        );
        let padded_a_len = padded_h
            .checked_mul(k)
            .context("sm_75 padded A size overflow")?;
        let padded_bt_len = padded_w
            .checked_mul(k)
            .context("sm_75 padded B^T size overflow")?;
        let mut padded_a = vec![0i8; padded_a_len];
        let mut padded_bt = vec![0i8; padded_bt_len];
        for row in 0..h {
            padded_a[row * k..(row + 1) * k].copy_from_slice(&a[row * k..(row + 1) * k]);
        }
        for row in 0..w {
            padded_bt[row * k..(row + 1) * k].copy_from_slice(&bt[row * k..(row + 1) * k]);
        }
        Ok((padded_a, padded_bt, padded_h as u32, padded_w as u32))
    }

    fn launch_transcript(
        &self,
        a: &[i8],
        bt: &[i8],
        logical_h: u32,
        logical_w: u32,
        padded_h: u32,
        padded_w: u32,
        physical_k: u32,
        effective_k: u32,
        rank: u32,
        a_noise_seed: &[u8; 32],
        batch: usize,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        ensure!(batch > 0 && batch <= 128, "invalid sm_75 batch size");
        ensure!(rank > 0, "sm_75 kernel requires a nonzero rank");
        ensure!(
            logical_h > 0 && logical_w > 0,
            "sm_75 kernel requires a non-empty tile"
        );
        ensure!(
            padded_h >= logical_h && padded_w >= logical_w,
            "sm_75 padded tile is too small"
        );
        ensure!(
            padded_h as usize % SM75_MMA_M == 0 && padded_w as usize % SM75_MMA_N == 0,
            "sm_75 padded tile is misaligned"
        );
        let expected_a_len = (padded_h as usize)
            .checked_mul(physical_k as usize)
            .and_then(|size| size.checked_mul(batch))
            .context("sm_75 A operand length overflow")?;
        let expected_bt_len = (padded_w as usize)
            .checked_mul(physical_k as usize)
            .and_then(|size| size.checked_mul(batch))
            .context("sm_75 B^T operand length overflow")?;
        ensure!(
            a.len() == expected_a_len,
            "sm_75 A operand length does not match its stride"
        );
        ensure!(
            bt.len() == expected_bt_len,
            "sm_75 B^T operand length does not match its stride"
        );
        let tiles_m = (padded_h as usize).div_ceil(SM75_MMA_M);
        let tiles_n = (padded_w as usize).div_ceil(SM75_MMA_N);
        let warps = tiles_m
            .checked_mul(tiles_n)
            .context("sm_75 tile warp count overflow")?;
        ensure!(
            warps > 0 && warps <= SM75_MAX_WARPS,
            "sm_75 tile requires {warps} warps"
        );
        let threads = (warps as usize)
            .checked_mul(32)
            .context("sm_75 block-thread count overflow")?;
        ensure!(
            threads <= self.max_threads_per_block,
            "sm_75 tile requests {threads} threads, device allows {}",
            self.max_threads_per_block
        );
        ensure!(
            threads <= self.max_block_dim_x,
            "sm_75 1D block requests {threads} threads on X, device allows {}",
            self.max_block_dim_x
        );
        ensure!(
            rank % (SM75_MMA_K as u32) == 0
                && physical_k % (SM75_MMA_K as u32) == 0
                && effective_k % (SM75_MMA_K as u32) == 0,
            "sm_75 kernel requires 16-aligned K dimensions"
        );
        ensure!(
            effective_k <= physical_k && effective_k / rank * rank == effective_k,
            "sm_75 kernel requires a rank-aligned effective K"
        );
        ensure!(
            effective_k > 0,
            "sm_75 kernel requires a non-empty effective K"
        );
        let accumulator_cells = (logical_h as usize)
            .checked_mul(logical_w as usize)
            .context("sm_75 accumulator cell count overflow")?;
        let accumulator_bytes = accumulator_cells
            .checked_mul(std::mem::size_of::<i32>())
            .context("sm_75 accumulator shared-memory size overflow")?;
        let shared_bytes = accumulator_bytes;
        ensure!(
            shared_bytes <= u32::MAX as usize,
            "sm_75 shared-memory request exceeds CUDA launch limit"
        );
        ensure!(
            shared_bytes <= i32::MAX as usize,
            "sm_75 shared-memory request exceeds CUDA function-attribute limit"
        );
        ensure!(
            shared_bytes <= self.max_dynamic_shared_memory,
            "sm_75 tile requests {shared_bytes} bytes, device allows {} bytes",
            self.max_dynamic_shared_memory
        );

        // Request the accumulator scratch size, bounded by the device limit above.
        unsafe {
            result::function::set_function_attribute(
                self.kernel.function,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )
        }
        .context("configuring sm_75 dynamic shared memory")?;

        let stream = self.context.default_stream();
        let d_a = stream
            .clone_htod(a)
            .context("copying padded A tile to CUDA")?;
        let d_bt = stream
            .clone_htod(bt)
            .context("copying padded B^T tile to CUDA")?;
        let mut d_transcript = stream
            .alloc_zeros::<u32>(16 * batch)
            .context("allocating sm_75 transcript")?;
        let mut d_digest = stream
            .alloc_zeros::<u8>(32 * batch)
            .context("allocating sm_75 digest")?;
        let d_a_noise_seed = stream
            .clone_htod(a_noise_seed)
            .context("copying sm_75 A noise seed to CUDA")?;
        (|| -> Result<(Vec<u32>, Vec<u8>)> {
            {
                let (a_ptr, _a_sync) = d_a.device_ptr(&stream);
                let (bt_ptr, _bt_sync) = d_bt.device_ptr(&stream);
                let (transcript_ptr, _transcript_sync) = d_transcript.device_ptr_mut(&stream);
                let (a_noise_seed_ptr, _a_noise_seed_sync) = d_a_noise_seed.device_ptr(&stream);
                let (digest_ptr, _digest_sync) = d_digest.device_ptr_mut(&stream);
                let null_pointer: sys::CUdeviceptr = 0;
                let resident = 0u32;
                let mut params: [*mut c_void; 15] = [
                    (&a_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&bt_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&logical_h as *const u32).cast_mut().cast(),
                    (&logical_w as *const u32).cast_mut().cast(),
                    (&padded_h as *const u32).cast_mut().cast(),
                    (&padded_w as *const u32).cast_mut().cast(),
                    (&physical_k as *const u32).cast_mut().cast(),
                    (&effective_k as *const u32).cast_mut().cast(),
                    (&rank as *const u32).cast_mut().cast(),
                    (&transcript_ptr as *const sys::CUdeviceptr)
                        .cast_mut()
                        .cast(),
                    (&a_noise_seed_ptr as *const sys::CUdeviceptr)
                        .cast_mut()
                        .cast(),
                    (&digest_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    param(&null_pointer),
                    param(&null_pointer),
                    param(&resident),
                ];
                unsafe {
                    result::launch_kernel(
                        self.kernel.function,
                        (batch as u32, 1, 1),
                        (threads as u32, 1, 1),
                        shared_bytes as u32,
                        stream.cu_stream(),
                        &mut params,
                    )
                }
                .context("launching pearl_sm75_mma_tile")?;
            }
            stream
                .synchronize()
                .context("synchronizing pearl_sm75_mma_tile")?;
            let transcript = stream
                .clone_dtoh(&d_transcript)
                .context("copying sm_75 transcript from CUDA")?;
            let digest = stream
                .clone_dtoh(&d_digest)
                .context("copying sm_75 digest from CUDA")?;
            Ok((transcript, digest))
        })()
    }
}

impl GemmKernel for MmaSm75CudaKernel {
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            architecture: KernelArchitecture::Sm75Mma,
            precision: KernelPrecision::Int7Int7Int32,
            certificate_version: 3,
        }
    }

    fn prepare_job(&self, problem: &Problem) -> Result<()> {
        // Drop the previous job before allocating the next, bounding VRAM across job changes.
        self.resident.borrow_mut().take();
        *self.resident.borrow_mut() = Some(self.prepare_resident(problem)?);
        Ok(())
    }
    fn batch_size(&self) -> usize {
        if self.resident.borrow().is_some() {
            1024
        } else {
            128
        }
    }

    fn compute_tile(&self, problem: &Problem, tile: &TileResult) -> Result<TileResult> {
        Ok(self
            .compute_tiles(problem, std::slice::from_ref(tile))?
            .remove(0))
    }

    fn compute_tiles(&self, problem: &Problem, tiles: &[TileResult]) -> Result<Vec<TileResult>> {
        if tiles.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(resident) = self.resident.borrow_mut().as_mut() {
            if resident.key == problem.commitment().a_noise_seed
                && resident.shape == (problem.m, problem.n, problem.k(), problem.config.rank)
            {
                return self.compute_resident(problem, tiles, resident);
            }
        }
        ensure!(
            tiles.len() <= 128,
            "sm_75 unprepared batch exceeds 128 tiles"
        );
        let first = &tiles[0];
        ensure!(
            tiles
                .iter()
                .all(|t| t.rows.len() == first.rows.len() && t.cols.len() == first.cols.len()),
            "sm_75 batch tile shapes differ"
        );
        let operands = problem.noised_tiles_operands(tiles)?;
        let mut a = Vec::new();
        let mut bt = Vec::new();
        let mut padded_h = 0;
        let mut padded_w = 0;
        for (tile, pair) in tiles.iter().zip(operands) {
            let (tile_a, tile_bt, h, w) = Self::padded_operands(problem, tile, pair)?;
            a.extend(tile_a);
            bt.extend(tile_bt);
            padded_h = h;
            padded_w = w;
        }
        let rank = problem.config.rank as usize;
        ensure!(rank > 0, "sm_75 kernel requires a nonzero rank");
        let effective_k = (problem.k() / rank) * rank;
        let commitment = problem.commitment();
        let (transcripts, digests) = self.launch_transcript(
            &a,
            &bt,
            first.rows.len() as u32,
            first.cols.len() as u32,
            padded_h,
            padded_w,
            problem.k() as u32,
            effective_k as u32,
            rank as u32,
            &commitment.a_noise_seed,
            tiles.len(),
        )?;
        tiles
            .iter()
            .enumerate()
            .map(|(index, tile)| {
                let transcript = transcripts[index * 16..(index + 1) * 16]
                    .try_into()
                    .unwrap();
                let digest = digests[index * 32..(index + 1) * 32].try_into().unwrap();
                ensure!(
                    digest == compute_jackpot_hash(&transcript, commitment.a_noise_seed),
                    "sm_75 fused BLAKE3 digest does not match the returned transcript"
                );
                Ok(TileResult {
                    transcript,
                    digest,
                    ..tile.clone()
                })
            })
            .collect()
    }
}

fn param<T>(value: &T) -> *mut c_void {
    (value as *const T).cast_mut().cast()
}

/// Shared resident mining adapter, with a separately compiled ISA for each supported device.
/// The small-tile implementation is retained only for oracle/self-test compatibility.
pub struct ProductionCudaKernel {
    base: MmaSm75CudaKernel,
    scan: LoadedCudaKernel,
    architecture: KernelArchitecture,
    groups_per_launch: u32,
    band: u32,
    scratch: RefCell<ScanScratch>,
}

struct ScanScratch {
    bound: CudaSlice<u32>,
    count: CudaSlice<u32>,
    hits: CudaSlice<u32>,
}

const HIT_CAPACITY: u32 = 256;
const HIT_WORDS: usize = 26;

impl ProductionCudaKernel {
    pub fn new(device: usize) -> Result<Self> {
        Self::from_context(CudaContext::new(device)?)
    }

    pub(crate) fn from_context(context: Arc<CudaContext>) -> Result<Self> {
        let (cubin, architecture, band) = match context.compute_capability()? {
            (7, 5) => (
                embedded_cubin!("PEARL_PRODUCTION_SM75"),
                KernelArchitecture::Sm75Mma,
                4,
            ),
            (9, 0) => (
                embedded_cubin!("PEARL_PRODUCTION_SM90"),
                KernelArchitecture::Sm90Mma,
                8,
            ),
            (12, 0) => (
                embedded_cubin!("PEARL_PRODUCTION_SM120"),
                KernelArchitecture::Sm120Mma,
                8,
            ),
            cc => bail!("unsupported production capability {cc:?}"),
        };
        let cubin = cubin.context("production cubin unavailable; rebuild with CUDA nvcc")?;
        let groups_per_launch = u32::try_from(
            context.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?
        .saturating_mul(if architecture == KernelArchitecture::Sm75Mma { 16 } else { 4 })
        .clamp(32, 4096);
        let scan = load_cuda_kernel(&context, cubin, "pearl_production_scan")?;
        let stream = context.default_stream();
        let scratch = RefCell::new(ScanScratch {
            bound: stream.alloc_zeros::<u32>(8)?,
            count: stream.alloc_zeros::<u32>(1)?,
            hits: stream.alloc_zeros::<u32>(HIT_CAPACITY as usize * HIT_WORDS)?,
        });
        Ok(Self {
            base: MmaSm75CudaKernel::with_cubin(context, cubin)?,
            scan,
            architecture,
            groups_per_launch,
            band,
            scratch,
        })
    }

    fn scan_once(
        &self,
        problem: &Problem,
        begin: u32,
        groups: u32,
        bound: [u8; 32],
    ) -> Result<Option<Vec<TileResult>>> {
        let resident = self.base.resident.borrow();
        let job = resident
            .as_ref()
            .context("prepare_job must precede production scan")?;
        ensure!(
            job.key == problem.commitment().a_noise_seed
                && job.shape == (problem.m, problem.n, problem.k(), problem.config.rank),
            "resident job does not match scan problem"
        );
        let stream = self.base.context.default_stream();
        let mut scratch = self.scratch.borrow_mut();
        let words: Vec<u32> = bound
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        stream.memcpy_htod(&words, &mut scratch.bound)?;
        stream.memset_zeros(&mut scratch.count)?;
        {
            let (a, _a) = job.a.device_ptr(&stream);
            let (bt, _bt) = job.bt.device_ptr(&stream);
            let (seed, _seed) = job.seed.device_ptr(&stream);
            let ScanScratch { bound, count, hits } = &mut *scratch;
            let (target, _target) = bound.device_ptr(&stream);
            let (count, _count) = count.device_ptr_mut(&stream);
            let (hits, _hits) = hits.device_ptr_mut(&stream);
            let k = u32::try_from(problem.k())?;
            let rank = u32::from(problem.config.rank);
            let mut args = [
                param(&a),
                param(&bt),
                param(&problem.m),
                param(&problem.n),
                param(&k),
                param(&rank),
                param(&begin),
                param(&self.band),
                param(&seed),
                param(&target),
                param(&HIT_CAPACITY),
                param(&count),
                param(&hits),
            ];
            unsafe {
                result::launch_kernel(
                    self.scan.function,
                    (groups, 1, 1),
                    (if self.architecture == KernelArchitecture::Sm75Mma { 256 } else { 128 }, 1, 1),
                    0,
                    stream.cu_stream(),
                    &mut args,
                )
            }?;
        }
        let count = stream.clone_dtoh(&scratch.count)?[0];
        if count > HIT_CAPACITY {
            return Ok(None);
        }
        if count == 0 {
            return Ok(Some(Vec::new()));
        }
        let records = stream.clone_dtoh(&scratch.hits.slice(..count as usize * HIT_WORDS))?;
        let mut tiles = Vec::with_capacity(count as usize);
        for record in records.chunks_exact(HIT_WORDS) {
            let mut tile = problem.empty_tile(record[0], record[1])?;
            tile.transcript.copy_from_slice(&record[2..18]);
            for (word, bytes) in record[18..26].iter().zip(tile.digest.chunks_exact_mut(4)) {
                bytes.copy_from_slice(&word.to_le_bytes());
            }
            ensure!(
                tile.digest == compute_jackpot_hash(&tile.transcript, job.key),
                "production candidate digest mismatch"
            );
            ensure!(
                le_digest_leq(&tile.digest, &bound),
                "GPU returned a candidate above target"
            );
            tiles.push(tile);
        }
        // Atomic append order is nondeterministic; stable order simplifies verification and replay.
        tiles.sort_by_key(|t| (t.t_rows, t.t_cols));
        ensure!(
            tiles
                .windows(2)
                .all(|w| (w[0].t_rows, w[0].t_cols) != (w[1].t_rows, w[1].t_cols)),
            "GPU returned duplicate candidates"
        );
        Ok(Some(tiles))
    }
}

fn le_digest_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().rev().cmp(b.iter().rev()) != std::cmp::Ordering::Greater
}

impl GemmKernel for ProductionCudaKernel {
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            architecture: self.architecture,
            precision: KernelPrecision::Int7Int7Int32,
            certificate_version: 3,
        }
    }
    fn prepare_job(&self, p: &Problem) -> Result<()> {
        self.base.prepare_job(p)
    }
    fn batch_size(&self) -> usize {
        self.base.batch_size()
    }
    fn compute_tile(&self, p: &Problem, t: &TileResult) -> Result<TileResult> {
        self.base.compute_tile(p, t)
    }
    fn compute_tiles(&self, p: &Problem, t: &[TileResult]) -> Result<Vec<TileResult>> {
        self.base.compute_tiles(p, t)
    }

    fn mining_groups(&self, p: &Problem) -> Result<Option<(u32, u32)>> {
        use pearl_ref::{KRYPTEX_COLS, KRYPTEX_ROWS};
        ensure!(
            p.m % 128 == 0
                && p.n % 128 == 0
                && p.k() % 64 == 0
                && p.config.rank > 0
                && p.config.rank % 64 == 0
                && p.k() % usize::from(p.config.rank) == 0,
            "production scan requires full 128x128 groups and K/rank multiples of 64"
        );
        ensure!(
            p.config.rows_pattern.indices_with_offset(0) == KRYPTEX_ROWS
                && p.config.cols_pattern.indices_with_offset(0) == KRYPTEX_COLS,
            "production scan requires the V3 8x16 periodic proof pattern"
        );
        let groups = (p.m / 128)
            .checked_mul(p.n / 128)
            .context("group count overflow")?;
        Ok(Some((groups, self.groups_per_launch)))
    }

    fn scan_groups(
        &self,
        p: &Problem,
        begin: u32,
        count: u32,
        bound: [u8; 32],
    ) -> Result<Vec<TileResult>> {
        let (total, limit) = self.mining_groups(p)?.context("group scan unavailable")?;
        ensure!(
            count > 0 && count <= limit && begin.checked_add(count).is_some_and(|end| end <= total),
            "invalid production group range"
        );
        if let Some(tiles) = self.scan_once(p, begin, count, bound)? {
            return Ok(tiles);
        }
        // Never silently lose hits at easy targets. Replay disjoint halves until they fit.
        ensure!(count > 1, "single group overflowed candidate queue");
        let left = count / 2;
        let mut tiles = self.scan_groups(p, begin, left, bound)?;
        tiles.extend(self.scan_groups(p, begin + left, count - left, bound)?);
        Ok(tiles)
    }

    fn mining_selftest(&self) -> Result<()> {
        // Exercise every lane, both staging buffers, cumulative slot wrap, band tails,
        // inclusive target equality, zero-hit filtering, queue replay, and job replacement.
        for (m, n, k) in [(128, 128, 256), (256, 384, 2176), (128, 128, 4096)] {
            let p = production_test_problem(m, n, k)?;
            self.prepare_job(&p)?;
            let groups = (m / 128) * (n / 128);
            let mut tiles = self.scan_groups(&p, 0, groups, [255; 32])?;
            ensure!(
                tiles.len() == groups as usize * 128,
                "production self-test lost tiles"
            );
            tiles.sort_by_key(|t| (t.t_rows, t.t_cols));
            ensure!(
                tiles
                    .windows(2)
                    .all(|w| (w[0].t_rows, w[0].t_cols) != (w[1].t_rows, w[1].t_cols)),
                "production self-test duplicated a tile across groups"
            );
            for tile in &tiles {
                ensure!(
                    *tile == p.tile(tile.t_rows, tile.t_cols)?,
                    "production tile differs from CPU"
                );
            }
            let first_group = self.scan_groups(&p, 0, 1, [255; 32])?;
            let bound = first_group[37].digest;
            let expected: Vec<_> = first_group
                .into_iter()
                .filter(|t| le_digest_leq(&t.digest, &bound))
                .collect();
            ensure!(
                self.scan_groups(&p, 0, 1, bound)? == expected,
                "production target compare mismatch"
            );
            ensure!(
                self.scan_groups(&p, 0, 1, [0; 32])?.is_empty(),
                "production zero target mismatch"
            );
        }
        self.base.resident.borrow_mut().take();
        Ok(())
    }
}

/// Small deterministic fixture with exactly the production proof pattern.
pub fn production_test_problem(m: u32, n: u32, k: u32) -> Result<Problem> {
    use pearl_ref::{MiningConfiguration, PeriodicPattern, KRYPTEX_COLS, KRYPTEX_ROWS};
    let header = pearl_ref::test_problem()?.header;
    let config = MiningConfiguration {
        common_dim: k,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        moe: None,
        rows_pattern: PeriodicPattern::from_list(&KRYPTEX_ROWS)?,
        cols_pattern: PeriodicPattern::from_list(&KRYPTEX_COLS)?,
    };
    let a = (0..m as usize * k as usize)
        .map(|i| (((i * 13 + 7) % 129) as i32 - 64) as i8)
        .collect();
    let bt = (0..n as usize * k as usize)
        .map(|i| (((i * 17 + 3) % 129) as i32 - 64) as i8)
        .collect();
    Problem::new(header, config, m, n, a, bt)
}

struct ResidentJob {
    key: [u8; 32],
    shape: (u32, u32, usize, u16),
    a: CudaSlice<i8>,
    bt: CudaSlice<i8>,
    seed: CudaSlice<u8>,
    rows: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    transcript: CudaSlice<u32>,
    digest: CudaSlice<u8>,
}

impl MmaSm75CudaKernel {
    fn prepare_resident(&self, problem: &Problem) -> Result<ResidentJob> {
        ensure!(
            problem.config.mma_type == MMAType::Int7xInt7ToInt32 && problem.config.moe.is_none(),
            "T4 requires dense V3 INT7 inputs"
        );
        let rank = problem.config.rank as u32;
        ensure!(
            rank >= 32 && rank.is_power_of_two() && problem.k() % 16 == 0,
            "unsupported T4 noise dimensions"
        );
        let stream = self.context.default_stream();
        let cubin = self.cubin;
        let uniform_kernel = load_cuda_kernel(&self.context, cubin, "pearl_sm75_uniform")?;
        let noise_kernel = load_cuda_kernel(&self.context, cubin, "pearl_sm75_apply_noise")?;
        let commitment = problem.commitment();
        let (permutation_a, permutation_b) = problem.noise_permutations();
        let prepare = |matrix: &[i8],
                       seed: &[u8; 32],
                       permutation: &[u32],
                       is_b: u32|
         -> Result<CudaSlice<i8>> {
            let cells = u32::try_from(matrix.len()).context("T4 matrix exceeds u32 indexing")?;
            let k = u32::try_from(problem.k())?;
            let uniform_len = (matrix.len() / problem.k())
                .checked_mul(rank as usize)
                .context("noise size overflow")?;
            let blocks = u32::try_from(uniform_len / 32)?;
            let mut data = stream.clone_htod(matrix)?;
            let mut uniform = stream.alloc_zeros::<i8>(uniform_len)?;
            let d_seed = stream.clone_htod(seed)?;
            let d_permutation = stream.clone_htod(permutation)?;
            {
                let (out, _guard) = uniform.device_ptr_mut(&stream);
                let (key, _key_guard) = d_seed.device_ptr(&stream);
                let mut args = [param(&out), param(&blocks), param(&key), param(&is_b)];
                unsafe {
                    result::launch_kernel(
                        uniform_kernel.function,
                        (blocks.div_ceil(128), 1, 1),
                        (128, 1, 1),
                        0,
                        stream.cu_stream(),
                        &mut args,
                    )
                }?;
            }
            {
                let (out, _guard) = data.device_ptr_mut(&stream);
                let (noise, _noise_guard) = uniform.device_ptr(&stream);
                let (perm, _perm_guard) = d_permutation.device_ptr(&stream);
                let mut args = [
                    param(&out),
                    param(&cells),
                    param(&k),
                    param(&rank),
                    param(&noise),
                    param(&perm),
                ];
                unsafe {
                    result::launch_kernel(
                        noise_kernel.function,
                        (cells.div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        stream.cu_stream(),
                        &mut args,
                    )
                }?;
            }
            stream.synchronize()?;
            Ok(data)
        };
        let a = prepare(&problem.a, &commitment.a_noise_seed, &permutation_a, 0)?;
        let bt = prepare(&problem.bt, &commitment.b_noise_seed, &permutation_b, 1)?;
        let tile = problem.empty_tile(0, 0)?;
        Ok(ResidentJob {
            key: commitment.a_noise_seed,
            shape: (problem.m, problem.n, problem.k(), problem.config.rank),
            a,
            bt,
            seed: stream.clone_htod(&commitment.a_noise_seed)?,
            rows: stream.alloc_zeros::<u32>(1024 * tile.rows.len())?,
            cols: stream.alloc_zeros::<u32>(1024 * tile.cols.len())?,
            transcript: stream.alloc_zeros::<u32>(1024 * 16)?,
            digest: stream.alloc_zeros::<u8>(1024 * 32)?,
        })
    }

    fn compute_resident(
        &self,
        problem: &Problem,
        tiles: &[TileResult],
        job: &mut ResidentJob,
    ) -> Result<Vec<TileResult>> {
        ensure!(tiles.len() <= 1024, "T4 resident batch exceeds 1024 tiles");
        let h = tiles[0].rows.len() as u32;
        let w = tiles[0].cols.len() as u32;
        for tile in tiles {
            ensure!(
                tile.rows.len() == h as usize && tile.cols.len() == w as usize,
                "inconsistent T4 tile shapes"
            );
            ensure!(
                tile.rows == problem.config.rows_pattern.indices_with_offset(tile.t_rows)
                    && tile.cols == problem.config.cols_pattern.indices_with_offset(tile.t_cols),
                "T4 tile pattern mismatch"
            );
            ensure!(
                tile.rows.iter().all(|&r| r < problem.m)
                    && tile.cols.iter().all(|&c| c < problem.n),
                "T4 tile out of bounds"
            );
        }
        let ph = h.div_ceil(8) * 8;
        let pw = w.div_ceil(8) * 8;
        let warps = (ph / 8) * (pw / 8);
        let threads = warps * 32;
        let shared = h
            .checked_mul(w)
            .and_then(|x| x.checked_mul(4))
            .context("T4 shared size overflow")?;
        ensure!(
            warps > 0
                && warps <= 32
                && threads as usize <= self.max_threads_per_block
                && threads as usize <= self.max_block_dim_x
                && shared as usize <= self.max_dynamic_shared_memory,
            "T4 tile exceeds launch limits"
        );
        let k = problem.k() as u32;
        let rank = problem.config.rank as u32;
        let effective = k / rank * rank;
        ensure!(effective > 0, "T4 effective K is zero");
        let rows: Vec<u32> = tiles.iter().flat_map(|t| t.rows.iter().copied()).collect();
        let cols: Vec<u32> = tiles.iter().flat_map(|t| t.cols.iter().copied()).collect();
        let stream = self.context.default_stream();
        stream.memcpy_htod(&rows, &mut job.rows.slice_mut(..rows.len()))?;
        stream.memcpy_htod(&cols, &mut job.cols.slice_mut(..cols.len()))?;
        unsafe {
            result::function::set_function_attribute(
                self.kernel.function,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared as i32,
            )
        }?;
        {
            let (a, _ag) = job.a.device_ptr(&stream);
            let (bt, _bg) = job.bt.device_ptr(&stream);
            let (seed, _sg) = job.seed.device_ptr(&stream);
            let (rows, _rg) = job.rows.device_ptr(&stream);
            let (cols, _cg) = job.cols.device_ptr(&stream);
            let (transcript, _tg) = job.transcript.device_ptr_mut(&stream);
            let (digest, _dg) = job.digest.device_ptr_mut(&stream);
            let resident = 1u32;
            let mut args = [
                param(&a),
                param(&bt),
                param(&h),
                param(&w),
                param(&ph),
                param(&pw),
                param(&k),
                param(&effective),
                param(&rank),
                param(&transcript),
                param(&seed),
                param(&digest),
                param(&rows),
                param(&cols),
                param(&resident),
            ];
            unsafe {
                result::launch_kernel(
                    self.kernel.function,
                    (tiles.len() as u32, 1, 1),
                    (threads, 1, 1),
                    shared,
                    stream.cu_stream(),
                    &mut args,
                )
            }?;
        }
        let transcripts = stream.clone_dtoh(&job.transcript.slice(..tiles.len() * 16))?;
        let digests = stream.clone_dtoh(&job.digest.slice(..tiles.len() * 32))?;
        tiles
            .iter()
            .enumerate()
            .map(|(index, tile)| {
                let transcript = transcripts[index * 16..(index + 1) * 16]
                    .try_into()
                    .unwrap();
                let digest = digests[index * 32..(index + 1) * 32].try_into().unwrap();
                ensure!(
                    digest == compute_jackpot_hash(&transcript, job.key),
                    "T4 resident digest mismatch"
                );
                Ok(TileResult {
                    transcript,
                    digest,
                    ..tile.clone()
                })
            })
            .collect()
    }
}
