//! CUDA host bindings for Pearl's resident mining kernels and correctness oracles.
//!
//! Production scans generate noise on CUDA, reuse resident matrices, and return only
//! target hits. Small-tile oracle APIs retain CPU preparation for independent tests.

use std::{
    ffi::{c_void, CString},
    sync::Arc,
};

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{result, sys, CudaContext, DevicePtr, DevicePtrMut};
use pearl_ref::{compute_jackpot_hash, MMAType, Problem, TileResult};

// Embed cubins so installed binaries do not depend on Cargo's build directory.
macro_rules! embedded_cubin {
    ($name:literal) => {{
        #[cfg(pearl_cuda_built)]
        { Some(include_bytes!(env!($name)).as_slice()) }
        #[cfg(not(pearl_cuda_built))]
        { None::<&'static [u8]> }
    }};
}

pub mod m5;
mod sm75;
pub use sm75::MmaSm75CudaKernel;
pub use sm75::ProductionCudaKernel;
pub use sm75::production_test_problem;

const SM120_MMA_M: usize = 16;
const SM120_MMA_N: usize = 8;
const SM120_MMA_K: usize = 32;
const SM120_MAX_WARPS: usize = 32;
const SM120_SWIZZLE_SPAN_BYTES: usize = 128;
const SM120_STAGE_BYTES: usize =
    (SM120_MMA_M + SM120_MMA_N) * SM120_SWIZZLE_SPAN_BYTES;
const SM120_PIPELINE_STAGES: usize = 2;
const SM90_WGMMA_M: usize = 64;
const SM90_WGMMA_N: usize = 8;
const SM90_WGMMA_K: usize = 32;
const SM90_MAX_WARPGROUPS: usize = 8;
const SM90_WGMMA_PIPELINE_STAGES: usize = 2;
const SM90_WGMMA_STAGE_BYTES: usize = (SM90_WGMMA_M + SM90_WGMMA_N) * SM90_WGMMA_K;
const SM90_WGMMA_ACCUM_BYTES: usize = SM90_WGMMA_M * SM90_WGMMA_N * std::mem::size_of::<i32>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelArchitecture {
    Naive,
    Sm120Mma,
    Sm75Mma,
    Sm90Wgmma,
    Sm90Mma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelPrecision {
    Int7Int7Int32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelMetadata {
    pub architecture: KernelArchitecture,
    pub precision: KernelPrecision,
    pub certificate_version: u32,
}

/// Common interface for per-architecture, per-precision GEMM implementations.
pub trait GemmKernel {
    fn device_matrix_preparation(&self) -> bool { false }
    fn metadata(&self) -> KernelMetadata;
    fn compute_tile(&self, problem: &Problem, tile: &TileResult) -> Result<TileResult>;
    fn prepare_job(&self, _problem: &Problem) -> Result<()> { Ok(()) }
    fn batch_size(&self) -> usize { 1 }
    fn compute_tiles(&self, problem: &Problem, tiles: &[TileResult]) -> Result<Vec<TileResult>> {
        tiles.iter().map(|tile| self.compute_tile(problem, tile)).collect()
    }
    /// Number of 128x128 groups and preferred groups per launch, if supported.
    fn mining_groups(&self, _problem: &Problem) -> Result<Option<(u32, u32)>> { Ok(None) }
    /// Returns only target-matching tiles. A successful call covers every tile in the range.
    fn scan_groups(&self, _problem: &Problem, _begin: u32, _count: u32, _bound: [u8; 32]) -> Result<Vec<TileResult>> {
        bail!("grouped mining is unsupported by this kernel")
    }
    fn mining_selftest(&self) -> Result<()> { Ok(()) }
}

/// Selects a production tensor-core implementation from the device capability.
///
/// The naive kernel is deliberately excluded from this function: it is an oracle for tests and
/// must never become the mining hot loop.
pub fn select_production_kernel(device_ordinal: usize) -> Result<Box<dyn GemmKernel>> {
    let context = CudaContext::new(device_ordinal)
        .with_context(|| format!("opening CUDA device {device_ordinal}"))?;
    let capability = context
        .compute_capability()
        .context("querying CUDA compute capability")?;

    match capability {
        (7, 5) | (9, 0) | (12, 0) => Ok(Box::new(ProductionCudaKernel::from_context(context)?)),
        (major, minor) => bail!(
            "no production tensor-core kernel for compute capability {major}.{minor}"
        ),
    }
}

struct LoadedCudaKernel {
    context: Arc<CudaContext>,
    module: sys::CUmodule,
    function: sys::CUfunction,
}

impl Drop for LoadedCudaKernel {
    fn drop(&mut self) {
        // Drop must never panic. The context is kept alive by this guard, and binding it here
        // makes module unload valid even when the worker is dropped on a different host thread.
        if let Err(error) = self.context.bind_to_thread() {
            log::error!(
                target: "pearl_gpu",
                "failed to bind CUDA context while unloading module: {:?}",
                error
            );
            return;
        }
        if let Err(error) = unsafe { result::module::unload(self.module) } {
            log::error!(target: "pearl_gpu", "failed to unload CUDA module: {:?}", error);
        }
    }
}

fn load_cuda_kernel(
    context: &Arc<CudaContext>,
    cubin: &[u8],
    symbol: &str,
) -> Result<LoadedCudaKernel> {
    context.bind_to_thread().context("binding CUDA context")?;
    // SAFETY: cubin is a complete binary image embedded by the build script and
    // remains valid for the duration of cuModuleLoadData.
    let module = unsafe { result::module::load_data(cubin.as_ptr().cast()) }
        .with_context(|| format!("loading embedded CUDA cubin for {symbol}"))?;
    let function = match unsafe { result::module::get_function(module, CString::new(symbol)?) } {
        Ok(function) => function,
        Err(error) => {
            if let Err(unload_error) = unsafe { result::module::unload(module) } {
                log::error!(
                    target: "pearl_gpu",
                    "failed to unload CUDA module after function lookup failed: {:?}",
                    unload_error
                );
            }
            return Err(error).with_context(|| format!("loading CUDA kernel {symbol}"));
        }
    };
    Ok(LoadedCudaKernel {
        context: context.clone(),
        module,
        function,
    })
}

pub struct NaiveCudaKernel {
    context: Arc<CudaContext>,
    kernel: LoadedCudaKernel,
}

impl NaiveCudaKernel {
    /// Creates the M2 oracle on a real CUDA device and selects the matching cubin by capability.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("opening CUDA device {device_ordinal}"))?;
        let capability = context
            .compute_capability()
            .context("querying CUDA compute capability")?;
        let cubin = match capability {
            (7, 5) => embedded_cubin!("PEARL_NAIVE_CUBIN_SM75"),
            (9, 0) => embedded_cubin!("PEARL_NAIVE_CUBIN_SM90"),
            (12, 0) => embedded_cubin!("PEARL_NAIVE_CUBIN_SM120"),
            (major, minor) => bail!("unsupported CUDA compute capability {major}.{minor}"),
        }
        .ok_or_else(|| anyhow::anyhow!("naive cubin was not built; install CUDA nvcc"))?;
        let kernel = load_cuda_kernel(&context, cubin, "pearl_naive_tile")?;
        Ok(Self { context, kernel })
    }

    pub fn device_name(&self) -> Result<String> {
        self.context.name().context("querying CUDA device name")
    }

    fn launch_transcript(
        &self,
        a: &[i8],
        bt: &[i8],
        h: u32,
        w: u32,
        k: u32,
        rank: u32,
    ) -> Result<[u32; 16]> {
        let stream = self.context.default_stream();
        let d_a = stream.clone_htod(a).context("copying A tile to CUDA")?;
        let d_bt = stream.clone_htod(bt).context("copying B^T tile to CUDA")?;
        let mut d_acc = stream
            .alloc_zeros::<i32>(h as usize * w as usize)
            .context("allocating naive CUDA accumulators")?;
        let mut d_transcript = stream
            .alloc_zeros::<u32>(16)
            .context("allocating naive CUDA transcript")?;

        (|| -> Result<[u32; 16]> {
            {
                let (a_ptr, _a_sync) = d_a.device_ptr(&stream);
                let (bt_ptr, _bt_sync) = d_bt.device_ptr(&stream);
                let (acc_ptr, _acc_sync) = d_acc.device_ptr_mut(&stream);
                let (transcript_ptr, _transcript_sync) = d_transcript.device_ptr_mut(&stream);
                let mut params: [*mut c_void; 8] = [
                    (&a_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&bt_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&h as *const u32).cast_mut().cast(),
                    (&w as *const u32).cast_mut().cast(),
                    (&k as *const u32).cast_mut().cast(),
                    (&rank as *const u32).cast_mut().cast(),
                    (&acc_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&transcript_ptr as *const sys::CUdeviceptr)
                        .cast_mut()
                        .cast(),
                ];

                unsafe {
                    result::launch_kernel(
                        self.kernel.function,
                        (1, 1, 1),
                        (1, 1, 1),
                        0,
                        stream.cu_stream(),
                        &mut params,
                    )
                }
                .context("launching pearl_naive_tile")?;
            }
            stream
                .synchronize()
                .context("synchronizing pearl_naive_tile")?;
            let values = stream
                .clone_dtoh(&d_transcript)
                .context("copying naive transcript from CUDA")?;
            values
                .try_into()
                .map_err(|_| anyhow::anyhow!("naive CUDA transcript had the wrong length"))
        })()
    }
}

impl GemmKernel for NaiveCudaKernel {
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            architecture: KernelArchitecture::Naive,
            precision: KernelPrecision::Int7Int7Int32,
            certificate_version: 3,
        }
    }

    fn compute_tile(&self, problem: &Problem, tile: &TileResult) -> Result<TileResult> {
        let (a, bt) = problem.noised_tile_operands(tile)?;
        let transcript = self.launch_transcript(
            &a,
            &bt,
            tile.rows.len() as u32,
            tile.cols.len() as u32,
            problem.k() as u32,
            problem.config.rank as u32,
        )?;
        let digest = compute_jackpot_hash(&transcript, problem.commitment().a_noise_seed);
        Ok(TileResult {
            t_rows: tile.t_rows,
            t_cols: tile.t_cols,
            rows: tile.rows.clone(),
            cols: tile.cols.clone(),
            transcript,
            digest,
        })
    }
}

/// sm_120a INT8 tensor-core implementation.  The host pads only the physical operands to the
/// instruction tile shape; transcript dimensions and all consensus arithmetic remain logical.
pub struct MmaSm120CudaKernel {
    context: Arc<CudaContext>,
    kernel: LoadedCudaKernel,
    max_threads_per_block: usize,
    max_block_dim_x: usize,
    max_dynamic_shared_memory: usize,
}

impl MmaSm120CudaKernel {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("opening CUDA device {device_ordinal}"))?;
        Self::from_context(context)
    }

    pub fn device_name(&self) -> Result<String> {
        self.context.name().context("querying CUDA device name")
    }

    fn from_context(context: Arc<CudaContext>) -> Result<Self> {
        let capability = context
            .compute_capability()
            .context("querying CUDA compute capability")?;
        if capability != (12, 0) {
            bail!(
                "sm_120 MMA kernel requires compute capability 12.0, found {}.{}",
                capability.0,
                capability.1
            );
        }
        let warp_size = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            .context("querying CUDA warp size")?;
        ensure!(
            warp_size == 32,
            "sm_120 MMA kernel requires 32-wide warps, found {warp_size}"
        );
        let max_threads_per_block = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .context("querying CUDA block-thread limit")?;
        let max_threads_per_block = usize::try_from(max_threads_per_block)
            .context("CUDA block-thread limit was negative")?;
        let max_block_dim_x = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
            .context("querying CUDA block X-dimension limit")?;
        let max_block_dim_x =
            usize::try_from(max_block_dim_x).context("CUDA block X-dimension limit was negative")?;
        let max_dynamic_shared_memory = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .context("querying sm_120 opt-in shared-memory limit")?;
        let max_dynamic_shared_memory = usize::try_from(max_dynamic_shared_memory)
            .context("sm_120 shared-memory limit was negative")?;
        let cubin = embedded_cubin!("PEARL_SM120_MMA_CUBIN")
            .ok_or_else(|| anyhow::anyhow!("sm_120 MMA cubin was not built; install CUDA nvcc"))?;
        let kernel = load_cuda_kernel(&context, cubin, "pearl_sm120_mma_tile")?;
        Ok(Self {
            context,
            kernel,
            max_threads_per_block,
            max_block_dim_x,
            max_dynamic_shared_memory,
        })
    }

    fn padded_operands(
        problem: &Problem,
        tile: &TileResult,
    ) -> Result<(Vec<i8>, Vec<i8>, u32, u32)> {
        ensure!(
            problem.config.mma_type == MMAType::Int7xInt7ToInt32,
            "sm_120 kernel only supports the official V3 INT8 representation"
        );
        ensure!(
            problem.config.moe.is_none(),
            "sm_120 dense kernel does not support grouped GEMM"
        );
        let (a, bt) = problem.noised_tile_operands(tile)?;
        let h = tile.rows.len();
        let w = tile.cols.len();
        let k = problem.k();
        ensure!(h > 0 && w > 0, "sm_120 kernel requires a non-empty tile");
        ensure!(h <= u32::MAX as usize, "sm_120 row count exceeds CUDA parameter width");
        ensure!(w <= u32::MAX as usize, "sm_120 column count exceeds CUDA parameter width");
        let padded_h = h
            .checked_add(SM120_MMA_M - 1)
            .context("sm_120 padded row count overflow")?
            / SM120_MMA_M;
        let padded_h = padded_h
            .checked_mul(SM120_MMA_M)
            .context("sm_120 padded row count overflow")?;
        let padded_w = w
            .checked_add(SM120_MMA_N - 1)
            .context("sm_120 padded column count overflow")?
            / SM120_MMA_N;
        let padded_w = padded_w
            .checked_mul(SM120_MMA_N)
            .context("sm_120 padded column count overflow")?;
        ensure!(padded_h <= u32::MAX as usize, "sm_120 padded rows exceed CUDA parameter width");
        ensure!(padded_w <= u32::MAX as usize, "sm_120 padded columns exceed CUDA parameter width");
        let padded_a_len = padded_h
            .checked_mul(k)
            .context("sm_120 padded A size overflow")?;
        let padded_bt_len = padded_w
            .checked_mul(k)
            .context("sm_120 padded B^T size overflow")?;
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
    ) -> Result<([u32; 16], [u8; 32])> {
        ensure!(rank > 0, "sm_120 kernel requires a nonzero rank");
        ensure!(logical_h > 0 && logical_w > 0, "sm_120 kernel requires a non-empty tile");
        ensure!(padded_h >= logical_h && padded_w >= logical_w, "sm_120 padded tile is too small");
        ensure!(
            padded_h as usize % SM120_MMA_M == 0 && padded_w as usize % SM120_MMA_N == 0,
            "sm_120 padded tile is misaligned"
        );
        let expected_a_len = (padded_h as usize)
            .checked_mul(physical_k as usize)
            .context("sm_120 A operand length overflow")?;
        let expected_bt_len = (padded_w as usize)
            .checked_mul(physical_k as usize)
            .context("sm_120 B^T operand length overflow")?;
        ensure!(a.len() == expected_a_len, "sm_120 A operand length does not match its stride");
        ensure!(bt.len() == expected_bt_len, "sm_120 B^T operand length does not match its stride");
        let tiles_m = (padded_h as usize).div_ceil(SM120_MMA_M);
        let tiles_n = (padded_w as usize).div_ceil(SM120_MMA_N);
        let warps = tiles_m
            .checked_mul(tiles_n)
            .context("sm_120 tile warp count overflow")?;
        ensure!(
            warps > 0 && warps <= SM120_MAX_WARPS,
            "sm_120 tile requires {warps} warps"
        );
        let threads = (warps as usize)
            .checked_mul(32)
            .context("sm_120 block-thread count overflow")?;
        ensure!(
            threads <= self.max_threads_per_block,
            "sm_120 tile requests {threads} threads, device allows {}",
            self.max_threads_per_block
        );
        ensure!(
            threads <= self.max_block_dim_x,
            "sm_120 1D block requests {threads} threads on X, device allows {}",
            self.max_block_dim_x
        );
        ensure!(
            rank % (SM120_MMA_K as u32) == 0
                && physical_k % (SM120_MMA_K as u32) == 0
                && effective_k % (SM120_MMA_K as u32) == 0,
            "sm_120 kernel requires 32-aligned K dimensions"
        );
        ensure!(
            effective_k <= physical_k && effective_k / rank * rank == effective_k,
            "sm_120 kernel requires a rank-aligned effective K"
        );
        ensure!(effective_k > 0, "sm_120 kernel requires a non-empty effective K");
        // Each warp owns two pipeline stages. Every logical row reserves a 128-byte
        // software-swizzled span, matching the CUDA kernel's 16-byte chunk addresses.
        let stage_storage = (warps as usize)
            .checked_mul(SM120_PIPELINE_STAGES)
            .and_then(|value| value.checked_mul(SM120_STAGE_BYTES))
            .context("sm_120 pipeline shared-memory size overflow")?;
        let accumulator_cells = (logical_h as usize)
            .checked_mul(logical_w as usize)
            .context("sm_120 accumulator cell count overflow")?;
        let accumulator_bytes = accumulator_cells
            .checked_mul(std::mem::size_of::<i32>())
            .context("sm_120 accumulator shared-memory size overflow")?;
        let shared_bytes = stage_storage
            .checked_add(accumulator_bytes)
            .context("sm_120 shared-memory size overflow")?;
        ensure!(
            shared_bytes <= u32::MAX as usize,
            "sm_120 shared-memory request exceeds CUDA launch limit"
        );
        ensure!(
            shared_bytes <= i32::MAX as usize,
            "sm_120 shared-memory request exceeds CUDA function-attribute limit"
        );
        ensure!(
            shared_bytes <= self.max_dynamic_shared_memory,
            "sm_120 tile requests {shared_bytes} bytes, device allows {} bytes",
            self.max_dynamic_shared_memory
        );

        // The 128B software-swizzled stages are intentionally larger than CUDA's legacy
        // 48 KiB default for some tile shapes. Request the exact dynamic shared-memory size
        // through the driver API; devices that cannot provide it return a checked launch error.
        unsafe {
            result::function::set_function_attribute(
                self.kernel.function,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )
        }
        .context("configuring sm_120 dynamic shared memory")?;

        let stream = self.context.default_stream();
        let d_a = stream
            .clone_htod(a)
            .context("copying padded A tile to CUDA")?;
        let d_bt = stream
            .clone_htod(bt)
            .context("copying padded B^T tile to CUDA")?;
        let mut d_transcript = stream
            .alloc_zeros::<u32>(16)
            .context("allocating sm_120 transcript")?;
        let mut d_digest = stream
            .alloc_zeros::<u8>(32)
            .context("allocating sm_120 digest")?;
        let d_a_noise_seed = stream
            .clone_htod(a_noise_seed)
            .context("copying sm_120 A noise seed to CUDA")?;
        (|| -> Result<([u32; 16], [u8; 32])> {
            {
                let (a_ptr, _a_sync) = d_a.device_ptr(&stream);
                let (bt_ptr, _bt_sync) = d_bt.device_ptr(&stream);
                let (transcript_ptr, _transcript_sync) = d_transcript.device_ptr_mut(&stream);
                let (a_noise_seed_ptr, _a_noise_seed_sync) = d_a_noise_seed.device_ptr(&stream);
                let (digest_ptr, _digest_sync) = d_digest.device_ptr_mut(&stream);
                let mut params: [*mut c_void; 12] = [
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
                ];
                unsafe {
                    result::launch_kernel(
                        self.kernel.function,
                        (1, 1, 1),
                        (threads as u32, 1, 1),
                        shared_bytes as u32,
                        stream.cu_stream(),
                        &mut params,
                    )
                }
                .context("launching pearl_sm120_mma_tile")?;
            }
            stream
                .synchronize()
                .context("synchronizing pearl_sm120_mma_tile")?;
            let transcript = stream
                .clone_dtoh(&d_transcript)
                .context("copying sm_120 transcript from CUDA")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("sm_120 transcript had the wrong length"))?;
            let digest = stream
                .clone_dtoh(&d_digest)
                .context("copying sm_120 digest from CUDA")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("sm_120 digest had the wrong length"))?;
            Ok((transcript, digest))
        })()
    }
}

impl GemmKernel for MmaSm120CudaKernel {
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            architecture: KernelArchitecture::Sm120Mma,
            precision: KernelPrecision::Int7Int7Int32,
            certificate_version: 3,
        }
    }

    fn compute_tile(&self, problem: &Problem, tile: &TileResult) -> Result<TileResult> {
        let (a, bt, padded_h, padded_w) = Self::padded_operands(problem, tile)?;
        let rank = problem.config.rank as usize;
        ensure!(rank > 0, "sm_120 kernel requires a nonzero rank");
        let effective_k = (problem.k() / rank) * rank;
        let commitment = problem.commitment();
        let (transcript, digest) = self.launch_transcript(
            &a,
            &bt,
            tile.rows.len() as u32,
            tile.cols.len() as u32,
            padded_h,
            padded_w,
            problem.k() as u32,
            effective_k as u32,
            rank as u32,
            &commitment.a_noise_seed,
        )?;
        let expected_digest = compute_jackpot_hash(&transcript, commitment.a_noise_seed);
        ensure!(
            digest == expected_digest,
            "sm_120 fused BLAKE3 digest does not match the returned transcript"
        );
        Ok(TileResult {
            t_rows: tile.t_rows,
            t_cols: tile.t_cols,
            rows: tile.rows.clone(),
            cols: tile.cols.clone(),
            transcript,
            digest,
        })
    }
}

/// sm_90a Hopper implementation. One CTA uses a dedicated producer warp and up to eight
/// four-warp consumer warpgroups, each of which owns a 64x8 tile. The CUDA kernel stages K32
/// panels with TMA bulk copies and consumes them with `wgmma.mma_async`; the host keeps the same
/// V3 operand and transcript contract as sm_120.
pub struct WgmmaSm90CudaKernel {
    context: Arc<CudaContext>,
    kernel: LoadedCudaKernel,
    max_threads_per_block: usize,
    max_block_dim_x: usize,
    max_dynamic_shared_memory: usize,
}

impl WgmmaSm90CudaKernel {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("opening CUDA device {device_ordinal}"))?;
        Self::from_context(context)
    }

    pub fn device_name(&self) -> Result<String> {
        self.context.name().context("querying CUDA device name")
    }

    fn from_context(context: Arc<CudaContext>) -> Result<Self> {
        let capability = context
            .compute_capability()
            .context("querying CUDA compute capability")?;
        ensure!(
            capability == (9, 0),
            "sm_90a WGMMA kernel requires compute capability 9.0, found {}.{}",
            capability.0,
            capability.1
        );
        let warp_size = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
            .context("querying CUDA warp size")?;
        ensure!(warp_size == 32, "sm_90a WGMMA requires 32-wide warps, found {warp_size}");
        let max_threads_per_block = usize::try_from(
            context
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                .context("querying CUDA block-thread limit")?,
        )
        .context("CUDA block-thread limit was negative")?;
        let max_block_dim_x = usize::try_from(
            context
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)
                .context("querying CUDA block X-dimension limit")?,
        )
        .context("CUDA block X-dimension limit was negative")?;
        let max_dynamic_shared_memory = usize::try_from(
            context
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
                .context("querying sm_90 opt-in shared-memory limit")?,
        )
        .context("sm_90 shared-memory limit was negative")?;
        let cubin = embedded_cubin!("PEARL_SM90_WGMMA_CUBIN")
            .ok_or_else(|| anyhow::anyhow!("sm_90 WGMMA cubin was not built; install CUDA nvcc"))?;
        let kernel = load_cuda_kernel(&context, cubin, "pearl_sm90_wgmma_tile")?;
        Ok(Self {
            context,
            kernel,
            max_threads_per_block,
            max_block_dim_x,
            max_dynamic_shared_memory,
        })
    }

    fn padded_operands(
        problem: &Problem,
        tile: &TileResult,
    ) -> Result<(Vec<i8>, Vec<i8>, u32, u32)> {
        ensure!(
            problem.config.mma_type == MMAType::Int7xInt7ToInt32,
            "sm_90 WGMMA only supports the official V3 INT8 representation"
        );
        ensure!(
            problem.config.moe.is_none(),
            "sm_90 WGMMA dense kernel does not support grouped GEMM"
        );
        let (a, bt) = problem.noised_tile_operands(tile)?;
        let h = tile.rows.len();
        let w = tile.cols.len();
        let k = problem.k();
        ensure!(h > 0 && w > 0, "sm_90 WGMMA requires a non-empty tile");
        let padded_h = h
            .checked_add(SM90_WGMMA_M - 1)
            .context("sm_90 padded row count overflow")?
            / SM90_WGMMA_M
            * SM90_WGMMA_M;
        let padded_w = w
            .checked_add(SM90_WGMMA_N - 1)
            .context("sm_90 padded column count overflow")?
            / SM90_WGMMA_N
            * SM90_WGMMA_N;
        ensure!(padded_h <= u32::MAX as usize, "sm_90 padded rows exceed CUDA parameter width");
        ensure!(padded_w <= u32::MAX as usize, "sm_90 padded columns exceed CUDA parameter width");
        let mut padded_a = vec![0i8; padded_h.checked_mul(k).context("sm_90 A size overflow")?];
        let mut padded_bt = vec![0i8; padded_w.checked_mul(k).context("sm_90 B size overflow")?];
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
    ) -> Result<([u32; 16], [u8; 32])> {
        ensure!(rank > 0, "sm_90 WGMMA requires a nonzero rank");
        ensure!(logical_h > 0 && logical_w > 0, "sm_90 WGMMA requires a non-empty tile");
        ensure!(padded_h >= logical_h && padded_w >= logical_w, "sm_90 padded tile is too small");
        ensure!(
            padded_h as usize % SM90_WGMMA_M == 0 && padded_w as usize % SM90_WGMMA_N == 0,
            "sm_90 padded tile is misaligned"
        );
        let expected_a_len = (padded_h as usize)
            .checked_mul(physical_k as usize)
            .context("sm_90 A operand length overflow")?;
        let expected_bt_len = (padded_w as usize)
            .checked_mul(physical_k as usize)
            .context("sm_90 B operand length overflow")?;
        ensure!(a.len() == expected_a_len, "sm_90 A operand length does not match its stride");
        ensure!(bt.len() == expected_bt_len, "sm_90 B operand length does not match its stride");
        let tiles_m = padded_h as usize / SM90_WGMMA_M;
        let tiles_n = padded_w as usize / SM90_WGMMA_N;
        let warpgroups = tiles_m
            .checked_mul(tiles_n)
            .context("sm_90 warpgroup count overflow")?;
        ensure!(
            warpgroups > 0 && warpgroups <= SM90_MAX_WARPGROUPS,
            "sm_90 tile requires {warpgroups} warpgroups"
        );
        let threads = warpgroups
            .checked_mul(128)
            .and_then(|v| v.checked_add(32))
            .context("sm_90 block-thread count overflow")?;
        ensure!(threads <= self.max_threads_per_block, "sm_90 tile requests {threads} threads");
        ensure!(threads <= self.max_block_dim_x, "sm_90 tile exceeds the block X dimension");
        ensure!(
            rank % SM90_WGMMA_K as u32 == 0
                && physical_k % SM90_WGMMA_K as u32 == 0
                && effective_k % SM90_WGMMA_K as u32 == 0,
            "sm_90 WGMMA requires 32-aligned K dimensions"
        );
        ensure!(effective_k > 0 && effective_k <= physical_k, "sm_90 effective K is invalid");
        ensure!(effective_k / rank * rank == effective_k, "sm_90 effective K is not rank-aligned");
        let barrier_bytes = warpgroups
            .checked_mul(SM90_WGMMA_PIPELINE_STAGES)
            .and_then(|v| v.checked_mul(std::mem::size_of::<u64>()))
            .context("sm_90 barrier storage overflow")?;
        let stage_bytes = warpgroups
            .checked_mul(SM90_WGMMA_PIPELINE_STAGES)
            .and_then(|v| v.checked_mul(SM90_WGMMA_STAGE_BYTES))
            .context("sm_90 pipeline storage overflow")?;
        let accumulator_bytes = warpgroups
            .checked_mul(SM90_WGMMA_ACCUM_BYTES)
            .context("sm_90 accumulator storage overflow")?;
        let shared_bytes = barrier_bytes
            .checked_mul(2)
            .and_then(|v| v.checked_add(stage_bytes))
            .and_then(|v| v.checked_add(accumulator_bytes))
            .context("sm_90 shared-memory size overflow")?;
        ensure!(shared_bytes <= u32::MAX as usize, "sm_90 shared-memory request exceeds CUDA width");
        ensure!(shared_bytes <= i32::MAX as usize, "sm_90 shared-memory request exceeds attribute width");
        ensure!(
            shared_bytes <= self.max_dynamic_shared_memory,
            "sm_90 tile requests {shared_bytes} bytes, device allows {}",
            self.max_dynamic_shared_memory
        );
        unsafe {
            result::function::set_function_attribute(
                self.kernel.function,
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )
        }
        .context("configuring sm_90 dynamic shared memory")?;

        let stream = self.context.default_stream();
        let d_a = stream.clone_htod(a).context("copying sm_90 A tile to CUDA")?;
        let d_bt = stream.clone_htod(bt).context("copying sm_90 B tile to CUDA")?;
        let mut d_transcript = stream.alloc_zeros::<u32>(16).context("allocating sm_90 transcript")?;
        let mut d_digest = stream.alloc_zeros::<u8>(32).context("allocating sm_90 digest")?;
        let d_a_noise_seed = stream
            .clone_htod(a_noise_seed)
            .context("copying sm_90 A noise seed to CUDA")?;
        (|| -> Result<([u32; 16], [u8; 32])> {
            let (a_ptr, _a_sync) = d_a.device_ptr(&stream);
            let (bt_ptr, _bt_sync) = d_bt.device_ptr(&stream);
            let (transcript_ptr, _transcript_sync) = d_transcript.device_ptr_mut(&stream);
            let (seed_ptr, _seed_sync) = d_a_noise_seed.device_ptr(&stream);
            let (digest_ptr, _digest_sync) = d_digest.device_ptr_mut(&stream);
            let mut params: [*mut c_void; 12] = [
                (&a_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                (&bt_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                (&logical_h as *const u32).cast_mut().cast(),
                (&logical_w as *const u32).cast_mut().cast(),
                (&padded_h as *const u32).cast_mut().cast(),
                (&padded_w as *const u32).cast_mut().cast(),
                (&physical_k as *const u32).cast_mut().cast(),
                (&effective_k as *const u32).cast_mut().cast(),
                (&rank as *const u32).cast_mut().cast(),
                (&transcript_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                (&seed_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                (&digest_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
            ];
            unsafe {
                result::launch_kernel(
                    self.kernel.function,
                    (1, 1, 1),
                    (threads as u32, 1, 1),
                    shared_bytes as u32,
                    stream.cu_stream(),
                    &mut params,
                )
            }
            .context("launching pearl_sm90_wgmma_tile")?;
            stream.synchronize().context("synchronizing pearl_sm90_wgmma_tile")?;
            drop(_transcript_sync);
            drop(_digest_sync);
            let transcript = stream
                .clone_dtoh(&d_transcript)
                .context("copying sm_90 transcript from CUDA")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("sm_90 transcript had the wrong length"))?;
            let digest = stream
                .clone_dtoh(&d_digest)
                .context("copying sm_90 digest from CUDA")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("sm_90 digest had the wrong length"))?;
            Ok((transcript, digest))
        })()
    }
}

impl GemmKernel for WgmmaSm90CudaKernel {
    fn metadata(&self) -> KernelMetadata {
        KernelMetadata {
            architecture: KernelArchitecture::Sm90Wgmma,
            precision: KernelPrecision::Int7Int7Int32,
            certificate_version: 3,
        }
    }

    fn compute_tile(&self, problem: &Problem, tile: &TileResult) -> Result<TileResult> {
        let (a, bt, padded_h, padded_w) = Self::padded_operands(problem, tile)?;
        let rank = problem.config.rank as usize;
        ensure!(rank > 0, "sm_90 WGMMA requires a nonzero rank");
        let effective_k = (problem.k() / rank) * rank;
        let commitment = problem.commitment();
        let (transcript, digest) = self.launch_transcript(
            &a,
            &bt,
            tile.rows.len() as u32,
            tile.cols.len() as u32,
            padded_h,
            padded_w,
            problem.k() as u32,
            effective_k as u32,
            rank as u32,
            &commitment.a_noise_seed,
        )?;
        let expected_digest = compute_jackpot_hash(&transcript, commitment.a_noise_seed);
        ensure!(digest == expected_digest, "sm_90 fused BLAKE3 digest does not match the transcript");
        Ok(TileResult {
            t_rows: tile.t_rows,
            t_cols: tile.t_cols,
            rows: tile.rows.clone(),
            cols: tile.cols.clone(),
            transcript,
            digest,
        })
    }
}
