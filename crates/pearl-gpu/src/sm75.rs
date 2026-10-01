//! T4/Turing host adapter for the dedicated m8n8k16 Tensor Core kernel.
use crate::{
    load_cuda_kernel, GemmKernel, KernelArchitecture, KernelMetadata, KernelPrecision,
    LoadedCudaKernel,
};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{result, sys, CudaContext, DevicePtr, DevicePtrMut};
use pearl_ref::{compute_jackpot_hash, Problem, TileResult};
use std::{ffi::c_void, sync::Arc};
const SM75_MMA_M: usize = 8;
const SM75_MMA_N: usize = 8;
const SM75_MMA_K: usize = 16;
const SM75_MAX_WARPS: usize = 32;
const SM75_TILE_BATCH_SIZE: usize = 128;

pub struct MmaSm75CudaKernel {
    context: Arc<CudaContext>,
    kernel: LoadedCudaKernel,
    max_threads_per_block: usize,
    max_block_dim_x: usize,
    max_dynamic_shared_memory: usize,
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
        let cubin = embedded_cubin!("PEARL_SM75_MMA_CUBIN")
            .ok_or_else(|| anyhow::anyhow!("sm_75 MMA cubin was not built; install CUDA nvcc"))?;
        let kernel = load_cuda_kernel(&context, cubin, "pearl_sm75_mma_tile")?;
        Ok(Self {
            context,
            kernel,
            max_threads_per_block,
            max_block_dim_x,
            max_dynamic_shared_memory,
        })
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
        tile_count: u32,
        a_noise_seed: &[u8; 32],
    ) -> Result<(Vec<[u32; 16]>, Vec<[u8; 32]>)> {
        ensure!(rank > 0, "sm_75 kernel requires a nonzero rank");
        ensure!(tile_count > 0 && tile_count as usize <= SM75_TILE_BATCH_SIZE,
            "sm_75 tile count is out of range");
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
            .context("sm_75 A operand length overflow")?;
        let expected_bt_len = (padded_w as usize)
            .checked_mul(physical_k as usize)
            .context("sm_75 B^T operand length overflow")?;
        let expected_batch_a = expected_a_len.checked_mul(tile_count as usize)
            .context("sm_75 A batch size overflow")?;
        let expected_batch_bt = expected_bt_len.checked_mul(tile_count as usize)
            .context("sm_75 B^T batch size overflow")?;
        ensure!(a.len() == expected_batch_a, "sm_75 A operand length does not match its batch");
        ensure!(bt.len() == expected_batch_bt, "sm_75 B^T operand length does not match its batch");
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
        // One XOR partial per warp; accumulators remain in registers.
        let accumulator_bytes = (warps as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .context("sm_75 fold shared-memory size overflow")?;
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
            .alloc_zeros::<u32>(16 * tile_count as usize)
            .context("allocating sm_75 transcript")?;
        let mut d_digest = stream
            .alloc_zeros::<u8>(32 * tile_count as usize)
            .context("allocating sm_75 digest")?;
        let d_a_noise_seed = stream
            .clone_htod(a_noise_seed)
            .context("copying sm_75 A noise seed to CUDA")?;
        (|| -> Result<(Vec<[u32; 16]>, Vec<[u8; 32]>)> {
            {
                let (a_ptr, _a_sync) = d_a.device_ptr(&stream);
                let (bt_ptr, _bt_sync) = d_bt.device_ptr(&stream);
                let (transcript_ptr, _transcript_sync) = d_transcript.device_ptr_mut(&stream);
                let (a_noise_seed_ptr, _a_noise_seed_sync) = d_a_noise_seed.device_ptr(&stream);
                let (digest_ptr, _digest_sync) = d_digest.device_ptr_mut(&stream);
                let mut params: [*mut c_void; 13] = [
                    (&a_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&bt_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
                    (&logical_h as *const u32).cast_mut().cast(),
                    (&logical_w as *const u32).cast_mut().cast(),
                    (&padded_h as *const u32).cast_mut().cast(),
                    (&padded_w as *const u32).cast_mut().cast(),
                    (&physical_k as *const u32).cast_mut().cast(),
                    (&effective_k as *const u32).cast_mut().cast(),
                    (&rank as *const u32).cast_mut().cast(),
                    (&tile_count as *const u32).cast_mut().cast(),
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
                        (tile_count, 1, 1),
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
            let transcript_words = stream
                .clone_dtoh(&d_transcript)
                .context("copying sm_75 transcripts from CUDA")?;
            let digests = stream
                .clone_dtoh(&d_digest)
                .context("copying sm_75 digests from CUDA")?;
            let transcripts = transcript_words.chunks_exact(16)
                .map(|words| words.try_into().expect("fixed transcript chunk"))
                .collect();
            let digests = digests.chunks_exact(32)
                .map(|bytes| bytes.try_into().expect("fixed digest chunk"))
                .collect();
            Ok((transcripts, digests))
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

    fn compute_tile(&self, problem: &Problem, tile: &TileResult) -> Result<TileResult> {
        self.compute_tiles(problem, std::slice::from_ref(tile))?
            .pop()
            .context("sm_75 returned an empty tile batch")
    }

    fn max_batch_size(&self) -> usize {
        SM75_TILE_BATCH_SIZE
    }

    fn compute_tiles(&self, problem: &Problem, tiles: &[TileResult]) -> Result<Vec<TileResult>> {
        ensure!(!tiles.is_empty(), "sm_75 tile batch cannot be empty");
        ensure!(tiles.len() <= SM75_TILE_BATCH_SIZE, "sm_75 tile batch is too large");
        let first = &tiles[0];
        let logical_h = first.rows.len();
        let logical_w = first.cols.len();
        let (a, bt, padded_h, padded_w) =
            crate::padded_batch_operands(problem, tiles, SM75_MMA_M, SM75_MMA_N)?;
        let rank = problem.config.rank as usize;
        ensure!(rank > 0, "sm_75 kernel requires a nonzero rank");
        let effective_k = (problem.k() / rank) * rank;
        let commitment = problem.commitment();
        let (transcripts, digests) = self.launch_transcript(
            &a,
            &bt,
            logical_h as u32,
            logical_w as u32,
            padded_h,
            padded_w,
            problem.k() as u32,
            effective_k as u32,
            rank as u32,
            tiles.len() as u32,
            &commitment.a_noise_seed,
        )?;
        tiles.iter().zip(transcripts).zip(digests).map(|((tile, transcript), digest)| {
            let expected_digest = compute_jackpot_hash(&transcript, commitment.a_noise_seed);
            ensure!(digest == expected_digest,
                "sm_75 fused BLAKE3 digest does not match the returned transcript");
            Ok(TileResult {
                t_rows: tile.t_rows,
                t_cols: tile.t_cols,
                rows: tile.rows.clone(),
                cols: tile.cols.clone(),
                transcript,
                digest,
            })
        }).collect()
    }
}
