//! M5 job setup and device orchestration.
//!
//! The CUDA hash kernels mirror the official `pearl-blake3` chunk/tree rules. The host
//! orchestration keeps matrix bytes and intermediate chaining values on the device until the
//! final 32-byte root is copied back. No candidate share is produced by this module.

use std::{
    ffi::c_void,
    fs,
    path::Path,
    process::Command,
    sync::{atomic::{AtomicBool, Ordering}, Arc},
};

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{
    result, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
};
use pearl_ref::{bind_root_a, bind_root_b, Commitment, Problem};
use serde::{Deserialize, Serialize};

use crate::{load_cuda_kernel, LoadedCudaKernel};

const BLAKE3_CHUNK_BYTES: usize = 1024;
const BLAKE3_DIGEST_BYTES: usize = 32;
const HASH_BLOCK_THREADS: u32 = 256;

/// CUDA-resident keyed BLAKE3 chunk/Merkle reducer used during job setup.
pub struct TensorHashCudaKernel {
    context: Arc<CudaContext>,
    leaf: LoadedCudaKernel,
    parent: LoadedCudaKernel,
}

impl TensorHashCudaKernel {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("opening CUDA device {device_ordinal}"))?;
        Self::from_context(context)
    }

    pub(crate) fn from_context(context: Arc<CudaContext>) -> Result<Self> {
        let capability = context.compute_capability().context("querying CUDA capability")?;
        let cubin = match capability {
            (7, 5) => embedded_cubin!("PEARL_TENSOR_HASH_CUBIN_SM75"),
            (9, 0) => embedded_cubin!("PEARL_TENSOR_HASH_CUBIN_SM90"),
            (12, 0) => embedded_cubin!("PEARL_TENSOR_HASH_CUBIN_SM120"),
            (major, minor) => bail!(
                "no tensor_hash cubin for compute capability {major}.{minor}"
            ),
        }
        .ok_or_else(|| anyhow::anyhow!("tensor_hash cubin was not built; install CUDA nvcc"))?;
        let leaf = load_cuda_kernel(&context, cubin, "pearl_tensor_hash_leaf_cv")?;
        let parent = load_cuda_kernel(&context, cubin, "pearl_tensor_hash_parent_cv")?;
        Ok(Self { context, leaf, parent })
    }

    pub fn device_name(&self) -> Result<String> {
        self.context.name().context("querying tensor_hash device name")
    }

    /// Hashes a non-empty 1024-byte-aligned byte string using the official keyed Merkle rules.
    pub fn hash_padded(&self, data: &[u8], key: [u8; 32]) -> Result<[u8; 32]> {
        ensure!(!data.is_empty(), "tensor_hash input must be non-empty");
        ensure!(
            data.len().is_multiple_of(BLAKE3_CHUNK_BYTES),
            "tensor_hash input must be padded to 1024 bytes"
        );
        let chunk_count = data.len() / BLAKE3_CHUNK_BYTES;
        ensure!(chunk_count <= u32::MAX as usize, "tensor_hash input has too many chunks");

        let stream = self.context.new_stream().context("creating tensor_hash stream")?;
        let d_data = stream.clone_htod(data).context("uploading tensor_hash input")?;
        let d_key = stream.clone_htod(&key).context("uploading tensor_hash key")?;
        let mut d_level = stream
            .alloc_zeros::<u8>(chunk_count * BLAKE3_DIGEST_BYTES)
            .context("allocating tensor_hash leaf level")?;
        self.launch_leaf(
            &stream,
            &d_data,
            &d_key,
            chunk_count as u32,
            (chunk_count == 1) as u32,
            &mut d_level,
        )?;

        let mut level_count = chunk_count;
        while level_count > 2 {
            let next_count = level_count.div_ceil(2);
            let mut d_next = stream
                .alloc_zeros::<u8>(next_count * BLAKE3_DIGEST_BYTES)
                .context("allocating tensor_hash parent level")?;
            self.launch_parent(
                &stream,
                &d_level,
                &d_key,
                level_count as u32,
                false,
                &mut d_next,
            )?;
            d_level = d_next;
            level_count = next_count;
        }

        if level_count == 2 {
            let mut d_root = stream
                .alloc_zeros::<u8>(BLAKE3_DIGEST_BYTES)
                .context("allocating tensor_hash root")?;
            self.launch_parent(
                &stream,
                &d_level,
                &d_key,
                2,
                true,
                &mut d_root,
            )?;
            stream.synchronize().context("synchronizing tensor_hash root")?;
            return stream
                .clone_dtoh(&d_root)
                .context("copying tensor_hash root")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("tensor_hash root had the wrong length"));
        }

        stream.synchronize().context("synchronizing tensor_hash leaf")?;
        stream
            .clone_dtoh(&d_level)
            .context("copying tensor_hash single-chunk root")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("tensor_hash root had the wrong length"))
    }

    /// Hashes a signed matrix exactly as `pearl_ref::Problem::commitment` pads and encodes it.
    pub fn hash_matrix_i8(&self, matrix: &[i8], key: [u8; 32]) -> Result<[u8; 32]> {
        ensure!(!matrix.is_empty(), "tensor_hash matrix must be non-empty");
        let mut bytes: Vec<u8> = matrix.iter().map(|&value| value as u8).collect();
        bytes.resize(bytes.len().div_ceil(BLAKE3_CHUNK_BYTES) * BLAKE3_CHUNK_BYTES, 0);
        self.hash_padded(&bytes, key)
    }

    fn launch_leaf(
        &self,
        stream: &Arc<CudaStream>,
        data: &CudaSlice<u8>,
        key: &CudaSlice<u8>,
        chunk_count: u32,
        root_output: u32,
        output: &mut CudaSlice<u8>,
    ) -> Result<()> {
        let (data_ptr, _data_sync) = data.device_ptr(stream);
        let (key_ptr, _key_sync) = key.device_ptr(stream);
        let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
        let mut params: [*mut c_void; 5] = [
            (&data_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
            (&chunk_count as *const u32).cast_mut().cast(),
            (&key_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
            (&root_output as *const u32).cast_mut().cast(),
            (&output_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
        ];
        let blocks = chunk_count.div_ceil(HASH_BLOCK_THREADS);
        unsafe {
            result::launch_kernel(
                self.leaf.function,
                (blocks, 1, 1),
                (HASH_BLOCK_THREADS, 1, 1),
                0,
                stream.cu_stream(),
                &mut params,
            )
        }
        .context("launching tensor_hash leaf kernel")
    }

    fn launch_parent(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u8>,
        key: &CudaSlice<u8>,
        input_count: u32,
        root_output: bool,
        output: &mut CudaSlice<u8>,
    ) -> Result<()> {
        let (input_ptr, _input_sync) = input.device_ptr(stream);
        let (key_ptr, _key_sync) = key.device_ptr(stream);
        let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
        let root_flag = root_output as u32;
        let mut params: [*mut c_void; 5] = [
            (&input_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
            (&input_count as *const u32).cast_mut().cast(),
            (&key_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
            (&root_flag as *const u32).cast_mut().cast(),
            (&output_ptr as *const sys::CUdeviceptr).cast_mut().cast(),
        ];
        let output_count = input_count.div_ceil(2);
        let blocks = output_count.div_ceil(HASH_BLOCK_THREADS);
        unsafe {
            result::launch_kernel(
                self.parent.function,
                (blocks, 1, 1),
                (HASH_BLOCK_THREADS, 1, 1),
                0,
                stream.cu_stream(),
                &mut params,
            )
        }
        .context("launching tensor_hash parent kernel")
    }
}

/// Computes both committed matrix roots on the GPU and derives the remaining official seeds.
pub struct GpuJobSetup<'a> {
    pub tensor_hash: &'a TensorHashCudaKernel,
}

impl<'a> GpuJobSetup<'a> {
    pub fn prepare(&self, problem: &Problem) -> Result<Commitment> {
        let job_key = problem.job_key();
        let root_a = self.tensor_hash.hash_matrix_i8(&problem.a, job_key)?;
        let root_b = self.tensor_hash.hash_matrix_i8(&problem.bt, job_key)?;
        let bound_a = bind_root_a(&root_a, problem.m);
        let bound_b = bind_root_b(&root_b, problem.n);
        let b_noise_seed = pearl_blake3::blake3_digest(
            &[&job_key[..], &bound_b[..]].concat(),
            None,
        );
        let a_noise_seed = pearl_blake3::blake3_digest(
            &[&b_noise_seed[..], &bound_a[..]].concat(),
            None,
        );
        Ok(Commitment {
            job_key,
            root_a,
            root_b,
            bound_a,
            bound_b,
            b_noise_seed,
            a_noise_seed,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobSlotState {
    Idle,
    Building,
    Ready,
    Running,
}

/// One half of the double-buffered job ring. Each slot has its own CUDA stream.
pub struct GpuJobSlot {
    pub index: usize,
    pub generation: u64,
    pub state: JobSlotState,
    pub stream: Arc<CudaStream>,
    pub commitment: Option<Commitment>,
}

pub struct GpuDeviceWorker {
    pub ordinal: usize,
    pub context: Arc<CudaContext>,
    pub tensor_hash: TensorHashCudaKernel,
    pub slots: [GpuJobSlot; 2],
    current_slot: usize,
    cancelled: CancellationToken,
}

impl GpuDeviceWorker {
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal)
            .with_context(|| format!("opening GPU worker device {ordinal}"))?;
        let tensor_hash = TensorHashCudaKernel::from_context(context.clone())?;
        let stream0 = context.new_stream().context("creating GPU job stream 0")?;
        let stream1 = context.new_stream().context("creating GPU job stream 1")?;
        let slots = [
            GpuJobSlot {
                index: 0,
                generation: 0,
                state: JobSlotState::Idle,
                stream: stream0,
                commitment: None,
            },
            GpuJobSlot {
                index: 1,
                generation: 0,
                state: JobSlotState::Idle,
                stream: stream1,
                commitment: None,
            },
        ];
        Ok(Self {
            ordinal,
            context,
            tensor_hash,
            slots,
            current_slot: 0,
            cancelled: CancellationToken::new(),
        })
    }

    pub fn build_next_job(&mut self, problem: &Problem) -> Result<usize> {
        let slot = 1 - self.current_slot;
        ensure!(
            self.slots[slot].state == JobSlotState::Idle,
            "GPU {} next job slot is not idle",
            self.ordinal
        );
        self.slots[slot].state = JobSlotState::Building;
        let result = GpuJobSetup {
            tensor_hash: &self.tensor_hash,
        }
        .prepare(problem);
        match result {
            Ok(commitment) => {
                self.slots[slot].generation = self.slots[slot].generation.wrapping_add(1);
                self.slots[slot].commitment = Some(commitment);
                self.slots[slot].state = JobSlotState::Ready;
                Ok(slot)
            }
            Err(error) => {
                self.slots[slot].state = JobSlotState::Idle;
                Err(error)
            }
        }
    }

    pub fn start_ready_job(&mut self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots.len(), "invalid GPU job slot {slot}");
        ensure!(
            self.slots[slot].state == JobSlotState::Ready,
            "GPU {} job slot {slot} is not ready",
            self.ordinal
        );
        self.current_slot = slot;
        self.slots[slot].state = JobSlotState::Running;
        Ok(())
    }

    pub fn finish_current_job(&mut self) {
        self.slots[self.current_slot].state = JobSlotState::Idle;
        self.slots[self.current_slot].commitment = None;
    }

    pub fn cancellation(&self) -> CancellationToken {
        self.cancelled.clone()
    }

    pub fn restart(&mut self) -> Result<()> {
        self.context.synchronize().context("synchronizing GPU before restart")?;
        let replacement = Self::new(self.ordinal)?;
        *self = replacement;
        Ok(())
    }
}

/// One worker/context per selected device. Workers never share a CUDA context or stream.
pub struct MultiGpuWorkers {
    pub workers: Vec<GpuDeviceWorker>,
}

impl MultiGpuWorkers {
    pub fn new(device_ordinals: &[usize]) -> Result<Self> {
        ensure!(!device_ordinals.is_empty(), "at least one GPU is required");
        let mut workers = Vec::with_capacity(device_ordinals.len());
        for &ordinal in device_ordinals {
            workers.push(GpuDeviceWorker::new(ordinal)?);
        }
        Ok(Self { workers })
    }

    pub fn all_visible() -> Result<Self> {
        let count = CudaContext::device_count().context("querying CUDA device count")?;
        ensure!(count > 0, "no CUDA devices are visible");
        let ordinals: Vec<usize> = (0..count as usize).collect();
        Self::new(&ordinals)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Clone, Debug, Default)]
pub struct PowerClockOptions {
    pub power_limit_watts: Option<u32>,
    pub graphics_clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
}

impl PowerClockOptions {
    /// Applies operator-requested limits through `nvidia-smi`; no shell is used and an error is
    /// returned before mining starts. Resetting limits is intentionally left to the operator.
    pub fn apply(&self, device_ordinal: usize) -> Result<()> {
        if self.power_limit_watts.is_none()
            && self.graphics_clock_mhz.is_none()
            && self.memory_clock_mhz.is_none()
        {
            return Ok(());
        }
        let mut command = Command::new("nvidia-smi");
        command.args(["-i", &device_ordinal.to_string()]);
        if let Some(watts) = self.power_limit_watts {
            command.args(["-pl", &watts.to_string()]);
        }
        if let Some(clock) = self.graphics_clock_mhz {
            command.args(["-lgc", &format!("{clock},{clock}")]);
        }
        if let Some(clock) = self.memory_clock_mhz {
            command.args(["-lmc", &format!("{clock},{clock}")]);
        }
        let status = command.status().context("running nvidia-smi power/clock command")?;
        ensure!(status.success(), "nvidia-smi rejected GPU power/clock options");
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneCandidate {
    pub warpgroups: u32,
    pub pipeline_stages: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneResult {
    pub device_key: String,
    pub candidate: TuneCandidate,
    pub tiles_per_second: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AutoTuneCache {
    pub entries: Vec<TuneResult>,
}

impl AutoTuneCache {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = fs::read(path).with_context(|| format!("reading autotune cache {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parsing autotune cache {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).with_context(|| format!("creating autotune directory {}", parent.display()))?;
        let temp = path.with_extension("tmp");
        let encoded = serde_json::to_vec_pretty(self).context("encoding autotune cache")?;
        fs::write(&temp, encoded).with_context(|| format!("writing autotune cache {}", temp.display()))?;
        fs::rename(&temp, path).with_context(|| format!("publishing autotune cache {}", path.display()))
    }

    pub fn choose<F>(
        &mut self,
        device_key: &str,
        candidates: &[TuneCandidate],
        mut measure: F,
    ) -> Result<TuneResult>
    where
        F: FnMut(&TuneCandidate) -> Result<f64>,
    {
        if let Some(found) = self.entries.iter().find(|entry| entry.device_key == device_key) {
            return Ok(found.clone());
        }
        ensure!(!candidates.is_empty(), "autotune candidate list is empty");
        let mut best: Option<TuneResult> = None;
        for candidate in candidates {
            let tiles_per_second = measure(candidate)?;
            ensure!(tiles_per_second.is_finite() && tiles_per_second >= 0.0, "invalid autotune score");
            if best.as_ref().is_none_or(|current| tiles_per_second > current.tiles_per_second) {
                best = Some(TuneResult {
                    device_key: device_key.to_owned(),
                    candidate: candidate.clone(),
                    tiles_per_second,
                });
            }
        }
        let selected = best.ok_or_else(|| anyhow::anyhow!("autotune produced no candidate"))?;
        self.entries.retain(|entry| entry.device_key != device_key);
        self.entries.push(selected.clone());
        Ok(selected)
    }
}
