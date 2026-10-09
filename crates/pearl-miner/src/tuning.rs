use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{ensure, Context, Result};
use pearl_gpu::{inspect_gpu, select_production_kernel, GemmKernel, GpuInfo};
use pearl_ref::test_problem;
use serde::{Deserialize, Serialize};

const PROFILE_VERSION: u32 = 2;
const KERNEL_ABI: &str = "pearl-v3-production-1";

// pearl_gpu::m5::AutoTuneCache is for M5 tensor-hash warpgroups, not the V3 mining scan.

fn kernel_version() -> String {
    format!("{KERNEL_ABI}-{}", pearl_gpu::kernel_source_hash())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Parameters {
    pub groups_per_launch: u32,
    pub mem_mode: String,
    pub streams: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Profile {
    version: u32,
    gpu: String,
    ordinal: usize,
    capability: (i32, i32),
    uuid: String,
    driver_version: String,
    toolkit_version: String,
    miner_version: String,
    kernel_version: String,
    timestamp: u64,
    pub parameters: Parameters,
    pub hashrate: f64,
    pub variance: f64,
    pub stability_seconds: u64,
    pub total_memory_bytes: u64,
    pub memory_bus_width_bits: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
    pub power_watts: Option<f64>,
    pub temperature_c: Option<u32>,
    pub utilization_gpu_percent: Option<u32>,
    pub utilization_memory_percent: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub warmup_seconds: u64,
    pub measure_seconds: u64,
    pub stability_seconds: u64,
    pub max_temperature_c: u32,
}

fn cache_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".pearl-miner/tuning"))
}

fn path(info: &GpuInfo) -> Result<PathBuf> {
    let arch = format!("sm{}{}", info.compute_capability.0, info.compute_capability.1);
    let uuid = info.uuid.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect::<String>();
    Ok(cache_dir()?.join(arch).join(format!("{uuid}.json")))
}

pub fn show(ordinal: usize) -> Result<()> {
    let info = inspect_gpu(ordinal)?;
    let file = path(&info)?;
    match fs::read_to_string(&file) {
        Ok(data) => println!("{}", serde_json::to_string_pretty(&serde_json::from_str::<Profile>(&data)?)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => println!("No tuning profile for {} (SM{}{}).", info.name, info.compute_capability.0, info.compute_capability.1),
        Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
    }
    Ok(())
}

pub fn clear(ordinal: usize) -> Result<()> {
    let info = inspect_gpu(ordinal)?;
    let file = path(&info)?;
    match fs::remove_file(&file) {
        Ok(()) => println!("Removed {}", file.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => println!("No tuning profile for {}", info.name),
        Err(error) => return Err(error).with_context(|| format!("removing {}", file.display())),
    }
    Ok(())
}

fn load(info: &GpuInfo, ordinal: usize) -> Result<Option<Profile>> {
    let file = path(info)?;
    let data = match fs::read_to_string(&file) {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", file.display())),
    };
    let Ok(profile) = serde_json::from_str::<Profile>(&data) else { return Ok(None) };
    Ok(compatible(&profile, info, ordinal).then_some(profile))
}

fn compatible(profile: &Profile, info: &GpuInfo, _ordinal: usize) -> bool {
    profile.version == PROFILE_VERSION && profile.gpu == info.name
        && profile.capability == info.compute_capability && profile.uuid == info.uuid
        && profile.driver_version == info.driver_version && profile.toolkit_version == info.toolkit_version
        && profile.miner_version == env!("CARGO_PKG_VERSION") && profile.kernel_version == kernel_version()
        && profile.parameters.groups_per_launch != 0 && profile.parameters.groups_per_launch <= 32768
        && match profile.parameters.mem_mode.as_str() {
            "fused" => profile.parameters.streams == 1,
            "pipelined" => (2..=8).contains(&profile.parameters.streams),
            "buffered" => info.compute_capability == (12, 0) && profile.parameters.streams == 1,
            _ => false,
        }
        && profile.hashrate.is_finite() && profile.hashrate > 0.0
}

fn candidate_score(rate: f64, variance: f64) -> Option<f64> {
    (rate.is_finite() && rate > 0.0 && variance.is_finite() && variance >= 0.0)
        .then_some(rate / (1.0 + variance.sqrt()))
}

fn save(info: &GpuInfo, ordinal: usize, profile: &Profile) -> Result<()> {
    let file = path(info, ordinal)?;
    write_profile(&file, profile)
}

fn write_profile(file: &std::path::Path, profile: &Profile) -> Result<()> {
    let parent = file.parent().context("tuning cache path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let temporary = file.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut output = File::create(&temporary).with_context(|| format!("creating {}", temporary.display()))?;
    serde_json::to_writer_pretty(&mut output, profile)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(&temporary, &file).with_context(|| format!("installing {}", file.display()))?;
    File::open(parent)?.sync_all().with_context(|| format!("syncing {}", parent.display()))
}

fn candidates(info: &GpuInfo) -> Vec<Parameters> {
    let sm = info.multiprocessors.max(1);
    let groups = match info.compute_capability {
        (7, 5) => [4, 8, 16],
        (9, 0) => [8, 16, 32],
        (12, 0) => [8, 16, 32],
        _ => [4, 8, 16],
    };
    let modes: &[(&str, &[u32])] = if info.compute_capability == (12, 0) {
        &[("fused", &[1]), ("pipelined", &[2, 4, 8]), ("buffered", &[1])]
    } else {
        &[("fused", &[1]), ("pipelined", &[2, 4, 8])]
    };
    let mut result = Vec::new();
    for multiplier in groups {
        for (mode, stream_counts) in modes {
            for streams in *stream_counts {
                result.push(Parameters {
                    groups_per_launch: sm.saturating_mul(multiplier).clamp(32, 32768),
                    mem_mode: (*mode).into(),
                    streams: *streams,
                });
            }
        }
    }
    result
}

fn focused_candidates(seeds: &[Parameters], fine: bool) -> Vec<Parameters> {
    let mut result = Vec::new();
    for seed in seeds {
        let batch_factors: &[f64] = if fine { &[0.95, 1.05] } else { &[0.75, 1.25] };
        let stream_values: Vec<u32> = match seed.mem_mode.as_str() {
            "pipelined" => [seed.streams.saturating_sub(1).max(2), seed.streams, (seed.streams + 1).min(8)].to_vec(),
            _ => vec![1],
        };
        for factor in batch_factors {
            let batch = ((seed.groups_per_launch as f64 * *factor).round() as u32).clamp(32, 32768);
            for &streams in &stream_values {
                let candidate = Parameters { groups_per_launch: batch, mem_mode: seed.mem_mode.clone(), streams };
                if !result.contains(&candidate) { result.push(candidate); }
            }
        }
    }
    result
}

type Scored = (Parameters, f64, f64);

fn score_measurement(parameters: &Parameters, measurement: Measurement) -> Option<Scored> {
    if measurement.pruned { return None; }
    Some((parameters.clone(), candidate_score(measurement.rate, measurement.variance)?, measurement.rate))
}

fn run_stage(device: usize, label: &str, choices: &[Parameters], settings: Settings, cutoff: Option<f64>) -> Vec<Scored> {
    let mut results = Vec::new();
    for (index, candidate) in choices.iter().enumerate() {
        println!("[TUNE] {label} candidate {}/{} {:?}", index + 1, choices.len(), candidate);
        match measure_isolated(device, candidate, settings, false, cutoff) {
            Ok(result) if result.pruned => println!("[TUNE] pruned below 80% of current best after two measurement windows"),
            Ok(result) => {
                let Some(scored) = score_measurement(candidate, result) else {
                    println!("[TUNE] rejected: invalid throughput or variance");
                    continue;
                };
                let (_, score, _) = scored;
                println!("[TUNE] measured={:.3} TH/s work={} variance={:.4} temp={:?}->{:?}C power={:?}W clocks={:?}/{:?}MHz gpu_util={:?}% mem_util={:?}% score={:.3} TH/s",
                    result.rate / 1e12, result.work, result.variance, result.start_temperature_c,
                    result.end_temperature_c, result.power_watts, result.graphics_clock_mhz, result.memory_clock_mhz,
                    result.utilization_gpu_percent, result.utilization_memory_percent, score / 1e12);
                results.push(scored);
            }
            Err(error) => println!("[TUNE] rejected: {error:#}"),
        }
    }
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

fn set_parameters(parameters: &Parameters) {
    std::env::set_var("PEARL_GROUPS_PER_LAUNCH", parameters.groups_per_launch.to_string());
    std::env::set_var("PEARL_MEM_MODE", &parameters.mem_mode);
    std::env::set_var("PEARL_STREAMS", parameters.streams.to_string());
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Measurement {
    pub rate: f64,
    pub variance: f64,
    pub work: u64,
    pub pruned: bool,
    pub start_temperature_c: Option<u32>,
    pub end_temperature_c: Option<u32>,
    pub power_watts: Option<f64>,
    pub graphics_clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
    pub utilization_gpu_percent: Option<u32>,
    pub utilization_memory_percent: Option<u32>,
}

pub fn probe(device: usize, parameters: &Parameters, settings: Settings, stable: bool, cutoff: Option<f64>) -> Result<Measurement> {
    set_parameters(parameters);
    let start_telemetry = inspect_gpu(device)?;
    ensure!(start_telemetry.temperature_c.map_or(true, |temp| temp <= settings.max_temperature_c),
        "GPU temperature {:?}C exceeds tuning limit {}C", start_telemetry.temperature_c, settings.max_temperature_c);
    let kernel = select_production_kernel(device)?;
    kernel.mining_selftest()?;
    let header = test_problem()?.header;
    let problem = crate::job::configured_problem(header, 0, kernel.device_matrix_preparation())?;
    kernel.prepare_job(&problem)?;
    let (groups, batch) = kernel.mining_groups(&problem)?.context("production grouped scan unavailable")?;
    ensure!(groups > 0 && batch > 0, "CUDA backend produced an empty scan range");
    let scan = |duration: Duration, cutoff: Option<f64>| -> Result<(f64, u64, f64, bool)> {
        let start = std::time::Instant::now();
        let mut cursor = 0;
        let mut work = 0u64;
        let mut bins = Vec::new();
        let mut bin_start = start;
        let mut bin_work = 0u64;
        let mut pruned = false;
        while start.elapsed() < duration {
            let count = batch.min(groups - cursor);
            ensure!(kernel.scan_groups(&problem, cursor, count, [0; 32])?.is_empty(), "zero-target tuning scan returned a candidate");
            let done = u64::from(count) * 128 * 128 * problem.k() as u64;
            work += done;
            bin_work += done;
            cursor = (cursor + count) % groups;
            if bin_start.elapsed() >= Duration::from_secs(1) {
                bins.push(bin_work as f64 / bin_start.elapsed().as_secs_f64());
                bin_work = 0;
                bin_start = std::time::Instant::now();
                if bins.len() >= 2 && cutoff.is_some_and(|limit| {
                    work as f64 / start.elapsed().as_secs_f64() < limit
                }) {
                    pruned = true;
                    break;
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        if bin_work > 0 { bins.push(bin_work as f64 / bin_start.elapsed().as_secs_f64()); }
        let mean = work as f64 / elapsed;
        let variance = if bins.len() < 2 { 0.0 } else {
            let avg = bins.iter().sum::<f64>() / bins.len() as f64;
            bins.iter().map(|v| ((v - avg) / avg).powi(2)).sum::<f64>() / bins.len() as f64
        };
        Ok((mean, work, variance, pruned))
    };
    if settings.warmup_seconds > 0 {
        let _ = scan(Duration::from_secs(settings.warmup_seconds), None)?;
    }
    let seconds = if stable { settings.stability_seconds } else { settings.measure_seconds };
    let (rate, work, variance, pruned) = scan(Duration::from_secs(seconds), if stable { None } else { cutoff })?;
    let end_telemetry = inspect_gpu(device)?;
    ensure!(end_telemetry.temperature_c.map_or(true, |temp| temp <= settings.max_temperature_c),
        "GPU temperature {:?}C exceeds tuning limit {}C", end_telemetry.temperature_c, settings.max_temperature_c);
    Ok(Measurement {
        rate, variance, work, pruned,
        start_temperature_c: start_telemetry.temperature_c,
        end_temperature_c: end_telemetry.temperature_c,
        power_watts: end_telemetry.power_watts,
        graphics_clock_mhz: end_telemetry.graphics_clock_mhz,
        memory_clock_mhz: end_telemetry.memory_clock_mhz,
        utilization_gpu_percent: end_telemetry.utilization_gpu_percent,
        utilization_memory_percent: end_telemetry.utilization_memory_percent,
    })
}

fn measure_isolated(device: usize, parameters: &Parameters, settings: Settings, stable: bool, cutoff: Option<f64>) -> Result<Measurement> {
    let executable = std::env::current_exe().context("locating miner executable for isolated tuning")?;
    let duration = if stable { settings.stability_seconds } else { settings.measure_seconds };
    let timeout = Duration::from_secs(settings.warmup_seconds.saturating_add(duration).saturating_add(90));
    let mut command = Command::new(executable);
    command.arg("--tune-probe").arg("--device").arg(device.to_string())
        .arg("--mem-mode").arg(&parameters.mem_mode)
        .arg("--streams").arg(parameters.streams.to_string())
        .arg("--groups-per-launch").arg(parameters.groups_per_launch.to_string())
        .arg("--tune-warmup").arg(settings.warmup_seconds.to_string())
        .arg("--tune-duration").arg(settings.measure_seconds.to_string())
        .arg("--tune-stability").arg(settings.stability_seconds.to_string());
    command.arg("--tune-max-temp-c").arg(settings.max_temperature_c.to_string());
    if stable { command.arg("--tune-probe-stable"); }
    if let Some(cutoff) = cutoff.filter(|_| !stable) {
        command.arg("--tune-cutoff-ths").arg((cutoff / 1e12).to_string());
    }
    let mut child = command.stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().context("starting isolated CUDA tuning candidate")?;
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait().context("checking isolated tuning process")? {
            let output = child.wait_with_output().context("collecting isolated tuning output")?;
            if !status.success() {
                anyhow::bail!("isolated tuning process exited {status}: {}", String::from_utf8_lossy(&output.stderr));
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            let json = stdout.lines().find_map(|line| line.strip_prefix("PEARL_TUNE_PROBE "))
                .context("isolated tuning process did not return a result")?;
            return serde_json::from_str(json).context("parsing isolated tuning result");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("isolated tuning candidate timed out after {} seconds", timeout.as_secs());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub fn apply(device: usize, settings: Settings, force: bool) -> Result<Parameters> {
    let info = inspect_gpu(device)?;
    let arch = format!("SM{}{}", info.compute_capability.0, info.compute_capability.1);
    println!("[TUNE] GPU {device}: {} ({arch}, {} SMs, {:.1} GiB VRAM, bus={} bit, driver={}, CUDA={})", info.name,
        info.multiprocessors, info.total_memory_bytes as f64 / 1024f64.powi(3),
        info.memory_bus_width_bits.map_or_else(|| "unknown".into(), |n| n.to_string()), info.driver_version, info.toolkit_version);
    if !force {
        if let Some(profile) = load(&info, device)? {
            println!("[TUNE] Verifying cached {:?}", profile.parameters);
            match measure_isolated(device, &profile.parameters, Settings {
                warmup_seconds: 1, measure_seconds: 3, stability_seconds: 3,
                max_temperature_c: settings.max_temperature_c,
            }, false, None) {
                Ok(check) if check.rate >= profile.hashrate * 0.85 => {
                    println!("[TUNE] Cache valid: {:.3} TH/s", check.rate / 1e12);
                    return Ok(profile.parameters);
                }
                Ok(check) => println!("[TUNE] Cache degraded: {:.3} TH/s vs {:.3} TH/s; retuning", check.rate / 1e12, profile.hashrate / 1e12),
                Err(error) => println!("[TUNE] Cached profile failed verification: {error:#}; retuning"),
            }
        }
    }
    let choices = candidates(&info);
    let mut results = run_stage(device, "coarse", &choices, settings, None);
    ensure!(!results.is_empty(), "all coarse tuning candidates failed");
    let seeds: Vec<_> = results.iter().take(2).map(|result| result.0.clone()).collect();
    let cutoff = results[0].2 * 0.80;
    let focused = focused_candidates(&seeds, false);
    results.extend(run_stage(device, "focused", &focused, settings, Some(cutoff)));
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    let fine = focused_candidates(&[results[0].0.clone()], true);
    let fine_cutoff = results[0].2 * 0.80;
    results.extend(run_stage(device, "fine", &fine, settings, Some(fine_cutoff)));
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (parameters, _, coarse_rate) = results.into_iter().next().context("tuning search produced no candidate")?;
    println!("[TUNE] Stability test: {:?}", parameters);
    let stable = measure_isolated(device, &parameters, settings, true, None)?;
    ensure!(stable.rate.is_finite() && stable.rate >= coarse_rate * 0.80 && stable.variance < 0.25,
        "selected CUDA profile failed stability validation (rate={:.3} TH/s variance={:.3})", stable.rate / 1e12, stable.variance);
    let final_info = inspect_gpu(device)?;
    let profile = Profile {
        version: PROFILE_VERSION, gpu: info.name.clone(), ordinal: device,
        capability: info.compute_capability, uuid: info.uuid.clone(),
        driver_version: info.driver_version.clone(), toolkit_version: info.toolkit_version.clone(),
        miner_version: env!("CARGO_PKG_VERSION").into(),
        kernel_version: kernel_version(),
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        parameters: parameters.clone(), hashrate: stable.rate, variance: stable.variance,
        stability_seconds: settings.stability_seconds,
        total_memory_bytes: final_info.total_memory_bytes,
        memory_bus_width_bits: final_info.memory_bus_width_bits,
        memory_clock_mhz: stable.memory_clock_mhz.or(final_info.memory_clock_mhz),
        power_watts: stable.power_watts,
        temperature_c: stable.end_temperature_c.or(final_info.temperature_c),
        utilization_gpu_percent: stable.utilization_gpu_percent,
        utilization_memory_percent: stable.utilization_memory_percent,
    };
    save(&info, device, &profile)?;
    println!("[TUNE] Best stable rate {:.3} TH/s; profile saved", stable.rate / 1e12);
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, capability: (i32, i32), multiprocessors: u32) -> GpuInfo {
        GpuInfo {
            name: name.into(), compute_capability: capability, multiprocessors,
            uuid: format!("GPU-{name}"), driver_version: "550.00".into(),
            toolkit_version: "13.0".into(), total_memory_bytes: 1 << 30,
            free_memory_bytes: 1 << 29, memory_bus_width_bits: Some(256),
            graphics_clock_mhz: Some(1500), memory_clock_mhz: Some(5000),
            power_watts: Some(100.0), temperature_c: Some(60),
            utilization_gpu_percent: Some(95), utilization_memory_percent: Some(70),
        }
    }

    #[test]
    fn candidate_spaces_are_architecture_specific_and_valid() {
        let t4 = candidates(&gpu("T4", (7, 5), 40));
        let h100 = candidates(&gpu("H100", (9, 0), 132));
        let blackwell = candidates(&gpu("RTX PRO 6000", (12, 0), 188));
        assert!(t4.iter().all(|c| c.groups_per_launch <= 32768 && c.mem_mode != "buffered"));
        assert!(h100.iter().any(|c| c.groups_per_launch >= 4096 && c.mem_mode == "pipelined"));
        assert!(blackwell.iter().any(|c| c.mem_mode == "buffered"));
        assert!(focused_candidates(&h100[..2], false).iter().all(|c| c.groups_per_launch > 0));
        assert!(focused_candidates(&h100[..2], true).iter().all(|c| (1..=8).contains(&c.streams)));
        assert_ne!(t4, h100);
    }

    #[test]
    fn profile_round_trip_and_gpu_mismatch_invalidate_cache() {
        let info = gpu("H100", (9, 0), 132);
        let profile = Profile {
            version: PROFILE_VERSION, gpu: info.name.clone(), ordinal: 0,
            capability: info.compute_capability, uuid: info.uuid.clone(),
            driver_version: info.driver_version.clone(), toolkit_version: info.toolkit_version.clone(),
            miner_version: env!("CARGO_PKG_VERSION").into(),
            kernel_version: kernel_version(),
            timestamp: 1, parameters: Parameters { groups_per_launch: 1024, mem_mode: "fused".into(), streams: 1 },
            hashrate: 1.0, variance: 0.0, stability_seconds: 120,
                total_memory_bytes: info.total_memory_bytes, memory_bus_width_bits: info.memory_bus_width_bits,
                memory_clock_mhz: info.memory_clock_mhz,
            power_watts: info.power_watts, temperature_c: info.temperature_c,
            utilization_gpu_percent: info.utilization_gpu_percent, utilization_memory_percent: info.utilization_memory_percent,
        };
        let restored: Profile = serde_json::from_str(&serde_json::to_string(&profile).unwrap()).unwrap();
        assert!(compatible(&restored, &info, 0));
        assert!(compatible(&restored, &info, 1)); // GPU UUID remains the identity after ordinal changes.
        assert!(!compatible(&restored, &gpu("T4", (7, 5), 40), 0));
        let mut changed_driver = info.clone();
        changed_driver.driver_version = "new-driver".into();
        assert!(!compatible(&restored, &changed_driver, 0));
    }

    #[test]
    fn profile_write_is_readable_and_replaces_atomically() {
        let info = gpu("T4", (7, 5), 40);
        let profile = Profile {
            version: PROFILE_VERSION, gpu: info.name.clone(), ordinal: 0,
            capability: info.compute_capability, uuid: info.uuid.clone(),
            driver_version: info.driver_version.clone(), toolkit_version: info.toolkit_version.clone(),
            miner_version: env!("CARGO_PKG_VERSION").into(), kernel_version: kernel_version(),
            timestamp: 1, parameters: Parameters { groups_per_launch: 160, mem_mode: "fused".into(), streams: 1 },
            hashrate: 100.0, variance: 0.0, stability_seconds: 120,
            total_memory_bytes: info.total_memory_bytes, memory_bus_width_bits: info.memory_bus_width_bits,
            memory_clock_mhz: info.memory_clock_mhz,
            power_watts: info.power_watts, temperature_c: info.temperature_c,
            utilization_gpu_percent: info.utilization_gpu_percent, utilization_memory_percent: info.utilization_memory_percent,
        };
        let folder = std::env::temp_dir().join(format!("pearl-profile-test-{}", std::process::id()));
        let file = folder.join("profile.json");
        write_profile(&file, &profile).unwrap();
        let restored: Profile = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        assert!(compatible(&restored, &info, 0));
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn score_rejects_invalid_measurements_and_penalizes_variance() {
        assert!(candidate_score(f64::NAN, 0.0).is_none());
        assert!(candidate_score(10.0, -1.0).is_none());
        assert!(candidate_score(10.0, 0.0).unwrap() > candidate_score(10.0, 1.0).unwrap());
        // Mock measurement path: choose the best stable real-work rate without CUDA.
        let samples = [("small", 8.0, 0.0), ("medium", 12.0, 0.25), ("noisy", 13.0, 4.0)];
        let best = samples.iter().max_by(|a, b| {
            candidate_score(a.1, a.2).unwrap().total_cmp(&candidate_score(b.1, b.2).unwrap())
        }).unwrap();
        assert_eq!(best.0, "medium");
    }

    #[test]
    fn mock_autotune_selects_fastest_stable_candidate_and_rejects_pruned_work() {
        let candidates = [
            Parameters { groups_per_launch: 64, mem_mode: "fused".into(), streams: 1 },
            Parameters { groups_per_launch: 128, mem_mode: "pipelined".into(), streams: 2 },
        ];
        let samples = [
            Measurement { rate: 100.0, variance: 0.0, work: 1000, pruned: false,
                start_temperature_c: Some(60), end_temperature_c: Some(65), power_watts: Some(100.0), graphics_clock_mhz: Some(1500), memory_clock_mhz: Some(5000), utilization_gpu_percent: Some(90), utilization_memory_percent: Some(40) },
            Measurement { rate: 1000.0, variance: 0.0, work: 1000, pruned: true,
                start_temperature_c: Some(60), end_temperature_c: Some(65), power_watts: Some(100.0), graphics_clock_mhz: Some(1500), memory_clock_mhz: Some(5000), utilization_gpu_percent: Some(90), utilization_memory_percent: Some(40) },
        ];
        let winner = candidates.iter().zip(samples).filter_map(|(candidate, measurement)|
            score_measurement(candidate, measurement)).max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(winner.0, candidates[0]);
    }
}
