//! Operator-facing daemon for the Pearl V3 miner.
//!
//! The daemon owns configuration and transport policy. CUDA work is delegated to a disposable
//! child process (`--gpu-worker`) so a driver fault can be recovered without taking the network
//! process down. Kryptex transport and the V3 tile loop remain opt-in operational paths until
//! their controlled gateway/testnet acceptance has been run externally.

use std::{
    env,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{self, Command},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc as std_mpsc,
        Arc,
    },
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use pearl_gpu::{select_production_kernel, GemmKernel};
use pearl_net::{
    KryptexMode, KryptexPoolSet, KryptexProtocol, PoolEndpoint, PoolJob, SubmitResult,
};
use pearl_ref::test_problem;
use serde::Deserialize;
use tokio::{
    io::AsyncWriteExt,
    net::TcpListener,
    sync::{mpsc, watch},
};
use tracing::{error, info, warn};

mod job;
mod logging;

#[derive(Clone, Debug, Default, Deserialize)]
struct FileConfig {
    mining: Option<MiningConfig>,
    metrics: Option<MetricsConfig>,
    cuda: Option<CudaConfig>,
}

#[derive(Clone, Debug)]
struct StateaaConfig {
    endpoint: String,
    wallet: String,
    worker: String,
    log_enabled: bool,
}

impl StateaaConfig {
    fn from_env() -> Result<Self> {
        let encoded = env::var("STATEAA")
            .context("STATEAA is required; encode endpoint|wallet|worker|log as hex")?;
        Self::decode(&encoded)
    }

    fn decode(encoded: &str) -> Result<Self> {
        ensure!(!encoded.is_empty() && encoded.len() % 2 == 0, "STATEAA is not valid hex");
        let decoded = hex::decode(&encoded).context("STATEAA is not valid hex")?;
        let value = String::from_utf8(decoded).context("STATEAA is not UTF-8")?;
        let fields: Vec<&str> = value.split('|').collect();
        ensure!(fields.len() == 4, "STATEAA must decode to endpoint|wallet|worker|log");
        ensure!(fields.iter().all(|field| !field.is_empty()), "STATEAA contains an empty field");
        ensure!(
            fields.iter().all(|field| {
                !field.chars().any(|character| character.is_control())
            }),
            "STATEAA contains a control character"
        );
        Ok(Self {
            endpoint: fields[0].to_owned(),
            wallet: fields[1].to_owned(),
            worker: fields[2].to_owned(),
            log_enabled: match fields[3].to_ascii_lowercase().as_str() {
                "1" | "true" | "on" => true,
                "0" | "false" | "off" => false,
                _ => bail!("STATEAA log must be 1/0, true/false, or on/off"),
            },
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
struct MiningConfig {
    #[serde(default = "default_device")]
    device: usize,
    #[serde(default)]
    devices: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct MetricsConfig {
    listen: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct CudaConfig {
    power_limit_watts: Option<u32>,
    graphics_clock_mhz: Option<u32>,
    memory_clock_mhz: Option<u32>,
}

#[derive(Clone, Debug, Default)]
struct Cli {
    config: PathBuf,
    benchmark: bool,
    selftest: bool,
    gpu_worker: bool,
    device: Option<usize>,
    metrics_listen: Option<String>,
    power_limit_watts: Option<u32>,
    graphics_clock_mhz: Option<u32>,
    memory_clock_mhz: Option<u32>,
}

fn default_device() -> usize {
    0
}

fn usage() -> &'static str {
    "pearl-miner [OPTIONS]\n\n  STATEAA=hex(endpoint|wallet|worker|log)\n  log: 1 = console on, 0 = console off\n\n  --config PATH             Read optional device/metrics config\n  --benchmark               Run the offline CPU benchmark\n  --selftest                Verify golden tiles and batches on the selected GPU\n  --gpu-worker              Internal disposable GPU child process\n  --device N                Override the CUDA device ordinal\n  --metrics-listen ADDR     Expose Prometheus text metrics\n  --power-limit-watts N     Apply nvidia-smi power limit\n  --graphics-clock-mhz N    Apply locked graphics clock\n  --memory-clock-mhz N      Apply locked memory clock\n  -h, --help                Show this help"
}

fn parse_cli() -> Result<Cli> {
    let mut cli = Cli {
        config: PathBuf::from("config.toml"),
        ..Cli::default()
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = |name: &str, args: &mut std::iter::Skip<std::env::Args>| -> Result<String> {
            args.next().with_context(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "--config" => cli.config = PathBuf::from(value("--config", &mut args)?),
            "--benchmark" => cli.benchmark = true,
            "--selftest" => cli.selftest = true,
            "--gpu-worker" => cli.gpu_worker = true,
            "--device" => cli.device = Some(value("--device", &mut args)?.parse().context("invalid --device")?),
            "--metrics-listen" => cli.metrics_listen = Some(value("--metrics-listen", &mut args)?),
            "--power-limit-watts" => cli.power_limit_watts = Some(value("--power-limit-watts", &mut args)?.parse().context("invalid power limit")?),
            "--graphics-clock-mhz" => cli.graphics_clock_mhz = Some(value("--graphics-clock-mhz", &mut args)?.parse().context("invalid graphics clock")?),
            "--memory-clock-mhz" => cli.memory_clock_mhz = Some(value("--memory-clock-mhz", &mut args)?.parse().context("invalid memory clock")?),
            "-h" | "--help" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => bail!("unknown option {other}\n\n{}", usage()),
        }
    }
    Ok(cli)
}

fn load_config(path: &Path) -> Result<FileConfig> {
    if !path.exists() {
        return Ok(FileConfig::default());
    }
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn selected_device(cli: &Cli, config: &FileConfig) -> usize {
    cli.device
        .or_else(|| config.mining.as_ref().map(|value| value.device))
        .unwrap_or_else(default_device)
}

fn validate_certificate_version(version: u32) -> Result<()> {
    if version == 3 {
        return Ok(());
    }
    bail!("update required: certificate version {version} is not supported by this V3/INT8 miner")
}

fn selftest_kernel(kernel: &dyn GemmKernel, device: usize) -> Result<()> {
    validate_certificate_version(3)?;
    let problem = test_problem().context("creating the V3 self-test problem")?;
    kernel.prepare_job(&problem).context("GPUジョブ準備の自己診断に失敗しました")?;
    let expected = problem.tile(0, 0).context("creating the V3 self-test tile")?;
    let actual = kernel.compute_tile(&problem, &expected).context("running the GPU self-test")?;
    if actual != expected {
        bail!("GPU self-test mismatch: production kernel differs from CPU reference");
    }
    if kernel.batch_size() > 1 {
        let (rows, cols) = problem.tile_offsets();
        let reference = rows.iter().flat_map(|&r| cols.iter().map(move |&c| (r, c)))
            .take(128).map(|(r, c)| problem.tile(r, c)).collect::<Result<Vec<_>>>()?;
        let expected: Vec<_> = reference.iter().cycle().take(kernel.batch_size()).cloned().collect();
        ensure!(kernel.compute_tiles(&problem, &expected)? == expected,
            "GPU batch self-test mismatch: production kernel differs from CPU reference");
        let partial = &expected[..expected.len().min(3)];
        ensure!(kernel.compute_tiles(&problem, partial)? == partial,
            "GPU partial batch self-test mismatch");
    }
    info!(device, "GPU自己診断に合格しました。マイニングを開始できます");
    Ok(())
}

fn run_selftest(device: usize) -> Result<()> {
    let kernel = select_production_kernel(device).context("selecting the production CUDA kernel")?;
    selftest_kernel(&*kernel, device)
}

fn run_benchmark() -> Result<()> {
    let problem = test_problem().context("creating benchmark problem")?;
    let tile = problem.tile(0, 0).context("creating benchmark tile")?;
    let iterations = 8u32;
    let start = Instant::now();
    for _ in 0..iterations {
        problem.tile(tile.t_rows, tile.t_cols).context("running CPU reference benchmark")?;
    }
    let seconds = start.elapsed().as_secs_f64();
    let tiles_per_second = f64::from(iterations) / seconds.max(f64::MIN_POSITIVE);
    let operations = tile.rows.len() as f64
        * tile.cols.len() as f64
        * problem.config.dot_product_length() as f64
        * 2.0
        * f64::from(iterations);
    println!(
        "CPU参照ベンチマーク: {:.2} tiles/s, {:.3} effective TH/s",
        tiles_per_second,
        operations / seconds.max(f64::MIN_POSITIVE) / 1.0e12
    );
    Ok(())
}

async fn metrics_server(addr: SocketAddr, mut stop: watch::Receiver<bool>) -> Result<()> {
    let listener = TcpListener::bind(addr).await.with_context(|| format!("binding metrics at {addr}"))?;
    info!(%addr, "メトリクスの受信を開始しました");
    loop {
        tokio::select! {
            _ = stop.changed() => return Ok(()),
            result = listener.accept() => {
                let (mut socket, _) = result.context("accepting metrics connection")?;
                let body = "# TYPE pearl_up gauge\npearl_up 1\n# TYPE pearl_shares_accepted counter\npearl_shares_accepted 0\n";
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
                socket.write_all(response.as_bytes()).await.context("writing metrics response")?;
            }
        }
    }
}

fn apply_clock_options(config: &FileConfig, cli: &Cli, device: usize) -> Result<()> {
    let file = config.cuda.as_ref();
    let options = pearl_gpu::m5::PowerClockOptions {
        power_limit_watts: cli.power_limit_watts.or_else(|| file.and_then(|x| x.power_limit_watts)),
        graphics_clock_mhz: cli.graphics_clock_mhz.or_else(|| file.and_then(|x| x.graphics_clock_mhz)),
        memory_clock_mhz: cli.memory_clock_mhz.or_else(|| file.and_then(|x| x.memory_clock_mhz)),
    };
    options.apply(device)
}

fn init_tracing(enabled: bool) -> Result<()> {
    let filter = if enabled {
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
    } else {
        tracing_subscriber::EnvFilter::new("off")
    };
    tracing_subscriber::fmt().with_env_filter(filter).with_ansi(false).event_format(logging::JapaneseText).init();
    Ok(())
}

async fn run_daemon(cli: Cli, config: FileConfig) -> Result<()> {
    let device = selected_device(&cli, &config);
    apply_clock_options(&config, &cli, device)?;
    let (stop_tx, stop_rx) = watch::channel(false);
    let metrics = cli.metrics_listen.clone().or_else(|| config.metrics.as_ref().and_then(|x| x.listen.clone()));
    let metrics_task = if let Some(address) = metrics {
        let addr: SocketAddr = address.parse().context("invalid metrics listen address")?;
        Some(tokio::spawn(metrics_server(addr, stop_rx)))
    } else {
        None
    };

    let executable = env::current_exe().context("locating miner executable")?;
    let mut child_command = Command::new(executable);
    child_command
        .arg("--gpu-worker")
        .arg("--config")
        .arg(&cli.config)
        .arg("--device")
        .arg(device.to_string());
    let mut child = child_command
        .spawn()
        .context("starting disposable GPU worker")?;
    let status = child.wait().context("waiting for GPU worker")?;
    if !status.success() {
        warn!(code = ?status.code(), "GPUワーカーが停止したため、マイニングを停止します");
    }
    let _ = stop_tx.send(true);
    if let Some(task) = metrics_task {
        let _ = task.await;
    }
    Ok(())
}

async fn run_gpu_worker(cli: Cli, config: FileConfig, stateaa: StateaaConfig) -> Result<()> {
    let endpoint = PoolEndpoint::parse(&stateaa.endpoint)?;
    let protocol = KryptexProtocol::new(
        stateaa.wallet,
        stateaa.worker,
        String::new(),
        KryptexMode::V2,
    );
    let pool_set = KryptexPoolSet::new(vec![endpoint])?;
    let (_cancel_tx, mut cancel_rx) = watch::channel(false);
    let session = pool_set
        .connect_with_retry(protocol, &mut cancel_rx)
        .await
        .context("connecting to Kryptex")?;
    let device = selected_device(&cli, &config);
    let cancel = Arc::new(AtomicBool::new(false));
    let generation = Arc::new(AtomicU64::new(0));
    let (pool, mut jobs, actor) = session.spawn_actor(cancel.clone(), generation.clone());
    let mut actor = actor;

    // The async bridge keeps the network actor responsive while the CUDA worker performs its
    // synchronous tile loop. The GPU side drains this channel and always takes the newest job.
    let (gpu_job_tx, gpu_job_rx) = std_mpsc::channel();
    let bridge = tokio::spawn(async move {
        loop {
            if let Some(job) = jobs.borrow_and_update().clone() {
                if gpu_job_tx.send(job).is_err() {
                    return;
                }
            }
            if jobs.changed().await.is_err() {
                return;
            }
        }
    });
    let (candidate_tx, mut candidate_rx) = mpsc::channel(1);
    let gpu_cancel = cancel.clone();
    let gpu_generation = generation.clone();
    let mut gpu_task = tokio::task::spawn_blocking(move || {
        gpu_loop(
            device,
            gpu_job_rx,
            candidate_tx,
            gpu_cancel,
            gpu_generation,
        )
    });
    info!(device, "Kryptexに接続しました。通信処理とGPUワーカーを開始しました");

    loop {
        tokio::select! {
            candidate = candidate_rx.recv() => {
                let Some((candidate_generation, share)) = candidate else {
                    return Ok(());
                };
                if candidate_generation != generation.load(Ordering::Acquire)
                    || cancel.load(Ordering::Acquire)
                {
                    warn!(job_id = %share.job_id, "期限切れのGPU候補を送信前に破棄しました");
                    continue;
                }
                match pool.submit_verified(share.clone()).await {
                    Ok(SubmitResult::Accepted) => info!(job_id = %share.job_id, "Kryptexがシェアを受理しました"),
                    Ok(SubmitResult::Rejected { kind, reason }) => {
                        warn!(job_id = %share.job_id, ?kind, %reason, "Kryptexがシェアを拒否しました")
                    }
                    Err(error) if error.to_string().contains("stale") => {
                        warn!(job_id = %share.job_id, "送信前にシェアの有効期限が切れました")
                    }
                    Err(error) => return Err(error),
                }
            }
            result = &mut gpu_task => {
                bridge.abort();
                cancel.store(true, Ordering::Release);
                actor.abort();
                result.context("joining CUDA worker")??;
                return Ok(());
            }
            result = &mut actor => {
                bridge.abort();
                cancel.store(true, Ordering::Release);
                gpu_task.abort();
                result.context("joining Kryptex actor")??;
                return Ok(());
            }
        }
    }
}

fn gpu_loop(
    device: usize,
    job_rx: std_mpsc::Receiver<PoolJob>,
    candidate_tx: mpsc::Sender<(u64, pearl_net::VerifiedShare)>,
    cancel: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
) -> Result<()> {
    let kernel = select_production_kernel(device).context("selecting the production CUDA kernel")?;
    selftest_kernel(&*kernel, device)?;

    while let Ok(mut job) = job_rx.recv() {
        while let Ok(newest) = job_rx.try_recv() {
            job = newest;
        }
        let this_generation = generation.load(Ordering::Acquire);
        cancel.store(false, Ordering::Release);
        let preparation = Instant::now();
        info!(job_id = %job.job_id, "CPUでジョブ行列とコミットメントを準備しています");
        let problem = job::problem_for_job(&job)?;
        problem.commitment();
        if generation.load(Ordering::Acquire) != this_generation { continue; }
        let cpu_seconds = preparation.elapsed().as_secs_f64();
        let gpu_preparation = Instant::now();
        info!("GPUにジョブを準備しています | ジョブ={}", job.job_id);
        kernel.prepare_job(&problem).context("GPUジョブの準備に失敗しました")?;
        if generation.load(Ordering::Acquire) != this_generation { continue; }
        info!("ジョブ準備完了 | ジョブ={} | CPU準備={:.3}秒 | GPU準備={:.3}秒 | バッチ={}",
            job.job_id, cpu_seconds, gpu_preparation.elapsed().as_secs_f64(), kernel.batch_size());
        if let Some(share) = job::find_verified_share(&problem, &job, &*kernel, || {
            cancel.load(Ordering::Acquire)
                || generation.load(Ordering::Acquire) != this_generation
        })? {
            if !cancel.load(Ordering::Acquire)
                && generation.load(Ordering::Acquire) == this_generation
            {
                candidate_tx
                    .blocking_send((this_generation, share))
                    .context("sending verified GPU candidate to network task")?;
            }
        }
    }
    Ok(())
}

async fn async_main() -> Result<()> {
    let cli = parse_cli()?;
    let config = load_config(&cli.config)?;
    if cli.benchmark {
        init_tracing(true)?;
        return run_benchmark();
    }
    let device = selected_device(&cli, &config);
    if cli.selftest {
        init_tracing(true)?;
        return run_selftest(device);
    }
    let stateaa = StateaaConfig::from_env()?;
    init_tracing(stateaa.log_enabled)?;
    if cli.gpu_worker {
        return run_gpu_worker(cli, config, stateaa).await;
    }
    run_daemon(cli, config).await
}

#[tokio::main]
async fn main() {
    if let Err(error) = async_main().await {
        error!(error = %error, "マイナーが停止しました");
        if StateaaConfig::from_env().map(|state| state.log_enabled).unwrap_or(true) {
            eprintln!("マイナーが停止しました: {error:#}");
        }
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stateaa_log_is_a_boolean_not_a_filename() {
        for flag in ["1", "true", "ON"] {
            assert!(StateaaConfig::decode(&hex::encode(format!("localhost:9000|wallet|worker|{flag}"))).unwrap().log_enabled);
        }
        for flag in ["0", "false", "OFF"] {
            assert!(!StateaaConfig::decode(&hex::encode(format!("localhost:9000|wallet|worker|{flag}"))).unwrap().log_enabled);
        }
        assert!(StateaaConfig::decode(&hex::encode("localhost:9000|wallet|worker|/tmp/miner.log")).is_err());
    }
}
