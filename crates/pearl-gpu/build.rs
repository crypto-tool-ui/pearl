use std::{
    collections::hash_map::DefaultHasher,
    env,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(pearl_cuda_built)");
    let kernel_hash = hash_cuda_sources(Path::new("../../cuda"));
    println!("cargo:rustc-env=PEARL_KERNEL_SOURCE_HASH={kernel_hash}");
    println!("cargo:rerun-if-changed=../../cuda/naive.cu");
    println!("cargo:rerun-if-changed=../../cuda/sm75_mma.cu");
    println!("cargo:rerun-if-changed=../../cuda/production.cu");
    println!("cargo:rerun-if-changed=../../cuda/sm120_mma.cu");
    println!("cargo:rerun-if-changed=../../cuda/sm90_wgmma.cu");
    println!("cargo:rerun-if-changed=../../cuda/tensor_hash.cu");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc was not found; CUDA cubins are skipped");
        println!("cargo:rustc-env=PEARL_CUDA_TOOLKIT_VERSION=unknown");
        return;
    };

    let toolkit = Command::new(&nvcc).arg("--version").output()
        .ok().filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|output| output.lines().find_map(|line| line.split("release ").nth(1).map(str::to_owned)))
        .and_then(|version| version.split(',').next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=PEARL_CUDA_TOOLKIT_VERSION={toolkit}");

    println!("cargo:rustc-cfg=pearl_cuda_built");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    for (arch, name) in [
        ("sm_75", "PEARL_PRODUCTION_SM75"),
        ("sm_90a", "PEARL_PRODUCTION_SM90"),
        ("sm_120a", "PEARL_PRODUCTION_SM120"),
    ] {
        let output = out_dir.join(format!("pearl_production_{arch}.cubin"));
        let status = Command::new(&nvcc)
            .args(["-std=c++17", "-O3", "-lineinfo", "-Xptxas=-v", "-cubin",
                &format!("-arch={arch}"), "-o", output.to_str().expect("UTF-8 path"),
                "../../cuda/production.cu"])
            .status().expect("execute nvcc for production kernel");
        assert!(status.success(), "production kernel compilation failed for {arch}");
        println!("cargo:rustc-env={name}={}", output.display());
    }
    for (arch, env_name) in [
        ("sm_75", "PEARL_NAIVE_CUBIN_SM75"),
        ("sm_90a", "PEARL_NAIVE_CUBIN_SM90"),
        ("sm_120a", "PEARL_NAIVE_CUBIN_SM120"),
    ] {
        let output = out_dir.join(format!("pearl_naive_{arch}.cubin"));
        let status = Command::new(&nvcc)
            .args([
                "-std=c++17",
                "-O3",
                "-cubin",
                &format!("-arch={arch}"),
                "-o",
                output.to_str().expect("UTF-8 build path"),
                "../../cuda/naive.cu",
            ])
            .status()
            .unwrap_or_else(|error| panic!("failed to execute nvcc for {arch}: {error}"));
        if !status.success() {
            panic!("nvcc failed while compiling the naive {arch} cubin");
        }
        println!("cargo:rustc-env={env_name}={}", output.display());
    }

    let output = out_dir.join("pearl_sm75_mma_sm_75.cubin");
    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "-cubin",
            "-arch=sm_75",
            "-o",
            output.to_str().expect("UTF-8 build path"),
            "../../cuda/sm75_mma.cu",
        ])
        .status()
        .unwrap_or_else(|error| panic!("failed to execute nvcc for sm_75 MMA: {error}"));
    if !status.success() {
        panic!("nvcc failed while compiling the sm_75 MMA cubin");
    }
    println!("cargo:rustc-env=PEARL_SM75_MMA_CUBIN={}", output.display());

    let output = out_dir.join("pearl_sm120_mma_sm_120a.cubin");
    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "-cubin",
            "-arch=sm_120a",
            "-o",
            output.to_str().expect("UTF-8 build path"),
            "../../cuda/sm120_mma.cu",
        ])
        .status()
        .unwrap_or_else(|error| panic!("failed to execute nvcc for sm_120a MMA: {error}"));
    if !status.success() {
        panic!("nvcc failed while compiling the sm_120a MMA cubin");
    }
    println!("cargo:rustc-env=PEARL_SM120_MMA_CUBIN={}", output.display());

    let output = out_dir.join("pearl_sm90_wgmma_sm_90a.cubin");
    let status = Command::new(&nvcc)
        .args([
            "-std=c++17",
            "-O3",
            "-cubin",
            "-arch=sm_90a",
            "-o",
            output.to_str().expect("UTF-8 build path"),
            "../../cuda/sm90_wgmma.cu",
        ])
        .status()
        .unwrap_or_else(|error| panic!("failed to execute nvcc for sm_90a WGMMA: {error}"));
    if !status.success() {
        panic!("nvcc failed while compiling the sm_90a WGMMA cubin");
    }
    println!("cargo:rustc-env=PEARL_SM90_WGMMA_CUBIN={}", output.display());

    for (arch, env_name) in [
        ("sm_75", "PEARL_TENSOR_HASH_CUBIN_SM75"),
        ("sm_90a", "PEARL_TENSOR_HASH_CUBIN_SM90"),
        ("sm_120a", "PEARL_TENSOR_HASH_CUBIN_SM120"),
    ] {
        let output = out_dir.join(format!("pearl_tensor_hash_{arch}.cubin"));
        let status = Command::new(&nvcc)
            .args([
                "-std=c++17",
                "-O3",
                "-cubin",
                &format!("-arch={arch}"),
                "-o",
                output.to_str().expect("UTF-8 build path"),
                "../../cuda/tensor_hash.cu",
            ])
            .status()
            .unwrap_or_else(|error| panic!("failed to execute nvcc for tensor hash {arch}: {error}"));
        if !status.success() {
            panic!("nvcc failed while compiling the tensor hash {arch} cubin");
        }
        println!("cargo:rustc-env={env_name}={}", output.display());
    }
}

fn hash_cuda_sources(root: &Path) -> String {
    fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, files);
            } else if matches!(path.extension().and_then(|ext| ext.to_str()), Some("cu" | "cuh" | "h" | "hpp")) {
                println!("cargo:rerun-if-changed={}", path.display());
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    let mut hasher = DefaultHasher::new();
    for path in files {
        path.hash(&mut hasher);
        std::fs::read(path).unwrap_or_default().hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(explicit_nvcc) = env::var_os("NVCC") {
        let candidate = PathBuf::from(explicit_nvcc);
        if nvcc_works(&candidate) {
            return Some(candidate);
        }
    }

    for variable in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        let Some(cuda_root) = env::var_os(variable) else {
            continue;
        };
        let candidate = PathBuf::from(cuda_root).join("bin/nvcc");
        if nvcc_works(&candidate) {
            return Some(candidate);
        }
    }

    let candidate = PathBuf::from("nvcc");
    nvcc_works(&candidate).then_some(candidate)
}

fn nvcc_works(candidate: &PathBuf) -> bool {
    Command::new(candidate)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
