use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(pearl_cuda_built)");
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
        return;
    };

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
