//! Reference inputs for tools/verify_sm75_noise.py; no CUDA or network needed.
use serde_json::json;
use zk_pow::circuit::pearl_noise::{
    compute_noise_for_indices, generate_permutation_matrix, generate_uniform_random_matrix,
};
fn main() {
    let mut cases = Vec::new();
    for (rank, k) in [(32, 64), (64, 1024), (128, 4096)] {
        let rows = 5;
        let indices: Vec<usize> = (0..rows).collect();
        let a_seed = [0x11u8; 32];
        let b_seed = [0xabu8; 32];
        let noise = compute_noise_for_indices(k, rank, (b_seed, a_seed), &indices, &indices);
        for (is_b, seed, expected_noise, label) in [
            (0, a_seed, &noise.a, b"A_tensor"),
            (1, b_seed, &noise.b, b"B_tensor"),
        ] {
            let mut padded = [0u8; 32];
            padded[..8].copy_from_slice(label);
            let permutation: Vec<u32> = generate_permutation_matrix(&padded, &seed, k, rank)
                .into_iter()
                .flatten()
                .collect();
            let uniform: Vec<i8> = generate_uniform_random_matrix(&padded, &seed, &indices, rank)
                .into_iter()
                .flatten()
                .collect();
            let raw: Vec<i8> = (0..rows * k)
                .map(|i| ((i * 17 % 129) as i32 - 64) as i8)
                .collect();
            let expected: Vec<i8> = raw
                .iter()
                .zip(expected_noise.iter().flatten())
                .map(|(&v, &n)| (v as i32 + n as i32) as i8)
                .collect();
            cases.push(json!({"is_b":is_b,"seed":seed,"permutation":permutation,"uniform":uniform,"raw":raw,"expected":expected,"rows":rows,"k":k,"rank":rank}));
        }
    }
    println!("{}", serde_json::to_string(&cases).unwrap());
}
