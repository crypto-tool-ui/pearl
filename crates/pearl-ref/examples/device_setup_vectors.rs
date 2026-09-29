//! CPU vectors for executing the actual device preparation source without CUDA.
use pearl_ref::Matrix;
use serde_json::json;
fn main() {
    let mut cases = Vec::new();
    for (seed, domain) in [(0u64, 0u64), (u64::MAX, 0x73706d2d612d3031), (0x123456789abcdef0, 0x73706d2d622d3031)] {
        for chunks in [1, 2, 3, 17, 512] {
            let matrix = Matrix::generated(seed, domain, chunks * 1024);
            let bytes: Vec<u8> = matrix.iter().map(|&v| v as u8).collect();
            let key = [0x93u8; 32];
            let root = pearl_blake3::blake3_digest(&bytes, Some(key));
            cases.push(json!({"seed":seed, "domain":domain, "bytes":bytes, "key":key, "root":root}));
        }
    }
    println!("{}", serde_json::to_string(&cases).unwrap());
}
