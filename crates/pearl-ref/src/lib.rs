//! CPU reference for Pearl certificate V3.
//!
//! Consensus structures, BLAKE3, noise generation, proof construction, and proof verification are
//! taken from the official `zk-pow` and `pearl-blake3` crates. This crate contains only the small
//! orchestration layer needed to expose complete tile transcripts to the CUDA correctness oracle.

use std::sync::{Arc, OnceLock};

use anyhow::{ensure, Context, Result};
use primitive_types::U256;
use serde::{Deserialize, Serialize};
use zk_pow::api::proof::Hash256;
use zk_pow::circuit::pearl_noise::compute_noise_for_indices;
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, MatrixMerkleProof, PlainProof};

pub use zk_pow::api::proof::SeedDerivation;
pub use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, MoEConfig, PeriodicPattern,
};
pub use zk_pow::api::proof_utils::compute_jackpot_hash;
pub use zk_pow::api::proof_utils::nbits_to_difficulty;
pub use zk_pow::api::sanity_checks::{extract_difficulty_bound, penalized_target_bound};
pub use zk_pow::api::seed::{bind_root_a, bind_root_b};
pub use zk_pow::api::verify::verify_plain_proof;

/// Kryptex's captured V3 dense work-unit shape. The pool job carries only the header and share
/// target; these values are the fixed Pearl V3 work convention documented by the Kryptex dialect.
pub const KRYPTEX_M: u32 = 131_072;
pub const KRYPTEX_N: u32 = 131_072;
pub const KRYPTEX_K: u32 = 4096;
pub const KRYPTEX_RANK: u16 = 128;
pub const KRYPTEX_FILL_DOMAIN: &[u8] = b"spm/fill/v1";
pub const KRYPTEX_ROWS: [u32; 8] = [0, 8, 16, 24, 32, 40, 48, 56];
pub const KRYPTEX_COLS: [u32; 16] = [
    0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57,
];

/// Immutable matrix, optionally regenerated only when a CPU proof needs it.
#[derive(Clone, Debug)]
pub struct Matrix {
    values: Arc<OnceLock<Vec<i8>>>,
    recipe: Option<(u64, u64, usize)>,
}
impl Matrix {
    pub fn generated(seed: u64, domain: u64, len: usize) -> Self {
        Self { values: Arc::new(OnceLock::new()), recipe: Some((seed, domain, len)) }
    }
    pub fn recipe(&self) -> Option<(u64, u64, usize)> { self.recipe }
    pub fn is_materialized(&self) -> bool { self.values.get().is_some() }
}
impl From<Vec<i8>> for Matrix {
    fn from(values: Vec<i8>) -> Self {
        Self { values: Arc::new(OnceLock::from(values)), recipe: None }
    }
}
impl std::ops::Deref for Matrix {
    type Target = [i8];
    fn deref(&self) -> &[i8] {
        self.values.get_or_init(|| {
            let (seed, domain, len) = self.recipe.expect("matrix recipe");
            fill_int7(seed, domain, len)
        })
    }
}

/// The committed inputs to one dense V3 proof-of-useful-work job.
#[derive(Clone, Debug)]
pub struct Problem {
    pub header: IncompleteBlockHeader,
    pub config: MiningConfiguration,
    pub m: u32,
    pub n: u32,
    /// A, row-major, with `m * k` entries.
    pub a: Matrix,
    /// B transpose, row-major, with `n * k` entries.
    pub bt: Matrix,
    commitment_cache: Arc<OnceLock<Commitment>>,
}

/// Commitment values reused by every tile in a problem.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub job_key: Hash256,
    pub root_a: Hash256,
    pub root_b: Hash256,
    pub bound_a: Hash256,
    pub bound_b: Hash256,
    pub b_noise_seed: Hash256,
    pub a_noise_seed: Hash256,
}

/// The complete CPU result for one committed jackpot tile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileResult {
    pub t_rows: u32,
    pub t_cols: u32,
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub transcript: [u32; 16],
    pub digest: Hash256,
}

impl Problem {
    /// Builds the deterministic dense V3 work unit used by the captured Kryptex dialect.
    ///
    /// This allocates the committed A and Bᵀ matrices (about 1 GiB at production dimensions) so
    /// the current correctness-first host API can feed the CUDA kernels and build proofs. A later
    /// streaming worker can replace the storage without changing the job or verifier contract.
    pub fn from_kryptex_header(header: IncompleteBlockHeader) -> Result<Self> {
        Self::from_kryptex_header_attempt(header, 0)
    }

    /// Independent committed matrices for subsequent scans of the same pool header.
    /// The attempt is local seed material, never a modification of the pool header.
    pub fn from_kryptex_header_attempt(header: IncompleteBlockHeader, attempt: u64) -> Result<Self> {
        let problem = Self::from_kryptex_header_attempt_deferred(header, attempt)?;
        let _ = &*problem.a;
        let _ = &*problem.bt;
        Ok(problem)
    }

    /// Deterministic matrix recipes for device generation; CPU bytes are deferred until proof.
    pub fn from_kryptex_header_attempt_deferred(header: IncompleteBlockHeader, attempt: u64) -> Result<Self> {
        Self::from_kryptex_header_attempt_shape(header, attempt, KRYPTEX_M, KRYPTEX_N, KRYPTEX_K)
    }

    /// Configurable dense V3 recipe. Rank and proof patterns remain fixed.
    /// Returns deferred matrices so large GPU jobs need not allocate host bytes until proof.
    pub fn from_kryptex_header_attempt_shape(
        header: IncompleteBlockHeader, attempt: u64, m: u32, n: u32, k: u32,
    ) -> Result<Self> {
        ensure!(m > 0 && n > 0 && m <= 1 << 24 && n <= 1 << 24,
            "M/N must be in 1..=2^24");
        ensure!(m % 128 == 0 && n % 128 == 0, "production M/N must be multiples of 128");
        ensure!((2048..=65536).contains(&k) && k % u32::from(KRYPTEX_RANK) == 0,
            "production K must be 2048..=65536 and divisible by rank 128");
        let a_len = (m as usize).checked_mul(k as usize).context("A size overflow")?;
        let b_len = (n as usize).checked_mul(k as usize).context("B size overflow")?;
        ensure!(a_len as u64 / 8 <= u64::from(u32::MAX)
            && b_len as u64 / 8 <= u64::from(u32::MAX), "matrix exceeds GPU generator indexing");
        ensure!(u64::from(m / 128) * u64::from(n / 128) <= u64::from(u32::MAX),
            "too many production groups");
        let config = MiningConfiguration {
            common_dim: k,
            rank: KRYPTEX_RANK,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&KRYPTEX_ROWS)?,
            cols_pattern: PeriodicPattern::from_list(&KRYPTEX_COLS)?,
            moe: None,
        };
        let mut key_material = Vec::with_capacity(128);
        key_material.extend_from_slice(&header.to_bytes());
        key_material.extend_from_slice(&config.to_bytes());
        let job_key = blake3::hash(&key_material);
        let seed = matrix_attempt_seed(job_key.as_bytes(), attempt);
        let a = Matrix::generated(seed, KRYPTEX_DOMAIN_A, a_len);
        let bt = Matrix::generated(seed, KRYPTEX_DOMAIN_BT, b_len);
        Ok(Self { header, config, m, n, a, bt,
            commitment_cache: Arc::new(OnceLock::new()) })
    }

    /// Returns row/column offsets without computing the tile transcript.
    pub fn tile_offsets(&self) -> (Vec<u32>, Vec<u32>) {
        let rows = (0..self.m)
            .filter(|&offset| self.config.rows_pattern.offset_is_valid(offset))
            .collect();
        let cols = (0..self.n)
            .filter(|&offset| self.config.cols_pattern.offset_is_valid(offset))
            .collect();
        (rows, cols)
    }

    /// Creates the metadata needed by a GPU kernel before it computes a transcript.
    pub fn empty_tile(&self, t_rows: u32, t_cols: u32) -> Result<TileResult> {
        let rows = self.config.rows_pattern.indices_with_offset(t_rows);
        let cols = self.config.cols_pattern.indices_with_offset(t_cols);
        ensure!(self.config.rows_pattern.offset_is_valid(t_rows), "invalid row tile offset {t_rows}");
        ensure!(self.config.cols_pattern.offset_is_valid(t_cols), "invalid column tile offset {t_cols}");
        ensure!(rows.last().copied().unwrap_or(0) < self.m, "row tile exceeds m");
        ensure!(cols.last().copied().unwrap_or(0) < self.n, "column tile exceeds n");
        Ok(TileResult {
            t_rows,
            t_cols,
            rows,
            cols,
            transcript: [0; 16],
            digest: [0; 32],
        })
    }

    /// Computes the official unkeyed job key from the serialized header and configuration.
    pub fn job_key(&self) -> Hash256 {
        let mut job_material = Vec::with_capacity(128);
        job_material.extend_from_slice(&self.header.to_bytes());
        job_material.extend_from_slice(&self.config.to_bytes());
        pearl_blake3::blake3_digest(&job_material, None)
    }

    /// Creates a dense V3 problem and checks only structural facts that are independent of a job
    /// verifier. The official verifier remains authoritative for consensus validity.
    pub fn new(header: IncompleteBlockHeader, config: MiningConfiguration, m: u32, n: u32,
        a: Vec<i8>, bt: Vec<i8>) -> Result<Self> {
        Self::new_matrices(header, config, m, n, a.into(), bt.into())
    }

    pub fn new_matrices(
        header: IncompleteBlockHeader,
        config: MiningConfiguration,
        m: u32,
        n: u32,
        a: Matrix,
        bt: Matrix,
    ) -> Result<Self> {
        ensure!(
            config.moe.is_none(),
            "M1 reference only supports dense V3 jobs"
        );
        let k = config.common_dim as usize;
        ensure!(
            m > 0 && n > 0 && k > 0,
            "problem dimensions must be nonzero"
        );
        ensure!(a.len() == m as usize * k, "A length does not match m*k");
        ensure!(bt.len() == n as usize * k, "B^T length does not match n*k");
        ensure!(
            a.iter().all(|&x| (-64..=64).contains(&x)),
            "A contains a value outside [-64, 64]"
        );
        ensure!(
            bt.iter().all(|&x| (-64..=64).contains(&x)),
            "B^T contains a value outside [-64, 64]"
        );
        Ok(Self {
            header,
            config,
            m,
            n,
            a,
            bt,
            commitment_cache: Arc::new(OnceLock::new()),
        })
    }

    pub fn k(&self) -> usize {
        self.config.common_dim as usize
    }

    pub fn commitment(&self) -> Commitment {
        self.commitment_cache
            .get_or_init(|| {
                let job_key = self.job_key();
                let root_a = keyed_matrix_root(&self.a, job_key);
                let root_b = keyed_matrix_root(&self.bt, job_key);
                self.commitment_from_roots(root_a, root_b)
            })
            .clone()
    }

    fn commitment_from_roots(&self, root_a: Hash256, root_b: Hash256) -> Commitment {
        let job_key = self.job_key();
                let bound_a = bind_root_a(&root_a, self.m);
                let bound_b = bind_root_b(&root_b, self.n);
                let b_noise_seed =
                    pearl_blake3::blake3_digest(&[&job_key[..], &bound_b[..]].concat(), None);
                let a_noise_seed =
                    pearl_blake3::blake3_digest(&[&b_noise_seed[..], &bound_a[..]].concat(), None);
                Commitment {
                    job_key,
                    root_a,
                    root_b,
                    bound_a,
                    bound_b,
                    b_noise_seed,
                    a_noise_seed,
                }
    }

    /// Installs accelerator roots. Device setup must pass CPU comparison at startup;
    /// submitted proofs still pass the independent official CPU verifier.
    pub fn install_accelerated_roots(&self, root_a: Hash256, root_b: Hash256) -> Result<()> {
        let commitment = self.commitment_from_roots(root_a, root_b);
        let cached = self.commitment_cache.get_or_init(|| commitment.clone());
        ensure!(*cached == commitment, "accelerator commitment conflicts with cached roots");
        Ok(())
    }

    /// Compact official noise permutations used by the resident GPU preparation path.
    pub fn noise_permutations(&self) -> (Vec<u32>, Vec<u32>) {
        use zk_pow::circuit::pearl_noise::generate_permutation_matrix;
        let commitment = self.commitment();
        let mut label_a = [0u8; 32];
        let mut label_b = [0u8; 32];
        label_a[..8].copy_from_slice(b"A_tensor");
        label_b[..8].copy_from_slice(b"B_tensor");
        let a = generate_permutation_matrix(&label_a, &commitment.a_noise_seed, self.k(), self.config.rank as usize);
        let b = generate_permutation_matrix(&label_b, &commitment.b_noise_seed, self.k(), self.config.rank as usize);
        (a.into_iter().flatten().collect(), b.into_iter().flatten().collect())
    }

    /// Returns the selected A rows and B^T columns after applying the official V3 noise.
    ///
    /// This is the host/device boundary for the M2 oracle. The noise construction and the
    /// operand addition remain the official reference implementation; CUDA only performs the
    /// resulting i8 dot products and transcript fold.
    pub fn noised_tile_operands(&self, tile: &TileResult) -> Result<(Vec<i8>, Vec<i8>)> {
        Ok(self.noised_tiles_operands(std::slice::from_ref(tile))?.remove(0))
    }

    /// Generate official noise once for the union of rows/columns in a bounded batch.
    /// Repeated A rows and the K-wide permutations are reused across adjacent tiles.
    pub fn noised_tiles_operands(&self, tiles: &[TileResult]) -> Result<Vec<(Vec<i8>, Vec<i8>)>> {
        use std::collections::BTreeSet;
        if tiles.is_empty() { return Ok(Vec::new()); }
        let mut rows = BTreeSet::new();
        let mut cols = BTreeSet::new();
        for tile in tiles {
            ensure!(tile.rows == self.config.rows_pattern.indices_with_offset(tile.t_rows), "tile rows do not match problem");
            ensure!(tile.cols == self.config.cols_pattern.indices_with_offset(tile.t_cols), "tile cols do not match problem");
            ensure!(tile.rows.iter().all(|&r| r < self.m) && tile.cols.iter().all(|&c| c < self.n), "tile indices out of bounds");
            rows.extend(tile.rows.iter().map(|&r| r as usize));
            cols.extend(tile.cols.iter().map(|&c| c as usize));
        }
        let rows: Vec<usize> = rows.into_iter().collect();
        let cols: Vec<usize> = cols.into_iter().collect();
        let commitment = self.commitment();
        let noise = compute_noise_for_indices(self.k(), self.config.rank as usize,
            (commitment.b_noise_seed, commitment.a_noise_seed), &rows, &cols);
        let combine = |indices: &[usize], matrix: &[i8], noise: &[Vec<i8>]| -> Result<Vec<Vec<i8>>> {
            indices.iter().zip(noise).map(|(&index, row_noise)| {
                matrix[index * self.k()..(index + 1) * self.k()].iter().zip(row_noise)
                    .map(|(&v, &n)| {
                        let value = v as i32 + n as i32;
                        ensure!((-128..=127).contains(&value), "noised operand does not fit i8");
                        Ok(value as i8)
                    }).collect()
            }).collect()
        };
        let a = combine(&rows, &self.a, &noise.a)?;
        let bt = combine(&cols, &self.bt, &noise.b)?;
        Ok(tiles.iter().map(|tile| {
            let selected_a = tile.rows.iter().flat_map(|&r| a[rows.binary_search(&(r as usize)).unwrap()].iter().copied()).collect();
            let selected_bt = tile.cols.iter().flat_map(|&c| bt[cols.binary_search(&(c as usize)).unwrap()].iter().copied()).collect();
            (selected_a, selected_bt)
        }).collect())
    }

    /// Computes one tile in the official reference order.
    pub fn tile(&self, t_rows: u32, t_cols: u32) -> Result<TileResult> {
        let rows = self.config.rows_pattern.indices_with_offset(t_rows);
        let cols = self.config.cols_pattern.indices_with_offset(t_cols);
        ensure!(
            self.config.rows_pattern.offset_is_valid(t_rows),
            "invalid row tile offset {t_rows}"
        );
        ensure!(
            self.config.cols_pattern.offset_is_valid(t_cols),
            "invalid column tile offset {t_cols}"
        );
        ensure!(
            rows.last().copied().unwrap_or(0) < self.m,
            "row tile exceeds m"
        );
        ensure!(
            cols.last().copied().unwrap_or(0) < self.n,
            "column tile exceeds n"
        );

        let commitment = self.commitment();
        let row_usize: Vec<usize> = rows.iter().map(|&x| x as usize).collect();
        let col_usize: Vec<usize> = cols.iter().map(|&x| x as usize).collect();
        let noise = compute_noise_for_indices(
            self.k(),
            self.config.rank as usize,
            (commitment.b_noise_seed, commitment.a_noise_seed),
            &row_usize,
            &col_usize,
        );

        let h = rows.len();
        let w = cols.len();
        let rank = self.config.rank as usize;
        let effective_k = self.config.dot_product_length();
        ensure!(
            effective_k.is_multiple_of(rank),
            "official dot product length is not rank aligned"
        );
        let mut acc = vec![0i32; h * w];
        let mut transcript = [0u32; 16];

        for end in (rank..=effective_k).step_by(rank) {
            for (u, &row) in rows.iter().enumerate() {
                for (v, &col) in cols.iter().enumerate() {
                    let mut sum = 0i32;
                    for l in end - rank..end {
                        let av = self.a[row as usize * self.k() + l] as i32 + noise.a[u][l] as i32;
                        let bv = self.bt[col as usize * self.k() + l] as i32 + noise.b[v][l] as i32;
                        sum += av * bv;
                    }
                    acc[u * w + v] += sum;
                }
            }
            let fold = acc.iter().fold(0u32, |value, &cell| value ^ cell as u32);
            let slot = (end / rank - 1) % 16;
            transcript[slot] = transcript[slot].rotate_left(13) ^ fold;
        }

        let digest = compute_jackpot_hash(&transcript, commitment.a_noise_seed);
        Ok(TileResult {
            t_rows,
            t_cols,
            rows,
            cols,
            transcript,
            digest,
        })
    }

    /// Enumerates tiles in the same row-major pattern order as the official miner.
    pub fn tiles(&self) -> Result<Vec<TileResult>> {
        let mut result = Vec::new();
        for t_rows in 0..self.m {
            if !self.config.rows_pattern.offset_is_valid(t_rows) {
                continue;
            }
            for t_cols in 0..self.n {
                if !self.config.cols_pattern.offset_is_valid(t_cols) {
                    continue;
                }
                result.push(self.tile(t_rows, t_cols)?);
            }
        }
        Ok(result)
    }

    /// Builds the official dense `PlainProof` for a tile.
    pub fn proof_for_tile(&self, tile: &TileResult) -> Result<PlainProof> {
        let commitment = self.commitment();
        ensure!(
            tile.rows == self.config.rows_pattern.indices_with_offset(tile.t_rows),
            "tile rows do not match problem"
        );
        ensure!(
            tile.cols == self.config.cols_pattern.indices_with_offset(tile.t_cols),
            "tile cols do not match problem"
        );
        Ok(PlainProof {
            m: self.m as usize,
            n: self.n as usize,
            k: self.k(),
            noise_rank: self.config.rank as usize,
            a: matrix_proof(
                &self.a,
                self.m as usize,
                self.k(),
                &tile.rows,
                commitment.job_key,
            )?,
            bt: matrix_proof(
                &self.bt,
                self.n as usize,
                self.k(),
                &tile.cols,
                commitment.job_key,
            )?,
            moe: None,
        })
    }

    /// Runs the official V3 eligibility and plain-proof verifier for a tile.
    pub fn verify_tile(&self, tile: &TileResult, nbits_override: Option<u32>) -> Result<()> {
        let proof = self.proof_for_tile(tile)?;
        check_cert_version_eligible(3, &proof)?;
        verify_plain_proof(&self.header, &proof, nbits_override, SeedDerivation::Salted)
    }

    /// Returns whether a digest meets the official V3 bound for this configuration.
    pub fn digest_meets(&self, tile: &TileResult, nbits: u32) -> bool {
        U256::from_little_endian(&tile.digest) <= extract_difficulty_bound(nbits, &self.config)
    }

    /// Verifies a tile against a pool target using the official compact-target verifier path.
    pub fn verify_tile_against_target(&self, tile: &TileResult, target: U256) -> Result<()> {
        self.verified_proof_against_target(tile, target).map(|_| ())
    }

    /// Builds and verifies once; submission serializes this exact proof rather than
    /// rebuilding both full matrix Merkle trees after successful verification.
    pub fn verified_proof_against_target(&self, tile: &TileResult, target: U256) -> Result<PlainProof> {
        let proof = self.proof_for_tile(tile)?;
        check_cert_version_eligible(3, &proof)?;
        verify_plain_proof(
            &self.header,
            &proof,
            Some(compact_from_target(target)),
            SeedDerivation::Salted,
        )?;
        Ok(proof)
    }
}

/// Encodes a 256-bit pool target as the compact `nbits` form accepted by the official verifier.
pub fn compact_from_target(target: U256) -> u32 {
    if target.is_zero() {
        return 0;
    }
    let mut size = target.bits().div_ceil(8);
    let mut mantissa: u32 = if size <= 3 {
        (target.low_u64() << (8 * (3 - size))) as u32
    } else {
        (target >> (8 * (size - 3))).low_u64() as u32
    };
    if mantissa & 0x0080_0000 != 0 {
        mantissa >>= 8;
        size += 1;
    }
    ((size as u32) << 24) | (mantissa & 0x00ff_ffff)
}

/// Computes the share bound used by the pool-facing miner.
///
/// Kryptex sends an absolute target, while the official verifier receives the equivalent compact
/// `nbits`. The compact representation can round down, so the miner must use the smaller of the
/// pool target and the decoded compact target before applying Pearl's rank penalty.
pub fn share_bound(target: U256, config: &MiningConfiguration) -> Option<U256> {
    let nbits = compact_from_target(target);
    if nbits == 0 {
        return None;
    }
    let compact_target = nbits_to_difficulty(nbits);
    let effective_target = compact_target.min(target);
    penalized_target_bound(effective_target, config)
}

fn keyed_matrix_root(matrix: &[i8], key: Hash256) -> Hash256 {
    let raw: Vec<u8> = matrix.iter().map(|&x| x as u8).collect();
    let padded = pearl_blake3::pad_to_chunk_boundary(&raw);
    pearl_blake3::blake3_digest(&padded, Some(key))
}

const KRYPTEX_DOMAIN_A: u64 = 0x7370_6d2d_612d_3031;
const KRYPTEX_DOMAIN_BT: u64 = 0x7370_6d2d_622d_3031;

fn matrix_attempt_seed(job_key: &[u8; 32], attempt: u64) -> u64 {
    let mut material = Vec::with_capacity(64);
    material.extend_from_slice(job_key);
    material.extend_from_slice(KRYPTEX_FILL_DOMAIN);
    if attempt != 0 {
        material.extend_from_slice(b"/attempt/");
        material.extend_from_slice(&attempt.to_le_bytes());
    }
    let digest = blake3::hash(&material);
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap())
}

fn fill_int7(seed: u64, domain: u64, len: usize) -> Vec<i8> {
    let mut state = seed ^ domain;
    let mut output = Vec::with_capacity(len);
    while output.len() < len {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut word = state;
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        word ^= word >> 31;
        for byte in word.to_le_bytes() {
            if output.len() == len {
                break;
            }
            output.push((byte & 0x7f) as i8 - 64);
        }
    }
    output
}

fn matrix_proof(
    matrix: &[i8],
    rows: usize,
    k: usize,
    selected: &[u32],
    key: Hash256,
) -> Result<MatrixMerkleProof> {
    let raw: Vec<u8> = matrix.iter().map(|&x| x as u8).collect();
    let padded = pearl_blake3::pad_to_chunk_boundary(&raw);
    let tree = pearl_blake3::MerkleTree::new(&padded, key);
    let leaf_indices = pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(
        &selected.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        (rows, k),
    );
    let proof = tree.get_multileaf_proof(&leaf_indices);
    Ok(MatrixMerkleProof {
        proof,
        row_indices: selected.iter().map(|&x| x as usize).collect(),
    })
}

/// Deterministic test-only dense input. The formula is a vector generator, not a consensus rule.
pub fn test_problem() -> Result<Problem> {
    let m = 64u32;
    let n = 64u32;
    let k = 1024u32;
    let rank = 64u16;
    let config = MiningConfiguration {
        common_dim: k,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 16, 24])?,
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 8, 9, 32, 33, 40, 41])?,
        moe: None,
    };
    let header = IncompleteBlockHeader {
        version: 0x2000_0000,
        prev_block: [0x11; 32],
        merkle_root: [0x22; 32],
        timestamp: 0x6666_6666,
        nbits: 0x207f_ffff,
    };
    let a: Vec<i8> = (0..m as usize * k as usize)
        .map(|i| (((i * 13 + 7) % 129) as i32 - 64) as i8)
        .collect();
    let bt: Vec<i8> = (0..n as usize * k as usize)
        .map(|i| (((i * 17 + 3) % 129) as i32 - 64) as i8)
        .collect();
    Problem::new(header, config, m, n, a, bt)
}

/// Changes only the serialized header timestamp for a deterministic candidate vector. The
/// official V3 reference has no separate nonce field in `IncompleteBlockHeader`; this helper is
/// intentionally test-only and does not define a network nonce encoding.
pub fn test_header_variant(problem: &Problem, timestamp: u32) -> Result<Problem> {
    let mut header = problem.header;
    header.timestamp = timestamp;
    Problem::new(
        header,
        problem.config,
        problem.m,
        problem.n,
        problem.a.to_vec(),
        problem.bt.to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_attempts_preserve_initial_vector_and_change_subsequent_work() {
        let key = [42; 32];
        let old = blake3::hash(&[&key[..], KRYPTEX_FILL_DOMAIN].concat());
        assert_eq!(matrix_attempt_seed(&key, 0), u64::from_le_bytes(old.as_bytes()[..8].try_into().unwrap()));
        let seeds: std::collections::BTreeSet<_> = (0..32).map(|i| matrix_attempt_seed(&key, i)).collect();
        assert_eq!(seeds.len(), 32);
        let a = fill_int7(matrix_attempt_seed(&key, 1), KRYPTEX_DOMAIN_A, 1024);
        let b = fill_int7(matrix_attempt_seed(&key, 2), KRYPTEX_DOMAIN_A, 1024);
        assert_ne!(a, b);
        assert!(a.iter().chain(&b).all(|x| (-64..=63).contains(x)));
    }

    #[test]
    fn batch_operands_match_official_noise_for_each_tile() {
        let problem = test_problem().unwrap();
        let (rows, cols) = problem.tile_offsets();
        let tiles: Vec<_> = rows.iter().take(3).flat_map(|&r| cols.iter().take(5).map(move |&c| (r,c)))
            .map(|(r,c)| problem.empty_tile(r,c).unwrap()).collect();
        let batch = problem.noised_tiles_operands(&tiles).unwrap();
        let commitment = problem.commitment();
        for (tile, (a, bt)) in tiles.iter().zip(batch) {
            let rows: Vec<_> = tile.rows.iter().map(|&v| v as usize).collect();
            let cols: Vec<_> = tile.cols.iter().map(|&v| v as usize).collect();
            let noise = compute_noise_for_indices(problem.k(), problem.config.rank as usize,
                (commitment.b_noise_seed, commitment.a_noise_seed), &rows, &cols);
            for (actual, indices, matrix, noise) in [(&a, &rows, &problem.a, &noise.a), (&bt, &cols, &problem.bt, &noise.b)] {
                for (i, &index) in indices.iter().enumerate() {
                    for k in 0..problem.k() {
                        assert_eq!(actual[i * problem.k() + k] as i32,
                            matrix[index * problem.k() + k] as i32 + noise[i][k] as i32);
                    }
                }
            }
        }
        assert!(problem.noised_tiles_operands(&[]).unwrap().is_empty());
        let mut bad = tiles[0].clone();
        bad.rows[0] += 1;
        assert!(problem.noised_tiles_operands(&[bad]).is_err());
    }

    #[test]
    fn official_verifier_accepts_reference_tile_and_rejects_header_mutation() {
        let problem = test_problem().unwrap();
        let tile = problem.tile(0, 0).unwrap();
        problem.verify_tile(&tile, None).unwrap();

        let mut bad_header = problem.header;
        bad_header.timestamp ^= 1;
        let proof = problem.proof_for_tile(&tile).unwrap();
        assert!(verify_plain_proof(&bad_header, &proof, None, SeedDerivation::Salted).is_err());
    }

    #[test]
    fn mutated_proof_is_rejected_before_submission() {
        let problem = test_problem().unwrap();
        let tile = problem.tile(0, 0).unwrap();
        let mut proof = problem.proof_for_tile(&tile).unwrap();
        proof.noise_rank = 32;
        check_cert_version_eligible(3, &proof).unwrap();
        assert!(verify_plain_proof(&problem.header, &proof, None, SeedDerivation::Salted).is_err());
    }

    #[test]
    fn tile_order_is_row_major_and_partition_is_exact() {
        let problem = test_problem().unwrap();
        let tiles = problem.tiles().unwrap();
        assert_eq!(tiles.len(), 128);
        assert_eq!((tiles[0].t_rows, tiles[0].t_cols), (0, 0));
        assert_eq!(tiles[1].t_rows, 0);
        assert!(tiles[1].t_cols > 0);
        assert!(tiles[8].t_rows > 0);

        let mut seen = vec![0u8; problem.m as usize * problem.n as usize];
        for tile in tiles {
            for &row in &tile.rows {
                for &col in &tile.cols {
                    seen[row as usize * problem.n as usize + col as usize] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&x| x == 1));
    }

    #[test]
    fn compact_target_bound_is_not_saturated_in_miner_helper() {
        let problem = test_problem().unwrap();
        assert!(penalized_target_bound(U256::MAX, &problem.config).is_none());
    }

    #[test]
    fn rank128_cpu_preflight_used_by_sm120_randomized_case() {
        let k = 1024usize;
        let config = MiningConfiguration {
            common_dim: k as u32,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ])
            .unwrap(),
            moe: None,
        };
        let a = (0..8 * k)
            .map(|i| (((i * 19 + 5) % 129) as i32 - 64) as i8)
            .collect();
        let bt = (0..16 * k)
            .map(|i| (((i * 23 + 11) % 129) as i32 - 64) as i8)
            .collect();
        let problem = Problem::new(
            IncompleteBlockHeader {
                version: 0x2000_0000,
                prev_block: [0x51; 32],
                merkle_root: [0x62; 32],
                timestamp: 0x7777_0002,
                nbits: 0x207f_ffff,
            },
            config,
            8,
            16,
            a,
            bt,
        )
        .unwrap();
        let tile = problem.tile(0, 0).unwrap();
        assert_eq!(tile.rows.len(), 8);
        assert_eq!(tile.cols.len(), 16);
        assert!(tile.transcript.iter().any(|&word| word != 0));
        assert_eq!(
            tile.digest,
            compute_jackpot_hash(&tile.transcript, problem.commitment().a_noise_seed)
        );
    }
}

#[cfg(test)]
mod deferred_matrix_tests {
    use super::*;
    #[test]
    fn deferred_matrix_matches_eager_bytes_and_shares_storage() {
        for len in [1, 7, 8, 9, 1024, 4097] {
            let matrix = Matrix::generated(u64::MAX, 0x123456789abcdef0, len);
            let clone = matrix.clone();
            assert!(!matrix.is_materialized());
            assert_eq!(&*matrix, fill_int7(u64::MAX, 0x123456789abcdef0, len));
            assert!(clone.is_materialized());
            assert_eq!(matrix.as_ptr(), clone.as_ptr());
        }
    }
    #[test]
    fn deferred_production_attempts_do_not_allocate_host_matrices() {
        let header = test_problem().unwrap().header;
        let first = Problem::from_kryptex_header_attempt_deferred(header.clone(), 0).unwrap();
        let next = Problem::from_kryptex_header_attempt_deferred(header, 1).unwrap();
        assert!(!first.a.is_materialized() && !first.bt.is_materialized());
        assert!(!next.a.is_materialized() && !next.bt.is_materialized());
        assert_ne!(first.a.recipe(), next.a.recipe());
        assert_ne!(first.a.recipe(), first.bt.recipe());
    }
    #[test]
    fn configurable_shapes_keep_default_recipe_and_defer_large_matrices() {
        let header = test_problem().unwrap().header;
        let default = Problem::from_kryptex_header_attempt_deferred(header.clone(), 7).unwrap();
        let explicit = Problem::from_kryptex_header_attempt_shape(header.clone(), 7,
            KRYPTEX_M, KRYPTEX_N, KRYPTEX_K).unwrap();
        assert_eq!(default.job_key(), explicit.job_key());
        assert_eq!(default.a.recipe(), explicit.a.recipe());
        assert_eq!(default.bt.recipe(), explicit.bt.recipe());
        for k in [4096, 8192, 16384] {
            let p = Problem::from_kryptex_header_attempt_shape(header.clone(), 7, 262144, 262144, k).unwrap();
            assert_eq!(p.a.len(), 262144usize * k as usize);
            assert_eq!(p.bt.len(), p.a.len());
            assert!(!p.a.is_materialized() && !p.bt.is_materialized());
            assert_eq!(p.config.rank, 128);
            assert!(p.empty_tile(262080, 262080).is_ok());
        }
        for (m, n, k) in [(0, 128, 4096), (129, 128, 4096), (128, 128, 1024),
            (128, 128, 2112), (128, 128, 65664), (1 << 25, 128, 4096)] {
            assert!(Problem::from_kryptex_header_attempt_shape(header.clone(), 0, m, n, k).is_err());
        }
    }

    #[test]
    fn accelerator_roots_cannot_replace_an_existing_commitment() {
        let problem = test_problem().unwrap();
        let expected = problem.commitment();
        problem.install_accelerated_roots(expected.root_a, expected.root_b).unwrap();
        let mut wrong = expected.root_a;
        wrong[0] ^= 1;
        assert!(problem.install_accelerated_roots(wrong, expected.root_b).is_err());
        assert_eq!(problem.commitment(), expected);
    }
}
