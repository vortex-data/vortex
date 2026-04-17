//! Synthetic corpus generator.
//!
//! Emits `N` unit-normalized f32 vectors packed row-major into a single file.
//! Vectors are generated in batches from a deterministic RNG seed and written
//! through a 4 KB-aligned buffer so the resulting file is usable with
//! `O_DIRECT` on Linux.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use aligned_vec::AVec;
use aligned_vec::ConstAlign;
use anyhow::Context;
use anyhow::Result;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

/// Alignment of the write buffer. 4 KB matches the minimum O_DIRECT alignment
/// on ext4/xfs/btrfs on Linux.
pub const ALIGN: usize = 4096;

/// Generate and write `n_vectors` unit-normalized f32 vectors of dimension
/// `dim` to `path`. Returns the number of bytes written (which equals
/// `n_vectors * dim * 4`).
pub fn generate(path: &Path, n_vectors: u64, dim: usize, seed: u64) -> Result<u64> {
    anyhow::ensure!(dim > 0, "dim must be positive");
    let bytes_per_vec = dim
        .checked_mul(std::mem::size_of::<f32>())
        .context("dim too large")?;
    let total_bytes = (n_vectors as u128)
        .checked_mul(bytes_per_vec as u128)
        .context("corpus size overflow")?;
    anyhow::ensure!(
        total_bytes <= u64::MAX as u128,
        "corpus is too large to represent in u64"
    );
    let total_bytes = total_bytes as u64;

    // Batch size: enough that write syscalls are large, but small enough that
    // the aligned buffer stays in L2. Target ~4 MB per write.
    let batch_vecs = ((4 * 1024 * 1024) / bytes_per_vec).max(1);
    let batch_bytes = batch_vecs * bytes_per_vec;
    // Round batch bytes up to ALIGN (4 KB) so each write is 4 KB aligned.
    let buf_bytes = batch_bytes.div_ceil(ALIGN) * ALIGN;
    let mut buf: AVec<u8, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, buf_bytes);
    buf.resize(buf_bytes, 0u8);

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("failed to create {}", path.display()))?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut remaining = n_vectors;
    while remaining > 0 {
        let this_batch = remaining.min(batch_vecs as u64) as usize;

        // SAFETY: buf is allocated as u8 but was over-allocated with 4-byte
        // alignment (ALIGN >= 4). We fill only the first this_batch * dim f32
        // slots and then slice back out the exact number of bytes.
        let floats = unsafe {
            std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, this_batch * dim)
        };

        for v in floats.chunks_exact_mut(dim) {
            let mut norm_sq = 0.0f32;
            for x in v.iter_mut() {
                let g: f32 = rng.sample(StandardNormal);
                *x = g;
                norm_sq += g * g;
            }
            // Rare safeguard against a degenerate all-zero vector.
            if norm_sq <= f32::MIN_POSITIVE {
                v[0] = 1.0;
                continue;
            }
            let inv = 1.0 / norm_sq.sqrt();
            for x in v.iter_mut() {
                *x *= inv;
            }
        }

        let write_len = this_batch * bytes_per_vec;
        file.write_all(&buf[..write_len])
            .context("failed to write corpus batch")?;
        remaining -= this_batch as u64;
    }

    file.sync_all().context("failed to fsync corpus file")?;
    Ok(total_bytes)
}

/// Generate a single random unit-normalized vector. Used for the query.
pub fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn generated_vectors_are_unit_normalized() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("corpus.bin");
        let bytes = generate(&path, 1024, 64, 1)?;
        assert_eq!(bytes, 1024 * 64 * 4);

        let data = std::fs::read(&path)?;
        assert_eq!(data.len() as u64, bytes);
        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };
        for v in floats.chunks_exact(64) {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "norm {}", n);
        }
        Ok(())
    }
}
