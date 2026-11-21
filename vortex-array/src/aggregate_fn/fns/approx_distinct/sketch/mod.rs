// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A mergeable HyperLogLog sketch for estimating the number of distinct values.
//!
//! A sketch with precision `p` keeps `m = 2^p` one-byte registers. Each inserted value is hashed to
//! 64 bits: the top `p` bits pick a register, and the register keeps the largest rank seen, where
//! the rank is one more than the number of leading zeros in the remaining `64 - p` bits. Two
//! sketches of the same precision merge by taking the larger value of each register, so merging
//! is commutative, associative, and idempotent.
//!
//! The estimate uses Otmar Ertl's improved estimator from "New cardinality estimation algorithms
//! for HyperLogLog sketches" (2017). It is accurate across the whole range, from a handful of
//! values to billions, without the empirical bias tables of HyperLogLog++.
//!
//! # Serialization
//!
//! ```text
//! ┌──────────────┬───────────────┬──────────────────────────────────────────────┐
//! │ byte 0       │ byte 1        │ bytes 2..                                    │
//! ├──────────────┼───────────────┼──────────────────────────────────────────────┤
//! │ encoding     │ precision `p` │ dense:  `2^p` registers, one byte each       │
//! │ 0 = dense    │               │ sparse: (index: u32 LE, rank: u8) entries,   │
//! │ 1 = sparse   │               │         strictly increasing by index         │
//! └──────────────┴───────────────┴──────────────────────────────────────────────┘
//! ```
//!
//! The sparse encoding stores only non-zero registers, so a low-cardinality column costs a few
//! bytes instead of `2^p`. [`HyperLogLog::serialize`] picks whichever encoding is smaller.

#[cfg(test)]
mod tests;

use twox_hash::XxHash3_64;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

/// The smallest supported precision: 16 registers, about 26% standard error.
pub const MIN_PRECISION: u8 = 4;

/// The largest supported precision: 65,536 registers (64 KiB), about 0.4% standard error.
///
/// Sketches live in file footers, so this bounds what one column adds to them.
pub const MAX_PRECISION: u8 = 16;

/// `1 / (2 ln 2)`, the limit of the HyperLogLog bias-correction constant as `m` grows.
const ALPHA_INFINITY: f64 = 0.721_347_520_444_481_7;

const DENSE_ENCODING: u8 = 0;
const SPARSE_ENCODING: u8 = 1;

/// Encoding byte plus precision byte.
const HEADER_LEN: usize = 2;

const SPARSE_ENTRY_LEN: usize = size_of::<u32>() + size_of::<u8>();

/// A HyperLogLog sketch with one byte per register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperLogLog {
    precision: u8,
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Creates an empty sketch with `2^precision` zeroed registers.
    ///
    /// Fails if `precision` is outside [`MIN_PRECISION`]`..=`[`MAX_PRECISION`].
    pub fn try_new(precision: u8) -> VortexResult<Self> {
        validate_precision(precision)?;
        Ok(Self {
            precision,
            registers: vec![0; 1 << precision],
        })
    }

    /// Returns the precision `p` this sketch was created with.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Returns the largest rank a register can hold, `64 - p + 1`.
    fn max_rank(&self) -> u8 {
        64 - self.precision + 1
    }

    /// Hashes `value` and records it in the sketch.
    #[inline]
    pub fn insert(&mut self, value: &[u8]) {
        // The hash function and its seed are part of the persisted format. Changing either
        // makes merged sketches from old and new files count every shared value twice.
        self.insert_hash(XxHash3_64::oneshot(value));
    }

    /// Records a 64-bit hash in the sketch.
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        let index = usize::try_from(hash >> (64 - u32::from(self.precision)))
            .vortex_expect("a register index has at most 16 bits");
        let remaining_bits = hash << self.precision;
        // The shift zero-fills the low `p` bits, so all-zero remaining bits would otherwise give
        // rank 65 rather than the largest rank, `64 - p + 1`.
        let rank = if remaining_bits == 0 {
            self.max_rank()
        } else {
            u8::try_from(remaining_bits.leading_zeros() + 1).vortex_expect("a rank is at most 64")
        };
        let register = &mut self.registers[index];
        *register = (*register).max(rank);
    }

    /// Merges `other` into this sketch by keeping the larger value of each register.
    ///
    /// Fails if the two sketches have different precisions.
    pub fn merge(&mut self, other: &Self) -> VortexResult<()> {
        vortex_ensure_eq!(self.precision, other.precision);
        for (register, &other_register) in self.registers.iter_mut().zip(&other.registers) {
            *register = (*register).max(other_register);
        }
        Ok(())
    }

    /// Returns the estimated number of distinct values inserted, rounded to the nearest integer.
    pub fn estimate(&self) -> u64 {
        let max_rank = usize::from(self.max_rank());
        let mut rank_counts = vec![0u32; max_rank + 1];
        for &register in &self.registers {
            rank_counts[usize::from(register)] += 1;
        }

        let register_count = f64::from(1u32 << self.precision);
        let empty_registers = f64::from(rank_counts[0]);
        // `sigma(1)` is infinite, so an empty sketch is answered directly.
        if empty_registers == register_count {
            return 0;
        }

        // Ertl's improved raw estimator (Algorithm 6 in the paper), where `q = 64 - p` and
        // `rank_counts[k]` is the number of registers holding rank `k`.
        let saturated_registers = f64::from(rank_counts[max_rank]);
        let mut harmonic_sum =
            register_count * tau((register_count - saturated_registers) / register_count);
        for &count in rank_counts[1..max_rank].iter().rev() {
            harmonic_sum = 0.5 * (harmonic_sum + f64::from(count));
        }
        harmonic_sum += register_count * sigma(empty_registers / register_count);

        let estimate = ALPHA_INFINITY * register_count * register_count / harmonic_sum;
        // A fully saturated sketch estimates infinity, which `as` saturates to `u64::MAX`.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the estimate is non-negative and saturating at u64::MAX is intended"
        )]
        let rounded = estimate.round() as u64;
        rounded
    }

    /// Serializes the sketch in the smaller of the dense and sparse encodings.
    pub fn serialize(&self) -> Vec<u8> {
        let non_empty_registers = self
            .registers
            .iter()
            .filter(|&&register| register != 0)
            .count();
        if non_empty_registers * SPARSE_ENTRY_LEN < self.registers.len() {
            self.serialize_sparse(non_empty_registers)
        } else {
            self.serialize_dense()
        }
    }

    fn serialize_dense(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LEN + self.registers.len());
        bytes.push(DENSE_ENCODING);
        bytes.push(self.precision);
        bytes.extend_from_slice(&self.registers);
        bytes
    }

    fn serialize_sparse(&self, non_empty_registers: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LEN + non_empty_registers * SPARSE_ENTRY_LEN);
        bytes.push(SPARSE_ENCODING);
        bytes.push(self.precision);
        for (index, &register) in self.registers.iter().enumerate() {
            if register != 0 {
                let index =
                    u32::try_from(index).vortex_expect("a register index has at most 16 bits");
                bytes.extend_from_slice(&index.to_le_bytes());
                bytes.push(register);
            }
        }
        bytes
    }

    /// Parses a sketch written by [`HyperLogLog::serialize`].
    ///
    /// Fails on an unknown encoding, an unsupported precision, a payload of the wrong length,
    /// or a register value that no insert could produce.
    pub fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        let [encoding, precision, payload @ ..] = bytes else {
            vortex_bail!(
                "approx_distinct sketch needs a {HEADER_LEN}-byte header, got {} bytes",
                bytes.len()
            );
        };
        let mut sketch = Self::try_new(*precision)?;
        match *encoding {
            DENSE_ENCODING => sketch.read_dense(payload)?,
            SPARSE_ENCODING => sketch.read_sparse(payload)?,
            unknown => vortex_bail!("unknown approx_distinct sketch encoding {unknown}"),
        }
        Ok(sketch)
    }

    fn read_dense(&mut self, payload: &[u8]) -> VortexResult<()> {
        vortex_ensure_eq!(payload.len(), self.registers.len());
        let max_rank = self.max_rank();
        if let Some(&invalid) = payload.iter().find(|&&register| register > max_rank) {
            vortex_bail!("approx_distinct register value {invalid} exceeds the maximum {max_rank}");
        }
        self.registers.copy_from_slice(payload);
        Ok(())
    }

    fn read_sparse(&mut self, payload: &[u8]) -> VortexResult<()> {
        let (entries, remainder) = payload.as_chunks::<SPARSE_ENTRY_LEN>();
        vortex_ensure!(
            remainder.is_empty(),
            "approx_distinct sparse payload has {} trailing bytes",
            remainder.len()
        );
        let max_rank = self.max_rank();
        let mut previous_index: Option<usize> = None;
        for &[index_0, index_1, index_2, index_3, rank] in entries {
            let index = usize::try_from(u32::from_le_bytes([index_0, index_1, index_2, index_3]))
                .map_err(|_| {
                vortex_err!("approx_distinct register index does not fit in usize")
            })?;
            vortex_ensure!(
                index < self.registers.len(),
                "approx_distinct register index {index} is out of range for {} registers",
                self.registers.len()
            );
            // Strictly increasing also rejects duplicates, so a corrupt entry can't overwrite one.
            vortex_ensure!(
                previous_index.is_none_or(|previous| previous < index),
                "approx_distinct sparse indices must be strictly increasing"
            );
            vortex_ensure!(
                (1..=max_rank).contains(&rank),
                "approx_distinct sparse rank {rank} is outside 1..={max_rank}"
            );
            self.registers[index] = rank;
            previous_index = Some(index);
        }
        Ok(())
    }
}

pub(super) fn validate_precision(precision: u8) -> VortexResult<()> {
    vortex_ensure!(
        (MIN_PRECISION..=MAX_PRECISION).contains(&precision),
        "approx_distinct precision must be within {MIN_PRECISION}..={MAX_PRECISION}, got {precision}"
    );
    Ok(())
}

/// Ertl's `sigma` series, which corrects for empty registers; infinite at `x = 1`.
fn sigma(mut x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    let mut weight = 1.0;
    let mut sum = x;
    loop {
        x *= x;
        let previous = sum;
        sum += x * weight;
        weight += weight;
        if sum == previous {
            return sum;
        }
    }
}

/// Ertl's `tau` series, which corrects for registers at the largest rank; zero at `x = 0` and `1`.
fn tau(mut x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let mut weight = 1.0;
    let mut sum = 1.0 - x;
    loop {
        x = x.sqrt();
        let previous = sum;
        weight *= 0.5;
        sum -= (1.0 - x) * (1.0 - x) * weight;
        if sum == previous {
            return sum / 3.0;
        }
    }
}
