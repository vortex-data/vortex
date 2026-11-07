// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A HyperLogLog sketch over 64-bit hashes.

use std::f64::consts::LN_2;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

/// Smallest supported precision: 16 registers.
pub(super) const MIN_PRECISION: u8 = 4;
/// Largest supported precision: 65,536 registers, so a register index fits in a `u16`.
pub(super) const MAX_PRECISION: u8 = 16;

/// Serialized tag of a sketch holding one byte per register.
const DENSE_TAG: u8 = 0;
/// Serialized tag of a sketch holding `(u16 LE index, u8 value)` pairs for the non-zero
/// registers only, in increasing index order.
const SPARSE_TAG: u8 = 1;
const SPARSE_ENTRY_LEN: usize = 3;

/// A HyperLogLog sketch with `2^precision` one-byte registers.
///
/// A hash's top `precision` bits select a register, which keeps the largest rank (one plus the
/// number of leading zeros) of the remaining `64 - precision` bits seen so far. The estimate uses
/// Ertl's improved raw estimator ("New cardinality estimation algorithms for HyperLogLog
/// sketches", 2017), which needs no empirical bias correction and is accurate from empty sketches
/// up to `2^64` distinct values, with a relative standard error of about `1.04 / sqrt(2^precision)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HllSketch {
    precision: u8,
    registers: Vec<u8>,
}

impl HllSketch {
    /// An empty sketch with `2^precision` registers.
    ///
    /// # Errors
    ///
    /// Returns an error if `precision` is outside `4..=16`.
    pub fn try_new(precision: u8) -> VortexResult<Self> {
        vortex_ensure!(
            (MIN_PRECISION..=MAX_PRECISION).contains(&precision),
            "HyperLogLog precision {precision} is outside {MIN_PRECISION}..={MAX_PRECISION}"
        );
        Ok(Self {
            precision,
            registers: vec![0; 1 << precision],
        })
    }

    /// The number of index bits.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Adds a 64-bit hash to the sketch.
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the index has at most 16 bits"
        )]
        let index = (hash >> (64 - self.precision)) as usize;
        // Shift the index bits out and set a sentinel bit, so the rank is at most
        // `64 - precision + 1` even when all remaining bits are zero.
        let rest = (hash << self.precision) | (1 << (self.precision - 1));
        #[expect(clippy::cast_possible_truncation, reason = "the rank is at most 61")]
        let rank = (rest.leading_zeros() + 1) as u8;
        // `index < 2^precision == registers.len()`, so the bounds check never fails.
        let register = &mut self.registers[index];
        if rank > *register {
            *register = rank;
        }
    }

    /// Adds every hash in `hashes` to the sketch.
    pub fn insert_hashes(&mut self, hashes: &[u64]) {
        for &hash in hashes {
            self.insert_hash(hash);
        }
    }

    /// Merges `other` into `self`, so `self` estimates the union of both inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the sketches have different precisions.
    pub fn merge(&mut self, other: &Self) -> VortexResult<()> {
        vortex_ensure_eq!(self.precision, other.precision);
        for (register, &other) in self.registers.iter_mut().zip(&other.registers) {
            *register = (*register).max(other);
        }
        Ok(())
    }

    /// Whether no hash was added.
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&register| register == 0)
    }

    /// The estimated number of distinct hashes added.
    pub fn estimate(&self) -> u64 {
        let registers = self.registers.len() as f64;
        let bits = 64 - usize::from(self.precision);

        // Histogram of register values, which range over `0..=bits + 1`.
        let mut counts = vec![0usize; bits + 2];
        for &register in &self.registers {
            counts[usize::from(register)] += 1;
        }
        if counts[0] == self.registers.len() {
            return 0;
        }

        let mut sum = registers * tau(1.0 - counts[bits + 1] as f64 / registers);
        for &count in counts[1..=bits].iter().rev() {
            sum = 0.5 * (sum + count as f64);
        }
        sum += registers * sigma(counts[0] as f64 / registers);

        let alpha_inf = 1.0 / (2.0 * LN_2);
        let estimate = (alpha_inf * registers * registers / sum).round();
        // The cast saturates, and `estimate` is finite and positive here.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rounded positive estimate, saturating"
        )]
        let estimate = estimate as u64;
        estimate.max(1)
    }

    /// Serializes the sketch, sparsely when few registers are set.
    pub fn serialize(&self) -> Vec<u8> {
        let non_zero = self.registers.iter().filter(|&&r| r != 0).count();
        if non_zero * SPARSE_ENTRY_LEN < self.registers.len() {
            let mut bytes = Vec::with_capacity(1 + non_zero * SPARSE_ENTRY_LEN);
            bytes.push(SPARSE_TAG);
            for (index, &register) in self.registers.iter().enumerate() {
                if register != 0 {
                    // `index < 2^16` since the precision is at most 16.
                    #[expect(clippy::cast_possible_truncation, reason = "precision <= 16")]
                    bytes.extend_from_slice(&(index as u16).to_le_bytes());
                    bytes.push(register);
                }
            }
            bytes
        } else {
            let mut bytes = Vec::with_capacity(1 + self.registers.len());
            bytes.push(DENSE_TAG);
            bytes.extend_from_slice(&self.registers);
            bytes
        }
    }

    /// Parses a sketch of `precision` written by [`Self::serialize`].
    ///
    /// # Errors
    ///
    /// Returns an error if `bytes` is not a valid sketch of `precision`.
    pub fn deserialize(precision: u8, bytes: &[u8]) -> VortexResult<Self> {
        let mut sketch = Self::try_new(precision)?;
        let max_rank = 64 - precision + 1;
        let Some((&tag, body)) = bytes.split_first() else {
            vortex_bail!("empty HyperLogLog sketch");
        };
        match tag {
            DENSE_TAG => {
                vortex_ensure_eq!(body.len(), sketch.registers.len());
                sketch.registers.copy_from_slice(body);
            }
            SPARSE_TAG => {
                let (entries, remainder) = body.as_chunks::<SPARSE_ENTRY_LEN>();
                vortex_ensure!(
                    remainder.is_empty(),
                    "HyperLogLog sparse sketch has {} trailing bytes",
                    remainder.len()
                );
                for &[lo, hi, register] in entries {
                    let index = usize::from(u16::from_le_bytes([lo, hi]));
                    vortex_ensure!(
                        index < sketch.registers.len(),
                        "HyperLogLog register index {index} out of bounds"
                    );
                    sketch.registers[index] = register;
                }
            }
            tag => vortex_bail!("unknown HyperLogLog sketch tag {tag}"),
        }
        vortex_ensure!(
            sketch
                .registers
                .iter()
                .all(|&register| register <= max_rank),
            "HyperLogLog register exceeds the maximum rank {max_rank}"
        );
        Ok(sketch)
    }
}

/// Ertl's `sigma` function, for the registers that are still zero.
fn sigma(mut fraction: f64) -> f64 {
    if fraction == 1.0 {
        return f64::INFINITY;
    }
    let mut weight = 1.0;
    let mut sum = fraction;
    loop {
        fraction *= fraction;
        let previous = sum;
        sum += fraction * weight;
        weight += weight;
        if sum == previous {
            return sum;
        }
    }
}

/// Ertl's `tau` function, for the registers that hold the maximum rank.
fn tau(mut fraction: f64) -> f64 {
    if fraction == 0.0 || fraction == 1.0 {
        return 0.0;
    }
    let mut weight = 1.0;
    let mut sum = 1.0 - fraction;
    loop {
        fraction = fraction.sqrt();
        let previous = sum;
        weight *= 0.5;
        sum -= (1.0 - fraction).powi(2) * weight;
        if sum == previous {
            return sum / 3.0;
        }
    }
}

/// The 64-bit finalizer of MurmurHash3, a cheap bijective mix with full avalanche.
///
/// Fixed-width values are hashed with it rather than a byte hash, since it is a handful of
/// arithmetic instructions that the compiler vectorizes over a slice.
#[inline]
pub(super) fn mix64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^ (value >> 33)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;

    fn sketch_of(precision: u8, values: impl IntoIterator<Item = u64>) -> VortexResult<HllSketch> {
        let mut sketch = HllSketch::try_new(precision)?;
        for value in values {
            sketch.insert_hash(mix64(value));
        }
        Ok(sketch)
    }

    fn relative_error(estimate: u64, actual: u64) -> f64 {
        (estimate as f64 - actual as f64).abs() / actual as f64
    }

    #[test]
    fn empty_sketch_estimates_zero() -> VortexResult<()> {
        let sketch = HllSketch::try_new(12)?;
        assert!(sketch.is_empty());
        assert_eq!(sketch.estimate(), 0);
        Ok(())
    }

    #[rstest]
    #[case(3)]
    #[case(17)]
    fn rejects_unsupported_precision(#[case] precision: u8) {
        assert!(HllSketch::try_new(precision).is_err());
    }

    /// Small cardinalities are nearly exact: few registers collide.
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(10)]
    #[case(100)]
    fn small_cardinalities_are_close(#[case] distinct: u64) -> VortexResult<()> {
        let sketch = sketch_of(12, (0..distinct).cycle().take(10_000))?;
        let estimate = sketch.estimate();
        assert!(
            estimate.abs_diff(distinct) <= distinct.div_ceil(50),
            "estimate {estimate} for {distinct} distinct values"
        );
        Ok(())
    }

    /// Random data stays within four standard errors of the true count across the small, mid
    /// and large ranges where plain HyperLogLog needs bias corrections.
    #[rstest]
    #[case(10, 1_000)]
    #[case(12, 1_000)]
    #[case(12, 5_000)]
    #[case(12, 20_000)]
    #[case(12, 200_000)]
    #[case(14, 1_000_000)]
    fn random_data_within_expected_error(
        #[case] precision: u8,
        #[case] distinct: u64,
    ) -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(u64::from(precision) * distinct);
        let mut sketch = HllSketch::try_new(precision)?;
        let mut seen: HashSet<u64> = HashSet::default();
        while seen.len() < usize::try_from(distinct).unwrap_or(usize::MAX) {
            let value: u64 = rng.random();
            seen.insert(value);
            sketch.insert_hash(mix64(value));
            // Re-inserting a value never changes the sketch.
            sketch.insert_hash(mix64(value));
        }
        let standard_error = 1.04 / f64::from(1u32 << precision).sqrt();
        let error = relative_error(sketch.estimate(), distinct);
        assert!(
            error < 4.0 * standard_error,
            "relative error {error} exceeds {} for {distinct} values at precision {precision}",
            4.0 * standard_error
        );
        Ok(())
    }

    #[test]
    fn merge_estimates_union() -> VortexResult<()> {
        let mut left = sketch_of(12, 0..60_000)?;
        let right = sketch_of(12, 40_000..100_000)?;
        left.merge(&right)?;
        assert_eq!(left, sketch_of(12, 0..100_000)?);
        assert!(relative_error(left.estimate(), 100_000) < 0.05);
        Ok(())
    }

    #[test]
    fn merge_rejects_mismatched_precision() -> VortexResult<()> {
        let mut left = HllSketch::try_new(12)?;
        assert!(left.merge(&HllSketch::try_new(11)?).is_err());
        Ok(())
    }

    #[rstest]
    #[case::empty(0)]
    #[case::sparse(100)]
    #[case::dense(100_000)]
    fn serialization_roundtrips(#[case] distinct: u64) -> VortexResult<()> {
        let sketch = sketch_of(12, 0..distinct)?;
        let bytes = sketch.serialize();
        let expected_tag = if distinct < 1_000 {
            SPARSE_TAG
        } else {
            DENSE_TAG
        };
        assert_eq!(bytes[0], expected_tag);
        assert!(bytes.len() <= 1 + (1 << 12));
        assert_eq!(HllSketch::deserialize(12, &bytes)?, sketch);
        Ok(())
    }

    #[rstest]
    #[case::empty(&[])]
    #[case::unknown_tag(&[7])]
    #[case::dense_too_short(&[DENSE_TAG, 0, 0])]
    #[case::sparse_trailing(&[SPARSE_TAG, 0])]
    #[case::sparse_out_of_bounds(&[SPARSE_TAG, 0xff, 0xff, 1])]
    #[case::rank_too_large(&[SPARSE_TAG, 0, 0, 60])]
    fn deserialize_rejects_invalid(#[case] bytes: &[u8]) {
        assert!(HllSketch::deserialize(12, bytes).is_err());
    }
}
