// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bool compression statistics.

use std::iter;

use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

/// Array of booleans and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct BoolStats {
    /// Number of null values.
    null_count: u32,
    /// Number of non-null values.
    value_count: u32,
    /// Number of `true` values among valid (non-null) elements.
    true_count: u32,
    /// Number of maximal runs of equal values, where null is treated as its own value.
    run_count: u32,
}

impl BoolStats {
    /// Generates stats, returning an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if getting validity mask fails or values exceed `u32` bounds.
    pub fn generate(input: &BoolArray, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if input.is_empty() {
            return Ok(Self {
                null_count: 0,
                value_count: 0,
                true_count: 0,
                run_count: 0,
            });
        }

        if input.all_invalid(ctx)? {
            return Ok(Self {
                null_count: u32::try_from(input.len())?,
                value_count: 0,
                true_count: 0,
                run_count: 1,
            });
        }

        let validity = input
            .as_ref()
            .validity()?
            .execute_mask(input.as_ref().len(), ctx)?;
        let null_count = validity.false_count();
        let value_count = validity.true_count();

        let bits = input.to_bit_buffer();

        let (true_count, run_count) = match validity.bit_buffer() {
            AllOr::All => (
                bits.true_count(),
                count_runs(&bits, iter::repeat(u64::MAX)),
            ),
            AllOr::None => unreachable!("all-invalid handled above"),
            AllOr::Some(v) => {
                // Clear the bits under nulls so that only valid trues are counted, and so that
                // adjacent nulls compare equal when counting runs.
                let valid_bits = &bits & v;
                (valid_bits.true_count(), count_runs(&valid_bits, v.chunks().iter_padded()))
            }
        };

        Ok(Self {
            null_count: u32::try_from(null_count)?,
            value_count: u32::try_from(value_count)?,
            true_count: u32::try_from(true_count)?,
            run_count: u32::try_from(run_count)?,
        })
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> u32 {
        self.null_count
    }

    /// Returns the number of non-null values.
    pub fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Returns the number of `true` values among valid elements.
    pub fn true_count(&self) -> u32 {
        self.true_count
    }

    /// Returns the number of maximal runs of equal values, where null is treated as its own value.
    ///
    /// This is `0` for an empty array and `1` for a constant (or all-null) array.
    pub fn run_count(&self) -> u32 {
        self.run_count
    }

    /// Returns `true` if all valid values are the same (all-true or all-false).
    pub fn is_constant(&self) -> bool {
        self.value_count > 0 && (self.true_count == 0 || self.true_count == self.value_count)
    }
}

/// Counts maximal runs of equal values in `values`.
///
/// `validity_words` yields the validity as 64-bit little-endian words aligned with `values`. A
/// position's value is the pair of its validity bit and value bit, so callers must clear value bits
/// under nulls for adjacent nulls to compare equal.
fn count_runs(values: &BitBuffer, validity_words: impl Iterator<Item = u64>) -> usize {
    let len = values.len();
    if len == 0 {
        return 0;
    }

    let value_words = values.chunks().iter_padded();
    let num_words = len.div_ceil(64);
    let tail_bits = len - (num_words - 1) * 64;
    let tail_mask = if tail_bits == 64 {
        u64::MAX
    } else {
        (1u64 << tail_bits) - 1
    };

    // Bit `i` of a word's transitions is set when position `i` differs from position `i - 1`.
    // The carry holds the last bit of the previous word and starts as the first bit of the first
    // word, so position 0 never counts as a transition.
    let mut carry = None;
    let mut transitions = 0;
    for (idx, (value, valid)) in value_words.zip(validity_words).enumerate() {
        let (value_carry, valid_carry) = carry.unwrap_or((value & 1, valid & 1));
        let diff = (value ^ ((value << 1) | value_carry)) | (valid ^ ((valid << 1) | valid_carry));
        let mask = if idx + 1 == num_words {
            tail_mask
        } else {
            u64::MAX
        };
        transitions += (diff & mask).count_ones() as usize;
        carry = Some((value >> 63, valid >> 63));
    }

    transitions + 1
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::BoolStats;

    #[test]
    fn test_all_true() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![true, true, true]),
            Validity::NonNullable,
        );
        let stats = BoolStats::generate(&array, &mut ctx)?;
        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.true_count, 3);
        assert!(stats.is_constant());
        Ok(())
    }

    #[test]
    fn test_all_false() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![false, false, false]),
            Validity::NonNullable,
        );
        let stats = BoolStats::generate(&array, &mut ctx)?;
        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.true_count, 0);
        assert!(stats.is_constant());
        Ok(())
    }

    #[test]
    fn test_mixed() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![true, false, true]),
            Validity::NonNullable,
        );
        let stats = BoolStats::generate(&array, &mut ctx)?;
        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.true_count, 2);
        assert!(!stats.is_constant());
        Ok(())
    }

    #[test]
    fn test_with_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = BoolArray::new(
            BitBuffer::from(vec![true, false, true]),
            Validity::from_iter([true, false, true]),
        );
        let stats = BoolStats::generate(&array, &mut ctx)?;
        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.true_count, 2);
        assert!(stats.is_constant());
        Ok(())
    }

    /// Counts runs one element at a time, treating null as its own value.
    fn naive_run_count(values: &[Option<bool>]) -> u32 {
        let transitions = values.windows(2).filter(|w| w[0] != w[1]).count();
        u32::try_from(transitions + usize::from(!values.is_empty())).unwrap()
    }

    fn nullable_array(values: &[Option<bool>]) -> BoolArray {
        BoolArray::new(
            BitBuffer::from_iter(values.iter().map(|v| v.unwrap_or(false))),
            Validity::from_iter(values.iter().map(Option::is_some)),
        )
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::single(vec![true])]
    #[case::constant(vec![true; 200])]
    #[case::alternating((0..200).map(|i| i % 2 == 0).collect())]
    #[case::runs_across_words((0..300).map(|i| (i / 64) % 2 == 0).collect())]
    #[case::transition_at_word_boundary((0..128).map(|i| i >= 63).collect())]
    #[case::exact_word((0..64).map(|i| i % 7 == 0).collect())]
    #[case::sparse((0..1000).map(|i| i % 97 == 0).collect())]
    fn test_run_count_non_nullable(#[case] values: Vec<bool>) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let expected = naive_run_count(&values.iter().copied().map(Some).collect::<Vec<_>>());
        let array = BoolArray::new(BitBuffer::from(values), Validity::NonNullable);
        let stats = BoolStats::generate(&array, &mut ctx)?;
        assert_eq!(stats.run_count(), expected);
        Ok(())
    }

    #[rstest]
    #[case::all_null(vec![None; 100])]
    #[case::null_between_equal(vec![Some(true), None, Some(true)])]
    #[case::null_run_matches_false(vec![Some(false), None, None, Some(false)])]
    #[case::mixed((0..300)
        .map(|i| match i % 5 {
            0 => None,
            1 | 2 => Some(true),
            _ => Some(false),
        })
        .collect())]
    #[case::null_runs_across_words((0..300)
        .map(|i| ((i / 50) % 3 != 0).then_some(i % 100 < 50))
        .collect())]
    fn test_run_count_nullable(#[case] values: Vec<Option<bool>>) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let stats = BoolStats::generate(&nullable_array(&values), &mut ctx)?;
        assert_eq!(stats.run_count(), naive_run_count(&values));
        Ok(())
    }

    #[rstest]
    #[case(1, 64)]
    #[case(3, 200)]
    #[case(63, 130)]
    fn test_run_count_sliced(#[case] start: usize, #[case] end: usize) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values: Vec<Option<bool>> = (0..300)
            .map(|i| (i % 11 != 0).then_some((i / 5) % 2 == 0))
            .collect();
        let array = nullable_array(&values)
            .into_array()
            .slice(start..end)?
            .execute::<BoolArray>(&mut ctx)?;
        let stats = BoolStats::generate(&array, &mut ctx)?;
        assert_eq!(stats.run_count(), naive_run_count(&values[start..end]));
        Ok(())
    }
}
