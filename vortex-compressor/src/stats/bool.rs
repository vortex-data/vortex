// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bool compression statistics.

use vortex_array::arrays::BoolArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use super::GenerateStatsOptions;

/// Array of booleans and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct BoolStats {
    /// The underlying source array.
    src: BoolArray,
    /// Number of null values.
    null_count: u32,
    /// Number of `true` values among valid (non-null) elements.
    true_count: u32,
    /// Number of non-null values.
    value_count: u32,
}

impl BoolStats {
    /// Generates stats with default options.
    pub fn generate(input: &BoolArray) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    /// Generates stats with provided options.
    ///
    /// For booleans, all stats are cheap to compute so the options are currently ignored.
    pub fn generate_opts(input: &BoolArray, opts: GenerateStatsOptions) -> Self {
        Self::generate_opts_fallible(input, opts)
            .vortex_expect("BoolStats::generate_opts should not fail")
    }

    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &BoolArray,
        _opts: GenerateStatsOptions,
    ) -> VortexResult<Self> {
        if input.is_empty() {
            return Ok(Self {
                src: input.clone(),
                null_count: 0,
                value_count: 0,
                true_count: 0,
            });
        }

        if input.all_invalid()? {
            return Ok(Self {
                src: input.clone(),
                null_count: u32::try_from(input.len())?,
                value_count: 0,
                true_count: 0,
            });
        }

        let validity = input.validity_mask()?;
        let null_count = validity.false_count();
        let value_count = validity.true_count();

        let bits = input.to_bit_buffer();

        // Count how many true values exist among valid elements.
        let true_count = match validity.bit_buffer() {
            AllOr::All => bits.true_count(),
            AllOr::None => unreachable!("all-invalid handled above"),
            AllOr::Some(v) => {
                // AND the bits with validity to only count valid trues.
                (&bits & v).true_count()
            }
        };

        Ok(Self {
            src: input.clone(),
            null_count: u32::try_from(null_count)?,
            value_count: u32::try_from(value_count)?,
            true_count: u32::try_from(true_count)?,
        })
    }

    /// Returns the underlying source array.
    pub fn source(&self) -> &BoolArray {
        &self.src
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

    /// Returns `true` if all valid values are the same (all-true or all-false).
    pub fn is_constant(&self) -> bool {
        self.value_count > 0 && (self.true_count == 0 || self.true_count == self.value_count)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::BoolStats;

    #[test]
    fn test_all_true() -> VortexResult<()> {
        let array = BoolArray::new(
            BitBuffer::from(vec![true, true, true]),
            Validity::NonNullable,
        );
        let stats = BoolStats::generate(&array);
        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.true_count, 3);
        assert!(stats.is_constant());
        Ok(())
    }

    #[test]
    fn test_all_false() -> VortexResult<()> {
        let array = BoolArray::new(
            BitBuffer::from(vec![false, false, false]),
            Validity::NonNullable,
        );
        let stats = BoolStats::generate(&array);
        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.true_count, 0);
        assert!(stats.is_constant());
        Ok(())
    }

    #[test]
    fn test_mixed() -> VortexResult<()> {
        let array = BoolArray::new(
            BitBuffer::from(vec![true, false, true]),
            Validity::NonNullable,
        );
        let stats = BoolStats::generate(&array);
        assert_eq!(stats.value_count, 3);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.true_count, 2);
        assert!(!stats.is_constant());
        Ok(())
    }

    #[test]
    fn test_with_nulls() -> VortexResult<()> {
        let array = BoolArray::new(
            BitBuffer::from(vec![true, false, true]),
            Validity::from_iter([true, false, true]),
        );
        let stats = BoolStats::generate(&array);
        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.true_count, 2);
        assert!(stats.is_constant());
        Ok(())
    }
}
