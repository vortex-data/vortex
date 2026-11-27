// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::decimal::DecimalArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::match_each_decimal_value_type;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use super::dictionary_encoded;
use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DescendantExclusion;
use crate::scheme::Encoded;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Dictionary encoding for low-cardinality decimal values.
///
/// The dictionary holds each distinct decimal once, so a wide decimal that the decimal scheme
/// would split into several parts gets one codes array instead of one per part.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalDictScheme;

impl Scheme for DecimalDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Decimal dict codes (child 1) are compact unsigned integers that should not be
    /// dict-encoded again.
    ///
    /// The values (child 0) keep integer dictionaries: distinct decimals that byte parts split
    /// into several parts can share a most significant part.
    ///
    /// Pull rules in `vortex-btrblocks` also keep `SequenceScheme`, `FoRScheme`, `ZigZagScheme` and
    /// `DeltaScheme` off the codes (child 1), and `RunEndScheme` and the RLE schemes off the values
    /// (child 0).
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(1),
        }]
    }

    /// Estimates the ratio from the exact distinct count, like `IntDictScheme`.
    ///
    /// A sample would under-rate a dictionary of thousands of distinct values: the sample holds
    /// almost as many distinct values as rows, so its dictionary is no smaller than its input.
    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.decimal_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DecimalDictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count > stats.value_count() / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Ignore nulls encoding for the estimate. We only focus on values, at their stored width,
        // since the values are compressed again.
        let bit_width = data.array_as_decimal().values_type().byte_width() * 8;
        let values_size = bit_width * distinct_values_count as usize;

        // Assume codes are compressed RLE + BitPacking.
        let codes_bw = u32::BITS - distinct_values_count.leading_zeros();
        let average_run_length = stats
            .average_run_length()
            .vortex_expect("runs are counted with the distinct values");
        let n_runs = (stats.value_count() / average_run_length) as usize;

        // Assume that codes will either be BitPack or RLE-BitPack.
        let codes_size_bp = codes_bw as usize * stats.value_count() as usize;
        let codes_size_rle_bp = usize::checked_mul(codes_bw as usize + 32, n_runs);

        let codes_size = usize::min(codes_size_bp, codes_size_rle_bp.unwrap_or(usize::MAX));

        let before = stats.value_count() as usize * bit_width;

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(
            before as f64 / (values_size + codes_size) as f64,
        ))
    }

    /// Children: values=0, codes=1. The codes carry the validity.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let dict = dictionary_encode(data.array_as_decimal(), exec_ctx)?;
        Ok(dictionary_encoded(self, &dict))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// Encodes a decimal array into a [`DictArray`] ordered by descending occurrence count, breaking
/// ties by ascending value order.
fn dictionary_encode(
    array: ArrayView<'_, Decimal>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    match_each_decimal_value_type!(array.values_type(), |T| {
        dictionary_encode_typed::<T>(array, ctx)
    })
}

/// Counts valid decimal values and builds a frequency-ordered [`DictArray`] for storage type `T`.
fn dictionary_encode_typed<T: NativeDecimalType + Hash>(
    array: ArrayView<'_, Decimal>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    let codes_validity = array.validity()?;
    let values_validity = match codes_validity {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let slice = array.buffer::<T>();
    let mask = codes_validity.execute_mask(array.len(), ctx)?;
    let mut counts = HashMap::<T, u32, FxBuildHasher>::with_hasher(FxBuildHasher);
    match mask {
        Mask::AllTrue(_) => {
            for &value in slice.as_slice() {
                *counts.entry(value).or_insert(0) += 1;
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(values) => {
            for idx in values.bit_buffer().set_indices() {
                *counts.entry(slice[idx]).or_insert(0) += 1;
            }
        }
    }

    let (values, codes) = super::encode_sorted_codes(
        slice.as_slice(),
        codes_validity,
        counts.into_iter(),
        |v| v,
        |a, b| a.cmp(&b),
    );
    let values = DecimalArray::try_new_handle_from_trusted_values(
        BufferHandle::new_host(values.into_byte_buffer()),
        T::DECIMAL_TYPE,
        array.decimal_dtype(),
        values_validity,
    )?
    .into_array();
    // SAFETY: `encode_sorted_codes` emits codes in `0..values.len()` for every valid entry
    // and `counts` holds the exact distinct valid value set of `array`.
    Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DecimalDictScheme;
    use crate::CascadingCompressor;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn decimal(values: impl IntoIterator<Item = i128>, validity: Validity) -> DecimalArray {
        DecimalArray::new(
            values.into_iter().collect::<Buffer<i128>>(),
            DecimalDType::new(38, 2),
            validity,
            &mut SESSION.create_execution_ctx(),
        )
    }

    #[rstest]
    #[case::non_nullable(Validity::NonNullable)]
    #[case::nullable(Validity::from_iter((0..1024).map(|i| i % 5 != 0)))]
    fn compresses_low_cardinality_decimals_to_dict(#[case] validity: Validity) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let distinct = [1i128 << 100, -(1i128 << 90), 7];
        let array = decimal((0..1024).map(|i| distinct[i % 3]), validity).into_array();

        let compressed =
            CascadingCompressor::new(vec![&DecimalDictScheme]).compress(&array, &mut ctx)?;

        assert!(compressed.is::<Dict>(), "{}", compressed.display_tree());
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn orders_decimal_dictionary_codes_by_frequency() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut raw = Vec::with_capacity(76);
        raw.push(9999i128);
        raw.extend(iter::repeat_n(-500i128, 5));
        raw.extend(iter::repeat_n(250i128, 20));
        raw.extend(iter::repeat_n(1000i128, 50));
        let array = decimal(raw, Validity::NonNullable).into_array();

        let compressed =
            CascadingCompressor::new(vec![&DecimalDictScheme]).compress(&array, &mut ctx)?;
        let dict = compressed.as_opt::<Dict>().expect("expected Dict");
        assert_arrays_eq!(
            dict.values(),
            decimal([1000, 250, -500, 9999], Validity::NonNullable),
            &mut ctx
        );
        let codes = dict.codes().as_::<Primitive>();
        let slice = codes.as_slice::<u8>();
        assert_eq!(slice[0], 3);
        assert!(slice[1..6].iter().all(|&c| c == 2));
        assert!(slice[6..26].iter().all(|&c| c == 1));
        assert!(slice[26..76].iter().all(|&c| c == 0));
        assert_arrays_eq!(compressed, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn skips_high_cardinality_decimals() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = decimal((0..1024).map(|i| i * 17), Validity::NonNullable).into_array();

        let compressed =
            CascadingCompressor::new(vec![&DecimalDictScheme]).compress(&array, &mut ctx)?;

        assert!(!compressed.is::<Dict>(), "{}", compressed.display_tree());
        Ok(())
    }
}
