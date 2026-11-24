// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use std::hash::Hash;

use vortex_array::ArrayId;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::IntegerPType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::dictionary_encoded;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::Encoded;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::stats::IntegerErasedStats;
use crate::stats::IntegerStats;
use crate::stats::IntegerTypedStats;

/// Dictionary encoding for low-cardinality integer values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntDictScheme;

impl Scheme for IntDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
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

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let bit_width = data.array_as_primitive().ptype().bit_width();
        let stats = data.integer_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count > stats.value_count() / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Ignore nulls encoding for the estimate. We only focus on values.

        let values_size = bit_width * distinct_values_count as usize;

        // TODO(connor): Should we just hardcode this instead of let the compressor choose?
        // Assume codes are compressed RLE + BitPacking.
        let codes_bw = u32::BITS - distinct_values_count.leading_zeros();

        let n_runs = (stats.value_count() / stats.average_run_length()) as usize;

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
        let stats = data.integer_stats(exec_ctx);
        let dict = dictionary_encode(data.array_as_primitive(), &stats)?;
        Ok(dictionary_encoded(self, &dict))
    }
}

/// Compresses an integer array into a dictionary array according to attached stats.
///
/// Distinct values are ordered by descending occurrence count (breaking ties deterministically
/// in ascending value order) so hot values receive the smallest codes.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &IntegerStats,
) -> VortexResult<DictArray> {
    match stats.erased() {
        IntegerErasedStats::U8(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::U16(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::U32(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::U64(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I8(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I16(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I32(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I64(typed) => dictionary_encode_typed(array, typed),
    }
}

/// Encodes a typed integer array into a [`DictArray`] ordered by descending frequency, breaking
/// ties by ascending value order.
fn dictionary_encode_typed<T: IntegerPType>(
    source_array: ArrayView<'_, Primitive>,
    typed: &IntegerTypedStats<T>,
) -> VortexResult<DictArray>
where
    NativeValue<T>: Eq + Hash,
{
    let distinct = typed.distinct().vortex_expect(
        "this must be present since `DictScheme` declared that we need distinct values",
    );
    super::dictionary_encode_from_counts(
        source_array,
        distinct
            .distinct_values()
            .iter()
            .map(|(value, &count)| (value.0, count)),
    )
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::DictArrayExt;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::dictionary_encode;
    use crate::stats::GenerateStatsOptions;
    use crate::stats::IntegerStats;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_dict_encode_integer_stats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let dict_array = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);
        assert_eq!(
            dict_array.values().as_::<Primitive>().as_slice::<i32>(),
            &[100, 200]
        );

        let expected = PrimitiveArray::new(
            buffer![100i32, 200, 100, 100, 100],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        assert_arrays_eq!(dict_array, expected, &mut ctx);
        Ok(())
    }

    /// Codes take the narrowest type that can address the dictionary, including when the
    /// dictionary exactly fills a type's range.
    #[rstest]
    #[case::fills_u8(256, PType::U8)]
    #[case::overflows_u8(257, PType::U16)]
    fn codes_are_as_narrow_as_the_dictionary_allows(
        #[case] distinct: i32,
        #[case] expected: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter(0..distinct);
        let stats = IntegerStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        let dict = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(PType::try_from(dict.codes().dtype())?, expected);
        Ok(())
    }

    #[test]
    fn test_frequency_ordered_codes_and_tie_break_determinism() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut data = Vec::with_capacity(78);
        data.push(99i64);
        data.push(-50);
        data.extend(iter::repeat_n(-1, 5));
        data.extend(iter::repeat_n(25, 20));
        data.extend(iter::repeat_n(10, 50));
        data.push(50);

        let opts = GenerateStatsOptions {
            count_distinct_values: true,
        };
        let array_a = PrimitiveArray::from_iter(data.iter().copied());
        let array_b = PrimitiveArray::from_iter(data.iter().rev().copied());
        let stats_a = IntegerStats::generate_opts(&array_a, opts, &mut ctx);
        let stats_b = IntegerStats::generate_opts(&array_b, opts, &mut ctx);

        let dict_a = dictionary_encode(array_a.as_view(), &stats_a)?;
        let dict_b = dictionary_encode(array_b.as_view(), &stats_b)?;
        assert!(dict_a.has_all_values_referenced());
        assert_eq!(
            dict_a.values().as_::<Primitive>().as_slice::<i64>(),
            &[10, 25, -1, -50, 50, 99]
        );
        assert_eq!(
            dict_a.values().as_::<Primitive>().as_slice::<i64>(),
            dict_b.values().as_::<Primitive>().as_slice::<i64>()
        );
        assert_arrays_eq!(dict_a, array_a, &mut ctx);
        assert_arrays_eq!(dict_b, array_b, &mut ctx);
        Ok(())
    }
}
