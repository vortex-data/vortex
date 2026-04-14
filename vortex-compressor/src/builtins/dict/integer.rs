// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use rustc_hash::FxBuildHasher;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_utils::aliases::hash_set::HashSet;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::builtins::is_integer_primitive;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::stats::IntegerStats;

impl Scheme for IntDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_integer_primitive(canonical)
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
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        let bit_width = data.array_as_primitive().ptype().bit_width();
        let stats = data.integer_stats();

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

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // TODO(connor): Fight the borrow checker (needs interior mutability)!
        let stats = data.integer_stats().clone();
        let dict = dictionary_encode(data.array_as_primitive(), &stats)?;

        // Values = child 0.
        let compressed_values = compressor.compress_child(dict.values(), &ctx, self.id(), 0)?;

        // Codes = child 1.
        let narrowed_codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(&mut compressor.execution_ctx())?
            .narrow()?
            .into_array();
        let compressed_codes = compressor.compress_child(&narrowed_codes, &ctx, self.id(), 1)?;

        // SAFETY: compressing codes does not change their values.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(dict.has_all_values_referenced())
                    .into_array(),
            )
        }
    }
}

/// Encodes a typed integer array into a [`DictArray`] by scanning for distinct values.
///
/// Because compression stats now estimate distinct counts without retaining the values
/// themselves, this macro rebuilds the exact set of distinct values directly from the array.
macro_rules! typed_encode {
    ($source_array:ident, $typ:ty) => {{
        let values_validity = match $source_array.validity()? {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };
        let codes_validity = $source_array.validity()?;

        let source = $source_array.as_slice::<$typ>();
        let validity_mask = $source_array.validity_mask();

        let mut seen: HashSet<NativeValue<$typ>, FxBuildHasher> =
            HashSet::with_hasher(FxBuildHasher);
        match validity_mask.bit_buffer() {
            AllOr::All => {
                for &v in source {
                    seen.insert(NativeValue(v));
                }
            }
            AllOr::None => {}
            AllOr::Some(mask) => {
                for (idx, &v) in source.iter().enumerate() {
                    if mask.value(idx) {
                        seen.insert(NativeValue(v));
                    }
                }
            }
        }

        let values: Buffer<$typ> = seen.iter().map(|x| x.0).collect();

        let max_code = values.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u8>>::encode(&values, source);
            PrimitiveArray::new(buf, codes_validity).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u16>>::encode(&values, source);
            PrimitiveArray::new(buf, codes_validity).into_array()
        } else {
            let buf = <DictEncoder as Encode<$typ, u32>>::encode(&values, source);
            PrimitiveArray::new(buf, codes_validity).into_array()
        };

        let values = PrimitiveArray::new(values, values_validity).into_array();
        // SAFETY: invariants enforced in DictEncoder.
        Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
    }};
}

/// Compresses an integer array into a dictionary array.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity from match on all integer types"
)]
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &IntegerStats,
) -> VortexResult<DictArray> {
    let _ = stats;
    match array.ptype() {
        vortex_array::dtype::PType::U8 => typed_encode!(array, u8),
        vortex_array::dtype::PType::U16 => typed_encode!(array, u16),
        vortex_array::dtype::PType::U32 => typed_encode!(array, u32),
        vortex_array::dtype::PType::U64 => typed_encode!(array, u64),
        vortex_array::dtype::PType::I8 => typed_encode!(array, i8),
        vortex_array::dtype::PType::I16 => typed_encode!(array, i16),
        vortex_array::dtype::PType::I32 => typed_encode!(array, i32),
        vortex_array::dtype::PType::I64 => typed_encode!(array, i64),
        other => vortex_error::vortex_bail!("unsupported integer ptype for dict encoding: {other}"),
    }
}

/// Stateless encoder that maps values to dictionary codes via a `HashMap`.
struct DictEncoder;

/// Trait for encoding values of type `T` into codes of type `I`.
trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

/// Implements [`Encode`] for an integer type with all code width variants (u8, u16, u32).
macro_rules! impl_encode {
    ($typ:ty) => { impl_encode!($typ, u8, u16, u32); };
    ($typ:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[expect(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_utils::aliases::hash_map::HashMap::<$typ, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value, code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value does not matter.
                    // SAFETY: we have exactly sized output to be as large as values.
                    unsafe { output.push_unchecked(codes.get(value).copied().unwrap_or_default()) };
                }

                output.freeze()
            }
        }
        )*
    };
}

impl_encode!(u8);
impl_encode!(u16);
impl_encode!(u32);
impl_encode!(u64);
impl_encode!(i8);
impl_encode!(i16);
impl_encode!(i32);
impl_encode!(i64);

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::dictionary_encode;
    use crate::stats::IntegerStats;

    #[test]
    fn test_dict_encode_integer_stats() {
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate_opts(
            &array,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let dict_array = dictionary_encode(array.as_view(), &stats).unwrap();
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![100i32, 200, 100, 100, 100],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        let undict = dict_array.as_array().to_primitive().into_array();
        assert_arrays_eq!(undict, expected);
    }
}
