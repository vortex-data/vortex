// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

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
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_utils::aliases::hash_set::HashSet;

use crate::CascadingCompressor;
use crate::builtins::FloatDictScheme;
use crate::builtins::IntDictScheme;
use crate::builtins::is_float_primitive;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateVerdict;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::FloatStats;
use crate::stats::GenerateStatsOptions;

impl Scheme for FloatDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_float_primitive(canonical)
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

    /// Float dict codes (child 1) are compact unsigned integers that should not be
    /// dict-encoded again. Float dict values (child 0) flow through ALP into integer-land,
    /// where integer dict encoding is redundant since the values are already deduplicated at
    /// the float level.
    ///
    /// Additional exclusions for codes (IntSequenceScheme, IntRunEndScheme, FoRScheme,
    /// ZigZagScheme, SparseScheme, RLE) are expressed as pull rules on those schemes in
    /// vortex-btrblocks.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(0),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        let stats = data.float_stats();

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

        // Let sampling determine the expected ratio.
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // TODO(connor): Fight the borrow checker (needs interior mutability)!
        let stats = data.float_stats().clone();
        let dict = dictionary_encode(data.array_as_primitive(), &stats)?;

        let has_all_values_referenced = dict.has_all_values_referenced();

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

        // SAFETY: compressing codes or values does not alter the invariants.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(has_all_values_referenced)
                    .into_array(),
            )
        }
    }
}

/// Encodes a typed float array into a [`DictArray`] by scanning for distinct values.
///
/// Because compression stats now estimate the distinct count without retaining the values
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
        // SAFETY: enforced by the DictEncoder.
        Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
    }};
}

/// Compresses a floating-point array into a dictionary array.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &FloatStats,
) -> VortexResult<DictArray> {
    let _ = stats;
    match array.ptype() {
        PType::F16 => typed_encode!(array, f16),
        PType::F32 => typed_encode!(array, f32),
        PType::F64 => typed_encode!(array, f64),
        other => vortex_bail!("unsupported float ptype for dict encoding: {other}"),
    }
}

/// Stateless encoder that maps values to dictionary codes via a `HashMap`.
struct DictEncoder;

/// Trait for encoding values of type `T` into codes of type `I`.
trait Encode<T, I> {
    /// Using the distinct value set, turn the values into a set of codes.
    fn encode(distinct: &[T], values: &[T]) -> Buffer<I>;
}

/// Implements [`Encode`] for a float type using its bit representation as the hash key.
macro_rules! impl_encode {
    ($typ:ty, $utyp:ty) => { impl_encode!($typ, $utyp, u8, u16, u32); };
    ($typ:ty, $utyp:ty, $($ityp:ty),+) => {
        $(
        impl Encode<$typ, $ityp> for DictEncoder {
            #[expect(clippy::cast_possible_truncation)]
            fn encode(distinct: &[$typ], values: &[$typ]) -> Buffer<$ityp> {
                let mut codes =
                    vortex_utils::aliases::hash_map::HashMap::<$utyp, $ityp>::with_capacity(
                        distinct.len(),
                    );
                for (code, &value) in distinct.iter().enumerate() {
                    codes.insert(value.to_bits(), code as $ityp);
                }

                let mut output = vortex_buffer::BufferMut::with_capacity(values.len());
                for value in values {
                    // Any code lookups which fail are for nulls, so their value does not matter.
                    output.push(codes.get(&value.to_bits()).copied().unwrap_or_default());
                }

                output.freeze()
            }
        }
        )*
    };
}

impl_encode!(f16, u16);
impl_encode!(f32, u32);
impl_encode!(f64, u64);

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
    use crate::stats::FloatStats;
    use crate::stats::GenerateStatsOptions;

    #[test]
    fn test_float_dict_encode() {
        let values = buffer![1f32, 2f32, 2f32, 0f32, 1f32];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(values, validity);

        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let dict_array = dictionary_encode(array.as_view(), &stats).unwrap();
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![1f32, 2f32, 2f32, 1f32, 1f32],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        let undict = dict_array.as_array().to_primitive().into_array();
        assert_arrays_eq!(undict, expected);
    }
}
