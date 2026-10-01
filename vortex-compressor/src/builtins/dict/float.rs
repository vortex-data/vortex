// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

use vortex_array::ArrayId;
use vortex_array::ArrayInput;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::half::f16;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::aggregates;
use crate::aggregates::FloatDistinct;
use crate::aggregates::FloatDistinctPartial;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::DescendantExclusion;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;

/// Dictionary encoding for low-cardinality float values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatDictScheme;

impl Scheme for FloatDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn aggregate_requirements(&self) -> Vec<AggregateFnRef> {
        vec![
            FloatDistinct.bind(EmptyOptions),
            Count.bind(NumericalAggregateOpts::include_nans()),
        ]
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
        data: &ArrayInput,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let frequencies = aggregates::float_distinct(data, exec_ctx);
        let value_count = aggregates::valid_count(data, exec_ctx);

        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = frequencies.distinct_count();

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count > value_count / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Let sampling determine the expected ratio.
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayInput,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let frequencies = aggregates::float_distinct(data, exec_ctx);
        let dict = dictionary_encode(data.array().as_::<Primitive>(), &frequencies)?;

        let has_all_values_referenced = dict.has_all_values_referenced();

        // Values = child 0.
        let compressed_values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;

        // Codes = child 1.
        let narrowed_codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?
            .into_array();
        let compressed_codes =
            compressor.compress_child(&narrowed_codes, &compress_ctx, self.id(), 1, exec_ctx)?;

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

/// Encodes a typed float array into a [`DictArray`] using retained native distinct values.
macro_rules! typed_encode {
    ($source_array:ident, $typed:ident, $typ:ty) => {{
        let distinct = $typed.values();

        let values_validity = match $source_array.validity()? {
            Validity::NonNullable => Validity::NonNullable,
            _ => Validity::AllValid,
        };
        let codes_validity = $source_array.validity()?;

        let values: Buffer<$typ> = distinct.iter().map(|x| x.0).collect();

        let max_code = values.len();
        let codes = if max_code <= u8::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u8>>::encode(
                &values,
                $source_array.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, codes_validity).into_array()
        } else if max_code <= u16::MAX as usize {
            let buf = <DictEncoder as Encode<$typ, u16>>::encode(
                &values,
                $source_array.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, codes_validity).into_array()
        } else {
            let buf = <DictEncoder as Encode<$typ, u32>>::encode(
                &values,
                $source_array.as_slice::<$typ>(),
            );
            PrimitiveArray::new(buf, codes_validity).into_array()
        };

        let values = PrimitiveArray::new(values, values_validity).into_array();
        // SAFETY: enforced by the DictEncoder.
        Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
    }};
}

/// Compresses a floating-point array into a dictionary array using retained aggregate frequencies.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    frequencies: &FloatDistinctPartial,
) -> VortexResult<DictArray> {
    match frequencies {
        FloatDistinctPartial::F16(typed) => typed_encode!(array, typed, f16),
        FloatDistinctPartial::F32(typed) => typed_encode!(array, typed, f32),
        FloatDistinctPartial::F64(typed) => typed_encode!(array, typed, f64),
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
    use vortex_array::ArrayInput;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::dictionary_encode;
    use crate::aggregates;

    #[test]
    fn test_float_dict_encode() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let values = buffer![1f32, 2f32, 2f32, 0f32, 1f32];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(values, validity);

        let input = ArrayInput::new(array.clone().into_array());
        let frequencies = aggregates::float_distinct(&input, &mut ctx);
        let dict_array = dictionary_encode(array.as_view(), &frequencies)?;
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![1f32, 2f32, 2f32, 1f32, 1f32],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        let undict = dict_array
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(undict, expected, &mut ctx);
        Ok(())
    }
    #[test]
    fn dictionary_preserves_float_bits_and_excludes_null_payloads() -> VortexResult<()> {
        let bits = [
            0,
            0x8000_0000,
            0x7fc0_0001,
            0x7fc0_0002,
            0,
            0x7fc0_0001,
            0x7fc0_0003,
        ];
        let values = bits.map(f32::from_bits);
        let validity = Validity::Array(
            BoolArray::from_iter([true, true, true, true, true, true, false]).into_array(),
        );
        let array = PrimitiveArray::new(values.into_iter().collect::<Buffer<_>>(), validity);
        let input = ArrayInput::new(array.clone().into_array());
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let distinct = aggregates::float_distinct(&input, &mut ctx);
        let dict = dictionary_encode(array.as_view(), &distinct)?;
        let dict_values = dict.values().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let mut dict_bits: Vec<_> = dict_values
            .as_slice::<f32>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        dict_bits.sort_unstable();
        assert_eq!(dict_bits, [0, 0x7fc0_0001, 0x7fc0_0002, 0x8000_0000]);

        let decoded = dict.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_bits: Vec<_> = decoded.as_slice::<f32>()[..6]
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(decoded_bits, bits[..6]);
        Ok(())
    }
}
