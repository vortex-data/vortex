// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;

use super::DictConstraints;
use super::DictEncoder;
use super::codes_ptype_for_max_len;
use super::values::ValuesDict;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::converted_buffer;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::validity::Validity;

/// Creates a decimal encoder with a fixed code width based on the value domain and dictionary
/// length limit.
///
/// Dictionary values are stored as `T`, which [`super::dict_encoder_in`] chooses as the wider of
/// the canonical values type of `decimal_dtype` and the array's stored type, so chunks whose values
/// are stored narrower or wider share one dictionary. The
/// bound includes a possible null entry, so nullable values may require wider codes even when few
/// distinct values occur. [`super::dict_encode_with_constraints`] narrows the codes after encoding;
/// callers using this encoder directly retain the selected width.
pub fn decimal_dict_builder<T: NativeDecimalType + Hash>(
    decimal_dtype: DecimalDType,
    nullability: Nullability,
    constraints: &DictConstraints,
    allocator: BufferAllocatorRef,
) -> Box<dyn DictEncoder> {
    // Nullable values need an additional dictionary entry for null.
    let domain_len = match T::DECIMAL_TYPE {
        DecimalType::I8 => 1 << u8::BITS,
        DecimalType::I16 => 1 << u16::BITS,
        DecimalType::I32 => 1 << u32::BITS,
        DecimalType::I64 | DecimalType::I128 | DecimalType::I256 => u64::MAX,
    }
    .saturating_add(<u64 as From<bool>>::from(nullability.is_nullable()));
    match codes_ptype_for_max_len(domain_len.min(constraints.max_len as u64)) {
        PType::U8 => Box::new(DecimalDictBuilder::<T, u8>::new_in(
            decimal_dtype,
            nullability,
            constraints,
            allocator,
        )),
        PType::U16 => Box::new(DecimalDictBuilder::<T, u16>::new_in(
            decimal_dtype,
            nullability,
            constraints,
            allocator,
        )),
        PType::U32 => Box::new(DecimalDictBuilder::<T, u32>::new_in(
            decimal_dtype,
            nullability,
            constraints,
            allocator,
        )),
        _ => Box::new(DecimalDictBuilder::<T, u64>::new_in(
            decimal_dtype,
            nullability,
            constraints,
            allocator,
        )),
    }
}

/// Dictionary encode decimal arrays, storing values in the native decimal type `T`.
///
/// Null values are stored in the values of the dictionary such that codes are always non-null.
pub struct DecimalDictBuilder<T, Code> {
    dict: ValuesDict<T, T, Code>,
    decimal_dtype: DecimalDType,
    nullability: Nullability,
}

impl<T, Code> DecimalDictBuilder<T, Code>
where
    T: NativeDecimalType + Hash,
    Code: UnsignedPType,
{
    pub fn new_in(
        decimal_dtype: DecimalDType,
        nullability: Nullability,
        constraints: &DictConstraints,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let max_dict_len = constraints
            .max_len
            .min(constraints.max_bytes / T::DECIMAL_TYPE.byte_width());
        Self {
            dict: ValuesDict::new_in(max_dict_len, allocator),
            decimal_dtype,
            nullability,
        }
    }
}

impl<T, Code> DictEncoder for DecimalDictBuilder<T, Code>
where
    T: NativeDecimalType + Hash,
    Code: UnsignedPType,
{
    /// Assigns codes to the values, converted to `T`.
    ///
    /// `DecimalArray` does not check its values against the precision, so a chunk stored wider
    /// than `T` may hold a valid value that does not fit `T`. Such a chunk gets no codes, which
    /// ends this dictionary, and the next encoder is created at the chunk's width. An empty
    /// dictionary has no codes to end, so it fails instead.
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        let decimal = array.clone().execute::<DecimalArray>(ctx)?;
        let mask = decimal.validity()?.execute_mask(array.len(), ctx)?;
        let values = match converted_buffer::<T>(&decimal, &mask) {
            Ok(values) => values,
            Err(_) if !self.dict.is_empty() && decimal.values_type() > T::DECIMAL_TYPE => {
                return Ok(PrimitiveArray::new(
                    Buffer::<Code>::empty(),
                    Validity::NonNullable,
                ));
            }
            Err(err) => return Err(err),
        };
        let codes = self.dict.encode(values.as_slice(), &mask, |value| value);
        Ok(PrimitiveArray::new(codes, Validity::NonNullable))
    }

    fn reset(&mut self) -> ArrayRef {
        let (values, nulls) = self.dict.reset();
        DecimalArray::new(
            values,
            self.decimal_dtype,
            Validity::from_bit_buffer(nulls, self.nullability),
        )
        .into_array()
    }

    fn codes_ptype(&self) -> PType {
        Code::PTYPE
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::decimal_dict_builder;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::DictArray;
    use crate::arrays::decimal::narrowed_decimal;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::dict::DictConstraints;
    use crate::builders::dict::UNCONSTRAINED;
    use crate::builders::dict::dict_encode;
    use crate::builders::dict::dict_encoder_in;
    use crate::dtype::BigCast;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::match_each_decimal_value_type;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
    const USIZE_PTYPE: PType = PType::min_unsigned_ptype_for_value(usize::MAX as u64);

    fn decimal_of(values: &[i64], precision: u8, validity: Validity) -> DecimalArray {
        let decimal_dtype = DecimalDType::new(precision, 2);
        match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |T| {
            let buffer = values
                .iter()
                .map(|&v| <T as BigCast>::from(v).vortex_expect("test value fits"))
                .collect::<Buffer<T>>();
            DecimalArray::new(buffer, decimal_dtype, validity)
        })
    }

    #[rstest]
    fn encode_decimal(
        #[values(2, 4, 9, 18, 38, 76)] precision: u8,
        #[values(
            Validity::NonNullable,
            Validity::from_iter([true, false, true, true, false, true])
        )]
        validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = decimal_of(&[12, -7, 12, 0, 5, -7], precision, validity).into_array();
        let dict = dict_encode(&array, &mut ctx)?;

        assert_eq!(
            dict.codes().dtype(),
            &DType::Primitive(PType::U8, NonNullable)
        );
        assert!(dict.values().len() <= 4);
        assert_eq!(dict.values().dtype(), array.dtype());
        assert_arrays_eq!(dict, array, &mut ctx);
        Ok(())
    }

    /// Chunks of one column may store their values in different widths; they must share one
    /// dictionary and decode back to the same values.
    #[test]
    fn encode_chunks_with_different_values_types() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let wide = decimal_of(&[300, -300, 300], 38, Validity::NonNullable);
        let narrow = narrowed_decimal(decimal_of(&[300, 1, 1], 38, Validity::NonNullable));
        assert_eq!(narrow.values_type(), DecimalType::I16);
        let narrow = narrow.into_array();

        let constraints = DictConstraints {
            max_len: u16::MAX as usize,
            ..UNCONSTRAINED
        };
        let wide = wide.into_array();
        let mut encoder = dict_encoder_in(&wide, &constraints, ctx.allocator().clone());
        let wide_codes = encoder.encode(&wide, &mut ctx)?;
        let narrow_codes = encoder.encode(&narrow, &mut ctx)?;
        let values = encoder.reset();
        assert_eq!(values.len(), 3);

        let codes = ChunkedArray::from_iter([wide_codes.into_array(), narrow_codes.into_array()])
            .into_array();
        let decoded = DictArray::try_new(codes, values)?.into_array();
        let expected = ChunkedArray::from_iter([
            wide,
            decimal_of(&[300, 1, 1], 38, Validity::NonNullable).into_array(),
        ])
        .into_array();
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    /// Values that need the full width of `i128` and `i256` hash and decode exactly.
    #[test]
    fn encode_full_width_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let big = 10i128.pow(37);
        let i128_array = DecimalArray::new(
            Buffer::copy_from([big + 1, -big, big + 1, big + 2]),
            DecimalDType::new(38, 0),
            Validity::NonNullable,
        )
        .into_array();
        let i256_array = DecimalArray::new(
            Buffer::copy_from([
                i256::from_parts(1, 1 << 70),
                i256::from_parts(2, 1 << 70),
                i256::from_parts(1, 1 << 70),
                i256::from_parts(1, -(1 << 70)),
            ]),
            DecimalDType::new(76, 0),
            Validity::NonNullable,
        )
        .into_array();

        for array in [i128_array, i256_array] {
            let dict = dict_encode(&array, &mut ctx)?;
            assert_eq!(dict.values().len(), 3);
            assert_arrays_eq!(dict, array, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn reset_clears_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let first =
            decimal_of(&[1, 0, 3], 9, Validity::from_iter([true, false, true])).into_array();
        let constraints = DictConstraints {
            max_len: u32::MAX as usize,
            ..UNCONSTRAINED
        };
        let mut encoder = dict_encoder_in(&first, &constraints, ctx.allocator().clone());

        assert_arrays_eq!(
            encoder.encode(&first, &mut ctx)?,
            buffer![0u32, 1, 2].into_array(),
            &mut ctx
        );
        assert_arrays_eq!(encoder.reset(), first, &mut ctx);

        let second = decimal_of(&[4, 5], 9, Validity::AllValid).into_array();
        assert_arrays_eq!(
            encoder.encode(&second, &mut ctx)?,
            buffer![0u32, 1].into_array(),
            &mut ctx
        );
        assert_arrays_eq!(encoder.reset(), second, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::i8(2, NonNullable, usize::MAX, PType::U8)]
    #[case::nullable_i8(2, Nullable, usize::MAX, PType::U16)]
    #[case::i16(4, NonNullable, usize::MAX, PType::U16)]
    #[case::nullable_i16(4, Nullable, usize::MAX, PType::U32)]
    #[case::i32(9, NonNullable, usize::MAX, PType::U32)]
    #[case::nullable_i32(9, Nullable, usize::MAX, USIZE_PTYPE)]
    #[case::i64(18, NonNullable, usize::MAX, USIZE_PTYPE)]
    #[case::i128(38, Nullable, usize::MAX, USIZE_PTYPE)]
    #[case::i256(76, Nullable, usize::MAX, USIZE_PTYPE)]
    #[case::empty_budget(4, Nullable, 0, PType::U8)]
    #[case::full_u8_codes(4, Nullable, 256, PType::U8)]
    #[case::widen_u8_codes(4, Nullable, 257, PType::U16)]
    #[case::full_u16_codes(9, Nullable, 65_536, PType::U16)]
    #[case::widen_u16_codes(9, Nullable, 65_537, PType::U32)]
    fn codes_fit_dictionary_bounds(
        #[case] precision: u8,
        #[case] nullability: Nullability,
        #[case] max_len: usize,
        #[case] expected_codes_ptype: PType,
    ) {
        let ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(precision, 2);
        let constraints = DictConstraints {
            max_len,
            ..UNCONSTRAINED
        };
        let encoder = match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |T| {
            decimal_dict_builder::<T>(
                decimal_dtype,
                nullability,
                &constraints,
                ctx.allocator().clone(),
            )
        });
        assert_eq!(encoder.codes_ptype(), expected_codes_ptype);
    }

    /// `DecimalArray` does not check its values against the precision. An array that stores a
    /// value too wide for the precision's values type is encoded at its stored width.
    #[test]
    fn encode_values_wider_than_precision() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Precision 4 uses `i16`, which neither value fits.
        let array = DecimalArray::new(
            buffer![40_000i32, -70_000, 40_000],
            DecimalDType::new(4, 2),
            Validity::NonNullable,
        )
        .into_array();
        let dict = dict_encode(&array, &mut ctx)?;
        assert_eq!(dict.values().len(), 2);
        assert_arrays_eq!(dict, array, &mut ctx);
        Ok(())
    }

    /// A chunk that holds a value too wide for the dictionary gets no codes, which ends the
    /// dictionary, instead of failing.
    #[test]
    fn wider_chunk_ends_dictionary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(4, 2);
        let first = DecimalArray::new(buffer![12i16, -7], decimal_dtype, Validity::NonNullable)
            .into_array();
        let second =
            DecimalArray::new(buffer![12i32, 40_000], decimal_dtype, Validity::NonNullable)
                .into_array();
        let mut encoder = dict_encoder_in(&first, &UNCONSTRAINED, ctx.allocator().clone());

        assert_eq!(encoder.encode(&first, &mut ctx)?.len(), 2);
        assert_eq!(encoder.encode(&second, &mut ctx)?.len(), 0);
        assert_arrays_eq!(encoder.reset(), first, &mut ctx);

        // A new encoder for the chunk stores its values at the chunk's width.
        let mut encoder = dict_encoder_in(&second, &UNCONSTRAINED, ctx.allocator().clone());
        assert_eq!(encoder.encode(&second, &mut ctx)?.len(), 2);
        assert_arrays_eq!(encoder.reset(), second, &mut ctx);
        Ok(())
    }

    #[test]
    fn encode_stops_at_max_len() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = decimal_of(&[1, 2, 3, 1], 38, Validity::NonNullable).into_array();
        let constraints = DictConstraints {
            max_len: 2,
            ..UNCONSTRAINED
        };
        let mut encoder = dict_encoder_in(&array, &constraints, ctx.allocator().clone());
        let codes = encoder.encode(&array, &mut ctx)?;
        assert_eq!(codes.len(), 2);
        assert_eq!(encoder.reset().len(), 2);
        Ok(())
    }
}
