// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

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
use crate::dtype::DType;
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
/// Dictionary values are stored as `T`, the canonical values type of `decimal_dtype`. The bound
/// includes a possible null entry, so nullable values may require wider codes even when few
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
        debug_assert_eq!(T::DECIMAL_TYPE, decimal_dtype.canonical_values_type());

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
    /// Assigns codes to the unscaled values.
    ///
    /// The input must have the dtype of the encoder, because the unscaled values only have a
    /// meaning together with their precision and scale.
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        vortex_ensure_eq!(
            array.dtype(),
            &DType::Decimal(self.decimal_dtype, self.nullability)
        );

        let decimal = array.clone().execute::<DecimalArray>(ctx)?;
        let mask = decimal.validity()?.execute_mask(array.len(), ctx)?;
        let codes = self
            .dict
            .encode(decimal.buffer::<T>().as_slice(), &mask, |value| value);

        // SAFETY: a non-nullable validity carries no length to match against the codes.
        Ok(unsafe { PrimitiveArray::new_unchecked(codes.freeze(), Validity::NonNullable) })
    }

    fn reset(&mut self) -> ArrayRef {
        let (values, nulls) = self.dict.reset();
        let validity = Validity::from_bit_buffer(nulls, self.nullability);

        // SAFETY: every dictionary value is pushed alongside its validity bit, so an array-backed
        // validity covers exactly the values. `encode` admits only input of this decimal dtype,
        // so every valid value is within the precision.
        unsafe { DecimalArray::new_unchecked(values, self.decimal_dtype, validity) }.into_array()
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
    use crate::arrays::WidenArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::dict::DictConstraints;
    use crate::builders::dict::UNCONSTRAINED;
    use crate::builders::dict::dict_encode;
    use crate::builders::dict::dict_encoder_in;
    use crate::dtype::BigCast;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
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
            DecimalArray::new(
                buffer,
                decimal_dtype,
                validity,
                &mut SESSION.create_execution_ctx(),
            )
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

    /// A narrow physical representation is a `Widen` over a narrower decimal child. It shares the
    /// dictionary of canonical chunks and decodes back to the same values.
    #[test]
    fn encode_widen_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let wide = decimal_of(&[300, -300, 300], 38, Validity::NonNullable).into_array();
        let narrow_values = decimal_of(&[300, 1, 1], 4, Validity::NonNullable).into_array();
        let narrow = WidenArray::try_new(narrow_values, wide.dtype().clone())?.into_array();

        let constraints = DictConstraints {
            max_len: u16::MAX as usize,
            ..UNCONSTRAINED
        };
        let mut encoder = dict_encoder_in(&wide, &constraints, ctx.allocator().clone());
        let wide_codes = encoder.encode(&wide, &mut ctx)?;
        let narrow_codes = encoder.encode(&narrow, &mut ctx)?;
        let values = encoder.reset();
        assert_eq!(values.len(), 3);

        let codes = ChunkedArray::from_iter([wide_codes.into_array(), narrow_codes.into_array()])
            .into_array();
        let decoded = DictArray::try_new(codes, values)?.into_array();
        let expected = ChunkedArray::from_iter([wide, narrow]).into_array();
        assert_arrays_eq!(decoded, expected, &mut ctx);

        Ok(())
    }

    /// Unscaled values of another decimal dtype would change their meaning in this dictionary, so
    /// the encoder rejects them and leaves the dictionary as it was.
    #[rstest]
    #[case::scale(DecimalDType::new(4, 0), NonNullable)]
    #[case::precision(DecimalDType::new(3, 2), NonNullable)]
    #[case::nullability(DecimalDType::new(4, 2), Nullable)]
    fn encode_rejects_other_dtype(
        #[case] other_dtype: DecimalDType,
        #[case] other_nullability: Nullability,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let first = DecimalArray::new(
            buffer![12i16, -7],
            DecimalDType::new(4, 2),
            Validity::NonNullable,
            &mut ctx,
        )
        .into_array();
        let other = DecimalArray::new(
            buffer![12i16, 99],
            other_dtype,
            Validity::from(other_nullability),
            &mut ctx,
        )
        .into_array();
        let mut encoder = dict_encoder_in(&first, &UNCONSTRAINED, ctx.allocator().clone());

        assert_eq!(encoder.encode(&first, &mut ctx)?.len(), 2);
        assert!(encoder.encode(&other, &mut ctx).is_err());
        assert_arrays_eq!(encoder.reset(), first, &mut ctx);

        Ok(())
    }

    /// Empty input adds no entries, not even a null entry for an all-null mask.
    #[rstest]
    fn encode_empty_adds_no_entries(
        #[values(Validity::AllValid, Validity::AllInvalid)] validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = decimal_of(&[], 9, validity).into_array();

        let dict = dict_encode(&array, &mut ctx)?;
        assert!(dict.values().is_empty());
        assert!(dict.codes().is_empty());

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
            &mut ctx,
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
            &mut ctx,
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
