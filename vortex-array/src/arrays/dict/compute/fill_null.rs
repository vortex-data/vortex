// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::scalar_fn::fns::operators::Operator;

impl FillNullKernel for Dict {
    fn fill_null(
        array: ArrayView<'_, Dict>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the fill value already exists in the dictionary, we can simply rewrite the null codes
        // to point to the value.
        let found_fill_values = array
            .values()
            .clone()
            .binary(
                ConstantArray::new(fill_value.clone(), array.values().len()).into_array(),
                Operator::Eq,
            )?
            .execute::<BoolArray>(ctx)?;

        // We found the fill value already in the values at this given index.
        let Some(existing_fill_value_index) =
            found_fill_values.bit_buffer_view().set_indices().next()
        else {
            // No fill values found, so we must canonicalize and fill_null.
            return Ok(Some(
                array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array()
                    .fill_null(fill_value.clone())?,
            ));
        };

        // A dictionary can hold more values than its codes type can index, so the index of an
        // unreferenced fill value can exceed the range of the codes. Non-nullable codes need no
        // fill code. For nullable codes, widen the codes to the smallest unsigned type that can
        // hold the fill index in that case. Codes are never negative, so the cast keeps them.
        let codes = array.codes();
        let codes = if codes.dtype().is_nullable() {
            let fill_index = existing_fill_value_index as u64;
            let codes = if fill_index > codes.dtype().as_ptype().max_value_as_u64() {
                let wide_ptype = PType::min_unsigned_ptype_for_value(fill_index);
                codes
                    .clone()
                    .cast(DType::Primitive(wide_ptype, Nullability::Nullable))?
            } else {
                codes.clone()
            };

            let codes_dtype = codes.dtype().as_nonnullable();
            codes.fill_null(Scalar::from(fill_index).cast(&codes_dtype)?)?
        } else {
            codes.clone()
        };

        // Fill the nulls of the values. Note that the precondition of this function states that the
        // fill value is non-null, so we do not have to worry about the nullability.
        let values = array.values().clone().fill_null(fill_value.clone())?;

        // SAFETY: invariants are still satisfied after patching nulls.
        unsafe {
            Ok(Some(
                DictArray::new_unchecked(codes, values)
                    .set_all_values_referenced(array.has_all_values_referenced())
                    .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::fill_null::FillNullKernel;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::u8_max(PrimitiveArray::from_option_iter([Some(0u8), None]), 255)]
    #[case::u8_overflow(PrimitiveArray::from_option_iter([Some(0u8), None]), 256)]
    #[case::u8_truncation(PrimitiveArray::from_option_iter([Some(0u8), None]), 300)]
    #[case::i8_max(PrimitiveArray::from_option_iter([Some(0i8), None]), 127)]
    #[case::i8_overflow(PrimitiveArray::from_option_iter([Some(0i8), None]), 128)]
    fn fill_unreferenced_value(
        #[case] codes: PrimitiveArray,
        #[case] fill_value: i32,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            codes.into_array(),
            PrimitiveArray::from_iter(0..=fill_value).into_array(),
        )?;

        let filled = dict.into_array().fill_null(fill_value)?;
        assert_arrays_eq!(filled, PrimitiveArray::from_iter([0, fill_value]), &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::u8_to_u16(PrimitiveArray::from_option_iter([Some(0u8), None]), 256, PType::U16)]
    #[case::u8_to_u16_truncation(PrimitiveArray::from_option_iter([Some(0u8), None]), 300, PType::U16)]
    #[case::i8_to_u8(PrimitiveArray::from_option_iter([Some(0i8), None]), 128, PType::U8)]
    #[case::i8_to_u16(PrimitiveArray::from_option_iter([Some(0i8), None]), 256, PType::U16)]
    #[case::u16_to_u32(PrimitiveArray::from_option_iter([Some(0u16), None]), 65_536, PType::U32)]
    fn fill_unreferenced_value_widens_codes(
        #[case] codes: PrimitiveArray,
        #[case] fill_value: i32,
        #[case] wide_ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            codes.into_array(),
            PrimitiveArray::from_iter(0..=fill_value).into_array(),
        )?;

        let filled = <Dict as FillNullKernel>::fill_null(
            dict.as_view(),
            &Scalar::from(fill_value),
            &mut ctx,
        )?
        .vortex_expect("kernel should fill the dictionary codes");
        let filled_dict = filled.as_::<Dict>();
        assert_eq!(filled_dict.codes().dtype().as_ptype(), wide_ptype);
        assert!(!filled_dict.codes().dtype().is_nullable());
        assert_arrays_eq!(filled, PrimitiveArray::from_iter([0, fill_value]), &mut ctx);

        Ok(())
    }

    #[test]
    fn non_nullable_codes_fill_unreferenced_value() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter((0..=300).map(|value| (value != 1).then_some(value)));
        let dict = DictArray::try_new(buffer![0u8, 1].into_array(), values.into_array())?;

        let filled = dict.into_array().fill_null(300)?;
        assert_arrays_eq!(filled, PrimitiveArray::from_iter([0, 300]), &mut ctx);

        Ok(())
    }

    #[test]
    fn nullable_codes_fill_in_values() {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2],
                Validity::from(BitBuffer::from(vec![true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(buffer![10, 20, 20], Validity::AllValid).into_array(),
        )
        .vortex_expect("operation should succeed in test");

        let filled = dict
            .into_array()
            .fill_null(Scalar::primitive(20, Nullability::NonNullable))
            .vortex_expect("operation should succeed in test");
        assert_arrays_eq!(filled, PrimitiveArray::from_iter([10, 20, 20]), &mut ctx);
        assert!(filled.all_valid(&mut ctx).unwrap());
    }
}
