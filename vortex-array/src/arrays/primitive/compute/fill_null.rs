// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::validity::Validity;

impl FillNullKernel for Primitive {
    fn fill_null(
        array: ArrayView<'_, Primitive>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let result_validity = Validity::from(fill_value.dtype().nullability());

        Ok(Some(match array.validity()? {
            Validity::Array(is_valid) => {
                let is_invalid = is_valid.execute::<BoolArray>(ctx)?.into_bit_buffer().not();
                match_each_native_ptype!(array.ptype(), |T| {
                    let mut buffer = array.to_buffer::<T>().into_mut();
                    let fill_value = fill_value
                        .as_primitive()
                        .typed_value::<T>()
                        .vortex_expect("top-level fill_null ensure non-null fill value");
                    for invalid_index in is_invalid.set_indices() {
                        buffer[invalid_index] = fill_value;
                    }
                    PrimitiveArray::new(buffer.freeze(), result_validity).into_array()
                })
            }
            _ => unreachable!("checked in entry point"),
        }))
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::compute::fill_null::BoolArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn fill_null_leading_none() {
        let arr = PrimitiveArray::from_option_iter([None, Some(8u8), None, Some(10), None]);
        let p = arr
            .into_array()
            .fill_null(Scalar::from(42u8))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([42u8, 8, 42, 10, 42]));
        assert!(
            p.as_ref()
                .validity()
                .unwrap()
                .to_mask(p.as_ref().len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }

    #[test]
    fn fill_null_all_none() {
        let arr = PrimitiveArray::from_option_iter([Option::<u8>::None, None, None, None, None]);

        let p = arr
            .into_array()
            .fill_null(Scalar::from(255u8))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([255u8, 255, 255, 255, 255]));
        assert!(
            p.as_ref()
                .validity()
                .unwrap()
                .to_mask(p.as_ref().len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }

    #[test]
    fn fill_null_nullable_non_null() {
        let arr = PrimitiveArray::new(
            buffer![8u8, 10, 12, 14, 16],
            Validity::Array(BoolArray::from_iter([true, true, true, true, true]).into_array()),
        );
        let p = arr
            .into_array()
            .fill_null(Scalar::from(255u8))
            .unwrap()
            .to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([8u8, 10, 12, 14, 16]));
        assert!(
            p.as_ref()
                .validity()
                .unwrap()
                .to_mask(p.as_ref().len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }

    #[test]
    fn fill_null_non_nullable() {
        let arr = buffer![8u8, 10, 12, 14, 16].into_array();
        let p = arr.fill_null(Scalar::from(255u8)).unwrap().to_primitive();
        assert_arrays_eq!(p, PrimitiveArray::from_iter([8u8, 10, 12, 14, 16]));
        assert!(
            p.as_ref()
                .validity()
                .unwrap()
                .to_mask(p.as_ref().len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }
}
