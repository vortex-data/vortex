// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::bitpacking::BitPacked;
use crate::bitpacking::array::BitPackedArrayExt;

fn build_with_validity(
    array: ArrayView<'_, BitPacked>,
    dtype: &DType,
    new_validity: Validity,
) -> VortexResult<ArrayRef> {
    Ok(BitPacked::try_new(
        array.packed().clone(),
        dtype.as_ptype(),
        new_validity,
        array
            .patches()
            .map(|patches| patches.map_values(|values| values.cast(dtype.clone())))
            .transpose()?,
        array.bit_width(),
        array.len(),
        array.offset(),
    )?
    .into_array())
}

impl CastReduce for BitPacked {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }
        let Some(new_validity) = array
            .validity()?
            .try_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };
        Ok(Some(build_with_validity(array, dtype, new_validity)?))
    }
}

impl CastKernel for BitPacked {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            return Ok(None);
        }
        let new_validity =
            array
                .validity()?
                .cast_nullability(dtype.nullability(), array.len(), ctx)?;
        Ok(Some(build_with_validity(array, dtype, new_validity)?))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::BitPackedArray;
    use crate::BitPackedData;

    fn bp(array: &ArrayRef, bit_width: u8) -> BitPackedArray {
        BitPackedData::encode(array, bit_width, &mut LEGACY_SESSION.create_execution_ctx()).unwrap()
    }

    #[test]
    fn test_cast_bitpacked_u8_to_u32() {
        let packed = bp(&buffer![10u8, 20, 30, 40, 50, 60].into_array(), 6);

        let casted = packed
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([10u32, 20, 30, 40, 50, 60])
        );
    }

    #[test]
    fn test_cast_bitpacked_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(5u16), None, Some(10), Some(15), None]);
        let packed = bp(&values.into_array(), 4);

        let casted = packed
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(bp(&buffer![0u8, 10, 20, 30, 40, 50, 60, 63].into_array(), 6))]
    #[case(bp(&buffer![0u16, 100, 200, 300, 400, 500].into_array(), 9))]
    #[case(bp(&buffer![0u32, 1000, 2000, 3000, 4000].into_array(), 12))]
    #[case(bp(&PrimitiveArray::from_option_iter([Some(1u32), None, Some(7), Some(15), None]).into_array(), 4))]
    fn test_cast_bitpacked_conformance(#[case] array: BitPackedArray) {
        test_cast_conformance(&array.into_array());
    }
}
