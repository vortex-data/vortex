// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;

use crate::Pco;
use crate::PcoData;
impl CastReduce for Pco {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_nullable() || !array.array().all_valid()? {
            // TODO(joe): fixme
            // We cannot cast to non-nullable since the validity containing nulls is used to decode
            // the PCO array, this would require rewriting tables.
            return Ok(None);
        }
        // PCO (Pcodec) is a compression encoding that stores data in a compressed format.
        // It can efficiently handle nullability changes without decompression, but type changes
        // require decompression since the compression algorithm is type-specific.
        // PCO supports: F16, F32, F64, I16, I32, I64, U16, U32, U64
        if array.dtype().eq_ignore_nullability(dtype) {
            // Create a new validity with the target nullability
            let unsliced_validity =
                child_to_validity(&array.slots()[0], array.dtype().nullability());
            let new_validity =
                unsliced_validity.cast_nullability(dtype.nullability(), array.len())?;

            let data = PcoData::new(
                array.chunk_metas.clone(),
                array.pages.clone(),
                dtype.as_ptype(),
                array.metadata.clone(),
                array.unsliced_n_rows(),
            )
            ._slice(array.slice_start(), array.slice_stop());

            return Ok(Some(
                Pco::try_new(dtype.clone(), data, new_validity)?.into_array(),
            ));
        }

        // For other casts (e.g., numeric type changes), decode to canonical and let PrimitiveArray handle it
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::Pco;

    #[test]
    fn test_cast_pco_f32_to_f64() {
        let values = PrimitiveArray::from_iter([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let pco = Pco::from_primitive(&values, 0, 128).unwrap();

        let casted = pco
            .into_array()
            .cast(DType::Primitive(PType::F64, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0, 5.0])
        );
    }

    #[test]
    fn test_cast_pco_nullability_change() {
        // Test casting from NonNullable to Nullable
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]);
        let pco = Pco::from_primitive(&values, 0, 128).unwrap();

        let casted = pco
            .into_array()
            .cast(DType::Primitive(PType::U32, Nullability::Nullable))
            .unwrap();
        assert_arrays_eq!(
            casted,
            PrimitiveArray::new(buffer![10u32, 20, 30, 40], Validity::AllValid,)
        );
    }

    #[test]
    fn test_cast_sliced_pco_nullable_to_nonnullable() {
        let values = PrimitiveArray::new(
            buffer![10u32, 20, 30, 40, 50, 60],
            Validity::from_iter([true, true, true, true, true, true]),
        );
        let pco = Pco::from_primitive(&values, 0, 128).unwrap();
        let sliced = pco.slice(1..5).unwrap();
        let casted = sliced
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        // Verify the values are correct
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([20u32, 30, 40, 50]));
    }

    #[test]
    fn test_cast_sliced_pco_part_valid_to_nonnullable() {
        let values = PrimitiveArray::from_option_iter([
            None,
            Some(20u32),
            Some(30),
            Some(40),
            Some(50),
            Some(60),
        ]);
        let pco = Pco::from_primitive(&values, 0, 128).unwrap();
        let sliced = pco.slice(1..5).unwrap();
        let casted = sliced
            .cast(DType::Primitive(PType::U32, Nullability::NonNullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([20u32, 30, 40, 50]));
    }

    #[rstest]
    #[case::f32(PrimitiveArray::new(
        buffer![1.23f32, 4.56, 7.89, 10.11, 12.13],
        Validity::NonNullable,
    ))]
    #[case::f64(PrimitiveArray::new(
        buffer![100.1f64, 200.2, 300.3, 400.4, 500.5],
        Validity::NonNullable,
    ))]
    #[case::i32(PrimitiveArray::new(
        buffer![100i32, 200, 300, 400, 500],
        Validity::NonNullable,
    ))]
    #[case::u64(PrimitiveArray::new(
        buffer![1000u64, 2000, 3000, 4000],
        Validity::NonNullable,
    ))]
    #[case::single(PrimitiveArray::new(
        buffer![42.42f64],
        Validity::NonNullable,
    ))]
    fn test_cast_pco_conformance(#[case] values: PrimitiveArray) {
        let pco = Pco::from_primitive(&values, 0, 128).unwrap();
        test_cast_conformance(&pco.into_array());
    }
}
