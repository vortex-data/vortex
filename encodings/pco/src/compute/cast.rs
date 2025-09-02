// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{PcoArray, PcoVTable};

impl CastKernel for PcoVTable {
    fn cast(&self, array: &PcoArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // PCO (Pcodec) is a compression encoding that stores data in a compressed format.
        // It can efficiently handle nullability changes without decompression, but type changes
        // require decompression since the compression algorithm is type-specific.
        // PCO supports: F16, F32, F64, I16, I32, I64, U16, U32, U64
        if array.dtype().eq_ignore_nullability(dtype) {
            // Create a new validity with the target nullability
            let new_validity = array
                .unsliced_validity
                .clone()
                .cast_nullability(dtype.nullability(), array.len())?;

            return Ok(Some(
                PcoArray::new(
                    array.chunk_metas.clone(),
                    array.pages.clone(),
                    dtype.clone(),
                    array.metadata.clone(),
                    array.unsliced_n_rows(),
                    new_validity,
                )
                ._slice(array.slice_start(), array.slice_stop())
                .into_array(),
            ));
        }

        // For other casts (e.g., numeric type changes), decode to canonical and let PrimitiveArray handle it
        Ok(None)
    }
}

register_kernel!(CastKernelAdapter(PcoVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::PcoArray;

    #[test]
    fn test_cast_pco_f32_to_f64() {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]),
            vortex_array::validity::Validity::NonNullable,
        );
        let pco = PcoArray::from_primitive(&values, 0, 128).unwrap();

        let casted = cast(
            pco.as_ref(),
            &DType::Primitive(PType::F64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::F64, Nullability::NonNullable)
        );

        let decoded = casted.to_primitive();
        let f64_values = decoded.as_slice::<f64>();
        assert_eq!(f64_values.len(), 5);
        assert!((f64_values[0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cast_pco_nullability_change() {
        // Test casting from NonNullable to Nullable
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![10u32, 20, 30, 40]),
            vortex_array::validity::Validity::NonNullable,
        );
        let pco = PcoArray::from_primitive(&values, 0, 128).unwrap();

        let casted = cast(
            pco.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[rstest]
    #[case::f32(PrimitiveArray::new(
        Buffer::copy_from(vec![1.23f32, 4.56, 7.89, 10.11, 12.13]),
        vortex_array::validity::Validity::NonNullable,
    ))]
    #[case::f64(PrimitiveArray::new(
        Buffer::copy_from(vec![100.1f64, 200.2, 300.3, 400.4, 500.5]),
        vortex_array::validity::Validity::NonNullable,
    ))]
    #[case::i32(PrimitiveArray::new(
        Buffer::copy_from(vec![100i32, 200, 300, 400, 500]),
        vortex_array::validity::Validity::NonNullable,
    ))]
    #[case::u64(PrimitiveArray::new(
        Buffer::copy_from(vec![1000u64, 2000, 3000, 4000]),
        vortex_array::validity::Validity::NonNullable,
    ))]
    #[case::single(PrimitiveArray::new(
        Buffer::copy_from(vec![42.42f64]),
        vortex_array::validity::Validity::NonNullable,
    ))]
    fn test_cast_pco_conformance(#[case] values: PrimitiveArray) {
        let pco = PcoArray::from_primitive(&values, 0, 128).unwrap();
        test_cast_conformance(pco.as_ref());
    }
}
