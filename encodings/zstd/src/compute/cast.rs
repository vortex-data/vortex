// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{ZstdArray, ZstdVTable};

impl CastKernel for ZstdVTable {
    fn cast(&self, array: &ZstdArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_nullable() || !array.all_valid() {
            // We cannot cast to non-nullable since the validity containing nulls is used to decode
            // the ZSTD array, this would require rewriting tables.
            return Ok(None);
        }
        // ZstdArray is a general-purpose compression encoding using Zstandard compression.
        // It can handle nullability changes without decompression by updating the validity
        // bitmap, but type changes require decompression since the compressed data is
        // type-specific and Zstd operates on raw bytes.
        if array.dtype().eq_ignore_nullability(dtype) {
            // Create a new validity with the target nullability
            let new_validity = array
                .unsliced_validity
                .clone()
                .cast_nullability(dtype.nullability(), array.len())?;

            return Ok(Some(
                ZstdArray::new(
                    array.dictionary.clone(),
                    array.frames.clone(),
                    dtype.clone(),
                    array.metadata.clone(),
                    array.unsliced_n_rows(),
                    new_validity,
                )
                ._slice(array.slice_start(), array.slice_stop())
                .into_array(),
            ));
        }

        // For other casts (e.g., type changes), decode to canonical and let the underlying array handle it
        Ok(None)
    }
}

register_kernel!(CastKernelAdapter(ZstdVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::validity::Validity;
    use vortex_array::{ToCanonical, assert_arrays_eq};
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::ZstdArray;

    #[test]
    fn test_cast_zstd_i32_to_i64() {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![1i32, 2, 3, 4, 5]),
            Validity::NonNullable,
        );
        let zstd = ZstdArray::from_primitive(&values, 0, 0).unwrap();

        let casted = cast(
            zstd.as_ref(),
            &DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        let decoded = casted.to_primitive();
        assert_arrays_eq!(decoded, PrimitiveArray::from_iter([1i64, 2, 3, 4, 5]));
    }

    #[test]
    fn test_cast_zstd_nullability_change() {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![10u32, 20, 30, 40]),
            Validity::NonNullable,
        );
        let zstd = ZstdArray::from_primitive(&values, 0, 0).unwrap();

        let casted = cast(
            zstd.as_ref(),
            &DType::Primitive(PType::U32, Nullability::Nullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_cast_sliced_zstd_nullable_to_nonnullable() {
        let values = PrimitiveArray::new(
            Buffer::copy_from(vec![10u32, 20, 30, 40, 50, 60]),
            Validity::from_iter([true, true, true, true, true, true]),
        );
        let zstd = ZstdArray::from_primitive(&values, 0, 128).unwrap();
        let sliced = zstd.slice(1..5);
        let casted = cast(
            sliced.as_ref(),
            &DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        // Verify the values are correct
        let decoded = casted.to_primitive();
        let u32_values = decoded.as_slice::<u32>();
        assert_eq!(u32_values, &[20, 30, 40, 50]);
    }

    #[test]
    fn test_cast_sliced_zstd_part_valid_to_nonnullable() {
        let values = PrimitiveArray::from_option_iter([
            None,
            Some(20u32),
            Some(30),
            Some(40),
            Some(50),
            Some(60),
        ]);
        let zstd = ZstdArray::from_primitive(&values, 0, 128).unwrap();
        let sliced = zstd.slice(1..5);
        let casted = cast(
            sliced.as_ref(),
            &DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        let decoded = casted.to_primitive();
        let expected = PrimitiveArray::from_iter([20u32, 30, 40, 50]);
        assert_arrays_eq!(decoded, expected);
    }

    #[rstest]
    #[case::i32(PrimitiveArray::new(
        Buffer::copy_from(vec![100i32, 200, 300, 400, 500]),
        Validity::NonNullable,
    ))]
    #[case::f64(PrimitiveArray::new(
        Buffer::copy_from(vec![1.1f64, 2.2, 3.3, 4.4, 5.5]),
        Validity::NonNullable,
    ))]
    #[case::single(PrimitiveArray::new(
        Buffer::copy_from(vec![42i64]),
        Validity::NonNullable,
    ))]
    #[case::large(PrimitiveArray::new(
        Buffer::copy_from((0..1000).map(|i| i as u32).collect::<Vec<_>>()),
        Validity::NonNullable,
    ))]
    fn test_cast_zstd_conformance(#[case] values: PrimitiveArray) {
        let zstd = ZstdArray::from_primitive(&values, 0, 0).unwrap();
        test_cast_conformance(zstd.as_ref());
    }
}
