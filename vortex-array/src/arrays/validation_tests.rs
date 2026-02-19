// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for array constructor validation.
//!
//! This module tests the validation logic for various array types to ensure
//! that constructors properly reject invalid inputs.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;

    use crate::IntoArray;
    use crate::arrays::BinaryView;
    use crate::arrays::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn test_chunked_array_validation_success() {
        // Valid case: all chunks have the same dtype.
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i32, 5, 6].into_array();
        let result = ChunkedArray::try_new(vec![chunk1, chunk2], PType::I32.into());
        assert!(result.is_ok());
    }

    #[test]
    fn test_chunked_array_validation_failure_mismatched_dtypes() {
        // Invalid case: chunks have different dtypes.
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i64, 5, 6].into_array();
        let result = ChunkedArray::try_new(vec![chunk1, chunk2], PType::I32.into());

        assert!(matches!(result, Err(VortexError::MismatchedTypes(_, _, _))));
        assert!(result.is_err());
    }

    #[test]
    fn test_decimal_array_validation_success() {
        // Valid case: buffer and validity have matching lengths.
        let buffer = Buffer::from_iter([100i128, 200, 300]);
        let decimal_dtype = crate::dtype::DecimalDType::new(10, 2);
        let result = DecimalArray::try_new(buffer, decimal_dtype, Validity::NonNullable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decimal_array_validation_failure_length_mismatch() {
        // Invalid case: validity length doesn't match buffer length.
        let buffer = Buffer::from_iter([100i128, 200, 300]);
        let validity = Validity::from_iter([true, false]); // Length 2, buffer is length 3.
        let decimal_dtype = crate::dtype::DecimalDType::new(10, 2);
        let result = DecimalArray::try_new(buffer, decimal_dtype, validity);

        assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
        assert!(result.is_err());
    }

    #[test]
    fn test_primitive_array_validation_success() {
        // Valid case: buffer and validity have matching lengths.
        let buffer = Buffer::from_iter([1i32, 2, 3]);
        let result = PrimitiveArray::try_new(buffer, Validity::NonNullable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_primitive_array_validation_failure_length_mismatch() {
        // Invalid case: validity length doesn't match buffer length.
        let buffer = Buffer::from_iter([1i32, 2, 3]);
        let validity = Validity::from_iter([true, false]); // Length 2, buffer is length 3.
        let result = PrimitiveArray::try_new(buffer, validity);

        assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
        assert!(result.is_err());
    }

    #[test]
    fn test_varbin_array_validation_success() {
        // Valid case: offsets are monotonically increasing and within bounds.
        let offsets = buffer![0i32, 3, 6, 10].into_array();
        let bytes = ByteBuffer::from(vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result = VarBinArray::try_new(
            offsets,
            bytes,
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_varbin_array_validation_non_monotonic_offsets_accepted() {
        // VarBin does not validate monotonicity of offsets at construction time.
        // Sortedness is enforced at the builder level instead.
        let offsets = buffer![0i32, 3, 2, 5].into_array(); // 3 -> 2 is decreasing.
        let bytes = ByteBuffer::from(vec![0u8, 1, 2, 3, 4]);
        let result = VarBinArray::try_new(
            offsets,
            bytes,
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_list_array_validation_success() {
        // Valid case: offsets are monotonically increasing.
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let offsets = buffer![0i64, 2, 3, 5].into_array();
        let result = ListArray::try_new(elements, offsets, Validity::NonNullable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_array_validation_failure_offsets_out_of_bounds() {
        // Invalid case: last offset exceeds elements length.
        let elements = buffer![1i32, 2, 3].into_array();
        let offsets = buffer![0i64, 2, 5].into_array(); // 5 > 3.
        let result = ListArray::try_new(elements, offsets, Validity::NonNullable);

        assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_size_list_array_validation_success() {
        // Valid case: elements length matches list_size * len.
        let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let result = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fixed_size_list_array_validation_failure_length_mismatch() {
        // Invalid case: elements length doesn't match list_size * len.
        let elements = buffer![1i32, 2, 3, 4, 5].into_array(); // 5 elements.
        let result = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 3); // Expects 2 * 3 = 6.

        assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
        assert!(result.is_err());
    }

    #[test]
    fn test_varbinview_array_validation_success() {
        // Valid case: simple inline strings.
        // Create inline views (length <= 12).
        let view1 = BinaryView::new_inlined(b"foo");
        let view2 = BinaryView::new_inlined(b"bar");

        let views = Buffer::from_iter([view1, view2]);
        let result = VarBinViewArray::try_new(
            views,
            Arc::new([]),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_varbinview_array_validation_failure_buffer_index_out_of_bounds() {
        // Invalid case: view references non-existent buffer.
        // Create a view that references buffer 1, but we only have 1 buffer (index 0).
        let data = b"this is a long string that needs a buffer";
        let view = BinaryView::make_view(data, 1, 0); // Buffer index 1.

        let views = Buffer::from_iter([view]);
        let buffers = Arc::new([ByteBuffer::from(data.to_vec())]);

        let result = VarBinViewArray::try_new(
            views,
            buffers,
            DType::Binary(Nullability::NonNullable),
            Validity::NonNullable,
        );

        assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
        assert!(result.is_err());
    }

    #[test]
    fn test_struct_array_validation_success() {
        // Valid case: all fields have the same length.
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![4.0f64, 5.0, 6.0].into_array();
        let fields = vec![field1, field2];
        let names = ["a", "b"];
        let result = StructArray::try_new(names.into(), fields, 3, Validity::NonNullable);
        assert!(result.is_ok());
    }

    #[test]
    fn test_struct_array_validation_failure_field_length_mismatch() {
        // Invalid case: fields have different lengths.
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![4.0f64, 5.0].into_array(); // Length 2, not 3.
        let fields = vec![field1, field2];
        let names = ["a", "b"];
        let result = StructArray::try_new(names.into(), fields, 3, Validity::NonNullable);

        assert!(matches!(result, Err(VortexError::InvalidArgument(_, _))));
        assert!(result.is_err());
    }
}
