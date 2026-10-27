// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for array constructor validation.
//!
//! This module tests the validation logic for various array types to ensure
//! that constructors properly reject invalid inputs.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbinview::build_views::BinaryView;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_chunked_array_validation_failure_mismatched_dtypes() {
        // Invalid case: chunks have different dtypes.
        let chunk1 = buffer![1i32, 2, 3].into_array();
        let chunk2 = buffer![4i64, 5, 6].into_array();
        let result = ChunkedArray::try_new(vec![chunk1, chunk2], PType::I32.into());

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("expected type: i32 but instead got i64"),
            "{err}"
        );
    }

    #[test]
    fn test_decimal_array_validation_success() {
        // Valid case: buffer and validity have matching lengths.
        let buffer = Buffer::from_iter([100i64, 200, 300]);
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

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("DecimalArray validity len does not match outer length"),
            "{err}"
        );
    }

    #[test]
    fn test_primitive_array_validation_failure_length_mismatch() {
        // Invalid case: validity length doesn't match buffer length.
        let buffer = Buffer::from_iter([1i32, 2, 3]);
        let validity = Validity::from_iter([true, false]); // Length 2, buffer is length 3.
        let result = PrimitiveArray::try_new(buffer, validity);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Buffer and validity length mismatch: buffer=3, validity=2"),
            "{err}"
        );
    }

    #[test]
    fn test_list_array_validation_failure_offsets_out_of_bounds() {
        let mut ctx = SESSION.create_execution_ctx();
        // Invalid case: last offset exceeds elements length.
        let elements = buffer![1i32, 2, 3].into_array();
        let offsets = buffer![0i64, 2, 5].into_array(); // 5 > 3.
        let result = ListArray::try_new(elements, offsets, Validity::NonNullable, &mut ctx);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Max offset 5 is beyond the length of the elements array 3"),
            "{err}"
        );
    }

    #[test]
    fn test_fixed_size_list_array_validation_failure_length_mismatch() {
        // Invalid case: elements length doesn't match list_size * len.
        let elements = buffer![1i32, 2, 3, 4, 5].into_array(); // 5 elements.
        let result = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 3); // Expects 2 * 3 = 6.

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("the `elements` array has the incorrect number of elements"),
            "{err}"
        );
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
            &mut SESSION.create_execution_ctx(),
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
            &mut SESSION.create_execution_ctx(),
        );

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("view at index 0 references invalid buffer: 1 out of bounds"),
            "{err}"
        );
    }

    #[test]
    fn test_struct_array_validation_failure_field_length_mismatch() {
        // Invalid case: fields have different lengths.
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![4.0f64, 5.0].into_array(); // Length 2, not 3.
        let fields = vec![field1, field2];
        let names = ["a", "b"];
        let result = StructArray::try_new(names.into(), fields, 3, Validity::NonNullable);

        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("StructArray field 1 has length 2 but expected 3"),
            "{err}"
        );
    }

    #[rstest::rstest]
    #[case::whole(vec![0, 2, 3], "éa".as_bytes().to_vec(), Validity::NonNullable, true)]
    #[case::inside_a_char(vec![0, 1, 3], "éa".as_bytes().to_vec(), Validity::NonNullable, false)]
    #[case::invalid_bytes(vec![0, 1, 2], vec![b'a', 0xFF], Validity::NonNullable, false)]
    #[case::invalid_bytes_at_a_null(
        vec![0, 1, 2, 3],
        vec![b'a', 0xFF, b'b'],
        Validity::from_iter([true, false, true]),
        true
    )]
    #[case::decreasing_offsets(vec![0, 2, 1], "abc".as_bytes().to_vec(), Validity::NonNullable, false)]
    #[case::offset_past_the_end(vec![0, 5, 1], "abc".as_bytes().to_vec(), Validity::NonNullable, false)]
    fn test_varbin_utf8_validation(
        #[case] offsets: Vec<i32>,
        #[case] bytes: Vec<u8>,
        #[case] validity: Validity,
        #[case] ok: bool,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let nullability = if matches!(validity, Validity::NonNullable) {
            Nullability::NonNullable
        } else {
            Nullability::Nullable
        };
        let result = VarBinArray::try_new(
            Buffer::from(offsets).into_array(),
            ByteBuffer::from(bytes),
            DType::Utf8(nullability),
            validity,
            &mut ctx,
        );
        assert_eq!(result.is_ok(), ok);
    }
}
