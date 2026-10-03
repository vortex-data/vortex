// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for array constructor validation.
//!
//! This module tests the validation logic for various array types to ensure
//! that constructors properly reject invalid inputs.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexError;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
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
            &mut array_session().create_execution_ctx(),
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
            &mut array_session().create_execution_ctx(),
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
    fn utf8_views(views: Vec<BinaryView>, buffer: Vec<u8>) -> bool {
        VarBinViewArray::try_new(
            Buffer::from_iter(views),
            Arc::new([ByteBuffer::from(buffer)]),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
            &mut array_session().create_execution_ctx(),
        )
        .is_ok()
    }

    #[test]
    fn test_varbinview_utf8_view_inside_a_character_rejected() {
        // The buffer is valid UTF-8 as a whole, but the view starts inside the two-byte `é`.
        let data = "héllo, a string long enough to be outlined".as_bytes();
        let view = BinaryView::make_view(&data[2..20], 0, 2);
        assert!(!utf8_views(vec![view], data.to_vec()));
    }

    #[test]
    fn test_varbinview_utf8_unreferenced_invalid_bytes_accepted() {
        // Bytes no view references may be anything, even in a buffer validated as a whole first.
        let mut data = "héllo, a string long enough to be outlined"
            .as_bytes()
            .to_vec();
        let len = data.len();
        data.push(0xFF);
        let view = BinaryView::make_view(&data[..len], 0, 0);
        assert!(utf8_views(vec![view], data));
    }

    #[test]
    fn test_varbinview_utf8_inlined() {
        assert!(utf8_views(
            vec![BinaryView::new_inlined("héllo".as_bytes())],
            vec![]
        ));
        assert!(!utf8_views(
            vec![BinaryView::new_inlined(&[b'a', 0xC3])],
            vec![]
        ));
    }

    #[rstest::rstest]
    #[case::whole(vec![0, 2, 3], "éa".as_bytes().to_vec(), Validity::NonNullable, true)]
    #[case::inside_a_character(vec![0, 1, 3], "éa".as_bytes().to_vec(), Validity::NonNullable, false)]
    #[case::invalid_bytes(vec![0, 1, 2], vec![b'a', 0xFF], Validity::NonNullable, false)]
    #[case::invalid_bytes_under_a_null(
        vec![0, 1, 2, 3],
        vec![b'a', 0xFF, b'b'],
        Validity::from_iter([true, false, true]),
        true
    )]
    fn test_varbin_utf8(
        #[case] offsets: Vec<i32>,
        #[case] bytes: Vec<u8>,
        #[case] validity: Validity,
        #[case] ok: bool,
    ) {
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
        );
        assert_eq!(result.is_ok(), ok);
    }

    /// Bytes that mix ASCII, every UTF-8 sequence length, and invalid bytes, so that random
    /// string boundaries land inside characters and on invalid bytes.
    fn random_bytes(rng: &mut StdRng, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len + 4);
        while bytes.len() < len {
            match rng.random_range(0..10) {
                0..=4 => bytes.push(rng.random_range(b'a'..=b'z')),
                5 => bytes.extend_from_slice("é".as_bytes()),
                6 => bytes.extend_from_slice("€".as_bytes()),
                7 => bytes.extend_from_slice("😀".as_bytes()),
                8 => bytes.push([0x80, 0xBF, 0xC3, 0xE2, 0xF0, 0xFF][rng.random_range(0..6)]),
                _ => bytes.push(rng.random()),
            }
        }
        bytes
    }

    fn random_validity(rng: &mut StdRng, len: usize) -> (Validity, Vec<bool>) {
        match rng.random_range(0..3) {
            0 => (Validity::NonNullable, vec![true; len]),
            1 => (Validity::AllValid, vec![true; len]),
            _ => {
                let valid: Vec<bool> = (0..len).map(|_| rng.random_bool(0.8)).collect();
                (Validity::from_iter(valid.iter().copied()), valid)
            }
        }
    }

    fn dtype_for(validity: &Validity) -> DType {
        match validity {
            Validity::NonNullable => DType::Utf8(Nullability::NonNullable),
            _ => DType::Utf8(Nullability::Nullable),
        }
    }

    /// The buffer-wide UTF-8 path must accept exactly the arrays whose every valid string is
    /// valid UTF-8 on its own.
    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "every length here is under 256"
    )]
    fn varbin_utf8_validation_matches_per_string() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0001);
        for _ in 0..2_000 {
            let len = rng.random_range(0..64);
            let bytes = random_bytes(&mut rng, len);
            let n = rng.random_range(1..8);
            let start = rng.random_range(0..=bytes.len());
            let mut offsets: Vec<u32> = (0..=n)
                .map(|_| rng.random_range(start..=bytes.len()) as u32)
                .collect();
            offsets.sort_unstable();
            let (validity, valid) = random_validity(&mut rng, n);
            let expected = offsets.windows(2).zip(&valid).all(|(o, &v)| {
                !v || std::str::from_utf8(&bytes[o[0] as usize..o[1] as usize]).is_ok()
            });
            let result = VarBinArray::try_new(
                PrimitiveArray::from_iter(offsets.clone()).into_array(),
                ByteBuffer::copy_from(&bytes),
                dtype_for(&validity),
                validity,
            );
            assert_eq!(
                result.is_ok(),
                expected,
                "bytes {bytes:?} offsets {offsets:?}"
            );
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "every length here is under 256"
    )]
    fn varbinview_utf8_validation_matches_per_string() {
        let mut rng = StdRng::seed_from_u64(0x5eed_0002);
        for _ in 0..2_000 {
            // Large buffers relative to the strings exercise the per-string path, small ones the
            // buffer-wide path.
            let data: Vec<Vec<u8>> = (0..rng.random_range(1..3))
                .map(|_| {
                    let len = rng.random_range(13..200);
                    random_bytes(&mut rng, len)
                })
                .collect();
            let n = rng.random_range(1..8);
            let mut strings = Vec::with_capacity(n);
            let mut views = Vec::with_capacity(n);
            for _ in 0..n {
                if rng.random_bool(0.3) {
                    let len = rng.random_range(0..=12);
                    let value = random_bytes(&mut rng, len);
                    let value = value[..value.len().min(12)].to_vec();
                    views.push(BinaryView::new_inlined(&value));
                    strings.push(value);
                } else {
                    let block = rng.random_range(0..data.len());
                    let buf = &data[block];
                    let start = rng.random_range(0..buf.len() - 12);
                    let end = rng.random_range(start + 13..=buf.len());
                    views.push(BinaryView::make_view(
                        &buf[start..end],
                        block as u32,
                        start as u32,
                    ));
                    strings.push(buf[start..end].to_vec());
                }
            }
            let (validity, valid) = random_validity(&mut rng, n);
            let expected = strings
                .iter()
                .zip(&valid)
                .all(|(s, &v)| !v || std::str::from_utf8(s).is_ok());
            let buffers: Arc<[ByteBuffer]> = data.iter().map(ByteBuffer::copy_from).collect();
            let result = VarBinViewArray::try_new(
                Buffer::copy_from(&views),
                buffers,
                dtype_for(&validity),
                validity,
                &mut array_session().create_execution_ctx(),
            );
            assert_eq!(result.is_ok(), expected, "data {data:?} views {views:?}");
        }
    }
}
