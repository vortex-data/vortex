// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteArray;
use arrow_array::types::ByteArrayType;
use arrow_schema::DataType;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Constant;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_array::builders::DynVarBinBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::matcher::Matcher;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::executor::validity::to_arrow_null_buffer;

/// Matches the encodings [`to_arrow_byte_array`] requires for export.
///
/// `Chunked` and `Constant` are matched to stop execution before it destroys them: they have
/// specialized `append_to_builder` impls (chunk-wise append, scalar repeat) that the builder
/// fallback exploits.
struct ArrowByteExportable;

impl Matcher for ArrowByteExportable {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<VarBin>() || array.is::<Chunked>() || array.is::<Constant>()).then_some(array)
    }
}

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_byte_array<T: ByteArrayType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    if !matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_)) {
        vortex_bail!(
            "Cannot convert Vortex array with dtype {} to Arrow byte array type {}",
            array.dtype(),
            T::DATA_TYPE
        );
    }

    // Exporting Binary to Utf8 is the only combination that needs value validation, which
    // `GenericByteArray::try_new` performs over the whole values buffer. Routing such exports
    // through the builder guarantees the buffer is exactly the concatenated values, making
    // whole-buffer validation equivalent to per-value validation.
    let source_is_utf8 = matches!(array.dtype(), DType::Utf8(_));
    let target_is_utf8 = matches!(T::DATA_TYPE, DataType::Utf8 | DataType::LargeUtf8);
    let validate_utf8 = target_is_utf8 && !source_is_utf8;

    let array = array.execute_until::<ArrowByteExportable>(ctx)?;

    // If the Vortex array is in VarBin format, we can directly convert it.
    if !validate_utf8 && let Some(array) = array.as_opt::<VarBin>() {
        return varbin_to_byte_array::<T>(array, false, ctx);
    }

    let mut builder = DynVarBinBuilder::with_capacity(
        array.dtype().clone(),
        T::Offset::PTYPE == PType::I64,
        array.len(),
    );
    array.append_to_builder(&mut builder, ctx)?;
    varbin_to_byte_array::<T>(builder.finish_into_varbin().as_view(), validate_utf8, ctx)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
///
/// `validate_utf8` must be set when the Vortex dtype does not already guarantee the bytes are
/// valid for `T` (i.e. Binary source, Utf8 target). Validation covers the whole bytes buffer, so
/// callers must pass a compact array whose buffer holds only the concatenated values.
fn varbin_to_byte_array<T: ByteArrayType>(
    array: ArrayView<'_, VarBin>,
    validate_utf8: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    // We must cast the offsets to the required offset type.
    let offsets = array
        .offsets()
        .cast(DType::Primitive(T::Offset::PTYPE, Nullability::NonNullable))?
        .execute::<Canonical>(ctx)?
        .into_primitive()
        .to_buffer::<T::Offset>()
        .into_arrow_offset_buffer();

    let data = array.bytes().clone().into_arrow_buffer();

    let null_buffer = to_arrow_null_buffer(array.validity()?, array.len(), ctx)?;
    if validate_utf8 {
        return Ok(Arc::new(GenericByteArray::<T>::try_new(
            offsets,
            data,
            null_buffer,
        )?));
    }
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(offsets, data, null_buffer)
    }))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrowSessionExt;

    #[test]
    fn mask_wrapped_varbin_exports() -> VortexResult<()> {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();

        let varbin = VarBinArray::from_vec(
            vec!["hello", "world", "vortex"],
            DType::Utf8(Nullability::NonNullable),
        );
        let mask = BoolArray::from_iter([true, false, true]);
        let masked =
            MaskFn.try_new_array(3, EmptyOptions, [varbin.into_array(), mask.into_array()])?;

        let field = Field::new("s", DataType::Utf8, true);
        let arrow = session
            .arrow()
            .execute_arrow(masked, Some(&field), &mut ctx)?;

        let strings = arrow.as_string::<i32>();
        assert_eq!(strings.len(), 3);
        assert!(!strings.is_null(0));
        assert!(strings.is_null(1));
        assert_eq!(strings.value(0), "hello");
        assert_eq!(strings.value(2), "vortex");
        Ok(())
    }

    fn make_utf8_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_str(["hello", "world", "this is a longer string for testing"])
    }

    fn make_binary_array() -> VarBinViewArray {
        VarBinViewArray::from_iter_bin([
            b"hello".as_slice(),
            b"world".as_slice(),
            b"this is a longer string for testing".as_slice(),
        ])
    }

    #[rstest]
    // Utf8 source can convert to all string types and binary types
    #[case::utf8_to_binary(make_utf8_array(), DataType::Binary)]
    #[case::utf8_to_large_binary(make_utf8_array(), DataType::LargeBinary)]
    #[case::utf8_to_utf8(make_utf8_array(), DataType::Utf8)]
    #[case::utf8_to_large_utf8(make_utf8_array(), DataType::LargeUtf8)]
    #[case::utf8_to_utf8_view(make_utf8_array(), DataType::Utf8View)]
    // Binary source can convert to all binary types and string types
    #[case::binary_to_binary(make_binary_array(), DataType::Binary)]
    #[case::binary_to_large_binary(make_binary_array(), DataType::LargeBinary)]
    #[case::binary_to_utf8(make_binary_array(), DataType::Utf8)]
    #[case::binary_to_large_utf8(make_binary_array(), DataType::LargeUtf8)]
    #[case::binary_to_binary_view(make_binary_array(), DataType::BinaryView)]
    // Note: utf8 → binary_view and binary → utf8_view require Vortex cast kernels that don't exist
    fn test_vortex_string_binary_to_arrow(
        #[case] vortex_array: VarBinViewArray,
        #[case] target_dtype: DataType,
    ) {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), true);
        let arrow = session
            .arrow()
            .execute_arrow(vortex_array.into_array(), Some(&field), &mut ctx)
            .unwrap();

        assert_eq!(arrow.data_type(), &target_dtype);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 0);

        // Verify the actual values by converting back to bytes
        let expected: Vec<&[u8]> = vec![b"hello", b"world", b"this is a longer string for testing"];

        for (i, expected_bytes) in expected.iter().enumerate() {
            let actual_bytes: &[u8] = match &target_dtype {
                DataType::Binary => arrow.as_binary::<i32>().value(i),
                DataType::LargeBinary => arrow.as_binary::<i64>().value(i),
                DataType::Utf8 => arrow.as_string::<i32>().value(i).as_bytes(),
                DataType::LargeUtf8 => arrow.as_string::<i64>().value(i).as_bytes(),
                DataType::BinaryView => arrow.as_binary_view().value(i),
                DataType::Utf8View => arrow.as_string_view().value(i).as_bytes(),
                _ => unreachable!(),
            };
            assert_eq!(actual_bytes, *expected_bytes, "Mismatch at index {i}");
        }
    }

    #[rstest]
    #[case::utf8_to_binary(DType::Utf8(Nullability::Nullable), DataType::Binary)]
    #[case::utf8_to_large_binary(DType::Utf8(Nullability::Nullable), DataType::LargeBinary)]
    #[case::utf8_to_utf8(DType::Utf8(Nullability::Nullable), DataType::Utf8)]
    #[case::utf8_to_large_utf8(DType::Utf8(Nullability::Nullable), DataType::LargeUtf8)]
    #[case::utf8_to_utf8_view(DType::Utf8(Nullability::Nullable), DataType::Utf8View)]
    #[case::binary_to_binary(DType::Binary(Nullability::Nullable), DataType::Binary)]
    #[case::binary_to_large_binary(DType::Binary(Nullability::Nullable), DataType::LargeBinary)]
    #[case::binary_to_utf8(DType::Binary(Nullability::Nullable), DataType::Utf8)]
    #[case::binary_to_large_utf8(DType::Binary(Nullability::Nullable), DataType::LargeUtf8)]
    #[case::binary_to_binary_view(DType::Binary(Nullability::Nullable), DataType::BinaryView)]
    fn test_nullable_vortex_string_binary_to_arrow(
        #[case] vortex_dtype: DType,
        #[case] target_dtype: DataType,
    ) {
        let vortex_array = VarBinViewArray::from_iter(
            [Some(b"hello".as_slice()), None, Some(b"world".as_slice())],
            vortex_dtype,
        );

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype.clone(), true);
        let arrow = session
            .arrow()
            .execute_arrow(vortex_array.into_array(), Some(&field), &mut ctx)
            .unwrap();

        assert_eq!(arrow.data_type(), &target_dtype);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 1);
        assert!(!arrow.is_null(0));
        assert!(arrow.is_null(1));
        assert!(!arrow.is_null(2));
    }

    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    fn binary_with_invalid_utf8_to_string_returns_error(#[case] target_dtype: DataType) {
        let vortex_array = VarBinViewArray::from_iter_bin([
            b"hello".as_slice(),
            b"\xff\xfe invalid utf8".as_slice(),
        ]);

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype, true);
        let result =
            session
                .arrow()
                .execute_arrow(vortex_array.into_array(), Some(&field), &mut ctx);

        assert!(result.is_err());
    }

    #[rstest]
    #[case(DataType::Utf8)]
    #[case(DataType::LargeUtf8)]
    #[case(DataType::Binary)]
    #[case(DataType::LargeBinary)]
    fn incompatible_vortex_dtype_returns_error(#[case] target_dtype: DataType) {
        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("test_field", target_dtype, true);
        let result = session.arrow().execute_arrow(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Some(&field),
            &mut ctx,
        );

        assert!(result.is_err());
    }

    #[test]
    fn filtered_utf8_view_export_does_not_retain_unselected_buffers() -> VortexResult<()> {
        let unselected = "x".repeat(1 << 20);
        let array =
            VarBinViewArray::from_iter_str(["selected", unselected.as_str(), unselected.as_str()]);
        let filtered = array
            .into_array()
            .filter(Mask::from_iter([true, false, false]))?;

        let session = array_session();
        let mut ctx = session.create_execution_ctx();
        let arrow = session
            .arrow()
            .execute_arrow(filtered.into_array(), None, &mut ctx)?;

        assert_eq!(arrow.as_string_view().value(0), "selected");
        assert!(
            arrow.get_array_memory_size() < unselected.len(),
            "filtered export retained unselected payload: {} bytes",
            arrow.get_array_memory_size()
        );
        Ok(())
    }
}
