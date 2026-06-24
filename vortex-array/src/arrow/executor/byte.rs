// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericByteArray;
use arrow_array::types::BinaryViewType;
use arrow_array::types::ByteArrayType;
use arrow_array::types::StringViewType;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrow::byte_view::execute_varbinview_to_arrow;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;

/// Convert a Vortex array into an Arrow GenericBinaryArray.
pub(super) fn to_arrow_byte_array<T: ByteArrayType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef>
where
    T::Offset: NativePType,
{
    // If the Vortex array is already in VarBin format, we can directly convert it.
    if let Some(array) = array.as_opt::<VarBin>() {
        return varbin_to_byte_array::<T>(array, ctx);
    }

    // Otherwise, we execute the array to a VarBinViewArray and convert to Arrow ByteView,
    // then cast to the target byte array type.
    let varbinview = array.execute::<VarBinViewArray>(ctx)?;
    let binary_view = match varbinview.dtype() {
        DType::Utf8(_) => execute_varbinview_to_arrow::<StringViewType>(&varbinview, ctx),
        DType::Binary(_) => execute_varbinview_to_arrow::<BinaryViewType>(&varbinview, ctx),
        _ => unreachable!("VarBinViewArray must have Utf8 or Binary dtype"),
    }?;
    arrow_cast::cast(&binary_view, &T::DATA_TYPE).map_err(VortexError::from)
}

/// Convert a Vortex VarBinArray into an Arrow GenericBinaryArray.
fn varbin_to_byte_array<T: ByteArrayType>(
    array: ArrayView<'_, VarBin>,
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
    Ok(Arc::new(unsafe {
        GenericByteArray::<T>::new_unchecked(offsets, data, null_buffer)
    }))
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use rstest::rstest;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrow::ArrowArrayExecutor;
    use crate::arrow::executor::byte::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

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
    // Utf8 source can convert to all string types and binary types (via arrow_cast)
    #[case::utf8_to_binary(make_utf8_array(), DataType::Binary)]
    #[case::utf8_to_large_binary(make_utf8_array(), DataType::LargeBinary)]
    #[case::utf8_to_utf8(make_utf8_array(), DataType::Utf8)]
    #[case::utf8_to_large_utf8(make_utf8_array(), DataType::LargeUtf8)]
    #[case::utf8_to_utf8_view(make_utf8_array(), DataType::Utf8View)]
    // Binary source can convert to all binary types and string types (via arrow_cast)
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
        let mut ctx = array_session().create_execution_ctx();
        let arrow = vortex_array
            .into_array()
            .execute_arrow(Some(&target_dtype), &mut ctx)
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

        let mut ctx = array_session().create_execution_ctx();
        let arrow = vortex_array
            .into_array()
            .execute_arrow(Some(&target_dtype), &mut ctx)
            .unwrap();

        assert_eq!(arrow.data_type(), &target_dtype);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 1);
        assert!(!arrow.is_null(0));
        assert!(arrow.is_null(1));
        assert!(!arrow.is_null(2));
    }
}
