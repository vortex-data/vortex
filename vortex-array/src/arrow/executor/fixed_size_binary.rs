// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeBinaryArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrow::executor::validity::to_arrow_null_buffer;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::vtable::ValidityHelper;

/// Convert a Vortex array to an Arrow `FixedSizeBinaryArray`.
///
/// Accepts either an extension array (e.g. UUID) or a plain `FixedSizeList(Primitive(U8), size)`.
pub(super) fn to_arrow_fixed_size_binary(
    array: ArrayRef,
    size: i32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let storage = if array.dtype().is_extension() {
        array
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone()
    } else {
        array
    };

    let fsl = storage.execute::<FixedSizeListArray>(ctx)?;

    match fsl.dtype() {
        DType::FixedSizeList(elem, list_size, _)
            if *list_size == size as u32
                && matches!(elem.as_ref(), DType::Primitive(PType::U8, _)) => {}
        other => {
            vortex_bail!("FixedSizeBinary({size}) requires FixedSizeList(U8, {size}), got {other}");
        }
    }

    let elements = fsl.elements().clone().execute::<PrimitiveArray>(ctx)?;
    let values = elements.into_buffer::<u8>().into_arrow_buffer();
    let null_buffer = to_arrow_null_buffer(fsl.validity(), fsl.len(), ctx)?;

    Ok(Arc::new(FixedSizeBinaryArray::new(
        size,
        values,
        null_buffer,
    )))
}

#[cfg(test)]
mod tests {
    use arrow_array::FixedSizeBinaryArray;
    use arrow_schema::DataType;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrow::ArrowArrayExecutor;
    use crate::dtype::Nullability;
    use crate::extension::uuid::Uuid;
    use crate::extension::uuid::vtable::UUID_BYTE_LEN;
    use crate::validity::Validity;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "UUID_BYTE_LEN always fits u32/i32"
    )]
    #[test]
    fn test_uuid_to_fixed_size_binary() {
        let u1 = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let u2 = uuid::Uuid::parse_str("f47ac10b-58cc-4372-a567-0e02b2c3d479").unwrap();

        let flat: Vec<u8> = [u1.as_bytes(), &[0u8; 16], u2.as_bytes()]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        let elements = PrimitiveArray::new(Buffer::from(flat), Validity::NonNullable).into_array();
        let validity = Validity::from(BitBuffer::from_iter([true, false, true]));
        let fsl = FixedSizeListArray::try_new(elements, UUID_BYTE_LEN as u32, validity, 3)
            .unwrap()
            .into_array();
        let uuid_array = ExtensionArray::new(Uuid::default(Nullability::Nullable).erased(), fsl);

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let arrow = uuid_array
            .into_array()
            .execute_arrow(
                Some(&DataType::FixedSizeBinary(UUID_BYTE_LEN as i32)),
                &mut ctx,
            )
            .unwrap();

        let expected = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            [Some(u1.as_bytes().as_slice()), None, Some(u2.as_bytes())].into_iter(),
            UUID_BYTE_LEN as i32,
        )
        .unwrap();
        assert_eq!(arrow.as_ref(), &expected as &dyn arrow_array::Array);
    }
}
