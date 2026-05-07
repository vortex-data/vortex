// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrowVTable`] impl for the UUID extension type.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeBinaryArray;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use arrow_schema::extension::Uuid as ArrowUuid;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrow::ArrowArrayExecutor;
use crate::arrow::ArrowVTable;
use crate::arrow::nulls;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::uuid::Uuid;
use crate::extension::uuid::UuidMetadata;
use crate::validity::Validity;

const UUID_BYTE_LEN: i32 = 16;

impl ArrowVTable for Uuid {
    fn vortex_ext_id(&self) -> ExtId {
        Uuid.id()
    }

    fn arrow_ext_name(&self) -> Option<&'static str> {
        Some(ArrowUuid::NAME)
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        _session: &VortexSession,
    ) -> VortexResult<Field> {
        let mut field = Field::new(
            name.to_string(),
            DataType::FixedSizeBinary(UUID_BYTE_LEN),
            dtype.is_nullable(),
        );
        field
            .try_with_extension_type(ArrowUuid)
            .vortex_expect("FixedSizeBinary[16] is correct type for ArrowUuid");
        Ok(field)
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<DType> {
        if !matches!(field.data_type(), DataType::FixedSizeBinary(UUID_BYTE_LEN)) {
            vortex_bail!(
                "UUID plugin requires FixedSizeBinary({UUID_BYTE_LEN}), got {}",
                field.data_type()
            );
        }
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            field.is_nullable().into(),
        );
        Ok(DType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype)?.erased(),
        ))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        // Vortex stores UUIDs as FixedSizeList<u8; 16>; Arrow's canonical UUID extension type is
        // backed by FixedSizeBinary[16]. We materialize the storage as Arrow's FixedSizeList and
        // reinterpret the byte buffer without copying.
        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();
        let storage_arrow_type = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            UUID_BYTE_LEN,
        );
        let arrow_storage = storage.execute_arrow(Some(&storage_arrow_type), ctx)?;

        let fsl = arrow_storage.as_fixed_size_list();
        let bytes = fsl
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .inner()
            .clone();

        Ok(Arc::new(FixedSizeBinaryArray::new(
            fsl.value_length(),
            bytes,
            fsl.nulls().cloned(),
        )))
    }

    fn from_arrow_array(&self, array: ArrowArrayRef, field: &Field) -> VortexResult<ArrayRef> {
        if !matches!(array.data_type(), DataType::FixedSizeBinary(UUID_BYTE_LEN)) {
            vortex_bail!(
                "UUID plugin requires FixedSizeBinary({UUID_BYTE_LEN}), got {}",
                array.data_type()
            );
        }

        let fsb = array.as_fixed_size_binary();
        let buffer = Buffer::from_arrow_buffer(fsb.values().clone(), Alignment::none());
        let u8_array = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(buffer),
            PType::U8,
            Validity::NonNullable,
        );
        let validity = nulls(fsb.nulls(), field.is_nullable());

        Ok(FixedSizeListArray::new(
            u8_array.into_array(),
            fsb.value_length() as u32,
            validity,
            fsb.len(),
        )
        .into_array())
    }
}
