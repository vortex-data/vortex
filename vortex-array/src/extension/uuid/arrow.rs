// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow plugin impls for the UUID extension type.
//!
//! UUIDs are a canonical Arrow extension type backed by `FixedSizeBinary[16]`. The Vortex side
//! stores them as `FixedSizeList<u8; 16>`, so the conversion is a zero-copy reinterpretation
//! of the byte buffer in both directions.

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
use vortex_array::arrow::ArrowSession;
use vortex_array::arrow::has_valid_extension_type;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrow::ArrowExport;
use crate::arrow::ArrowExportVTable;
use crate::arrow::ArrowImport;
use crate::arrow::ArrowImportVTable;
use crate::arrow::ArrowSessionExt;
use crate::arrow::nulls;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtVTable;
use crate::extension::uuid::Uuid;
use crate::extension::uuid::UuidMetadata;
use crate::validity::Validity;

const UUID_BYTE_LEN: i32 = 16;

static ARROW_UUID: CachedId = CachedId::new(ArrowUuid::NAME);

impl ArrowExportVTable for Uuid {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_UUID
    }

    fn vortex_id(&self) -> Id {
        Uuid.id()
    }

    // Encode all of these.
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let mut field = Field::new(
            name.to_string(),
            DataType::FixedSizeBinary(UUID_BYTE_LEN),
            dtype.is_nullable(),
        );
        field
            .try_with_extension_type(ArrowUuid)
            .vortex_expect("FixedSizeBinary[16] is correct type for ArrowUuid");
        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_uuid = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Uuid>())
            .unwrap_or(false);
        if !is_uuid {
            return Ok(ArrowExport::Unsupported(array));
        }
        Ok(ArrowExport::Exported(try_fsl_to_fsb(array, ctx)?))
    }
}

impl ArrowImportVTable for Uuid {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_UUID
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        if !has_valid_extension_type::<ArrowUuid>(field) {
            return Ok(None);
        }

        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            field.is_nullable().into(),
        );

        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype)?.erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        _field: &Field,
        dtype: &DType,
    ) -> VortexResult<ArrowImport> {
        let DType::Extension(dtype) = dtype else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !matches!(array.data_type(), DataType::FixedSizeBinary(UUID_BYTE_LEN))
            || !dtype.is::<Uuid>()
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let fsb = array.as_fixed_size_binary();
        let buffer = Buffer::from_arrow_buffer(fsb.values().clone(), Alignment::none());
        let u8_array = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(buffer),
            PType::U8,
            Validity::NonNullable,
        );
        let validity = nulls(fsb.nulls(), dtype.is_nullable())?;

        let storage = FixedSizeListArray::new(
            u8_array.into_array(),
            fsb.value_length() as u32,
            validity,
            fsb.len(),
        )
        .into_array();
        Ok(ArrowImport::Imported(
            ExtensionArray::new(dtype.clone(), storage).into_array(),
        ))
    }
}

fn try_fsl_to_fsb(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef> {
    let executed = array.execute::<ExtensionArray>(ctx)?;
    let storage = executed.storage_array().clone();
    let storage_arrow_type = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        UUID_BYTE_LEN,
    );

    let storage_field = Field::new(
        String::new(),
        storage_arrow_type,
        storage.dtype().is_nullable(),
    );

    let session = ctx.session().clone();
    let arrow_storage = session
        .arrow()
        .execute_arrow(storage, Some(&storage_field), ctx)?;

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
