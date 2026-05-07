// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::{BoolArray, FixedSizeListArray, PrimitiveArray};
use crate::arrow::{ArrowSessionExt, ArrowVTable, nulls};
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::extension::uuid::Uuid;
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_array::{Array, ArrayRef as ArrowArrayRef, FixedSizeBinaryArray};
use arrow_schema::extension::Uuid as ArrowUuid;
use arrow_schema::{DataType, Field};
use std::sync::Arc;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::PType;
use vortex_buffer::{Alignment, Buffer};
use vortex_error::{VortexExpect, VortexResult};
use vortex_session::VortexSession;

impl ArrowVTable for Uuid {
    // We implement a special execution pathway to make sure we transmute from Vortex's
    // FixedSizeList<u8; 16> format to Arrow's expected FixedSizeBinary[16] format.
    fn execute_arrow(
        &self,
        array: ArrayRef,
        physical_type: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        // This VTable can only handle Arrow UUID extension type
        if !physical_type.has_valid_extension_type::<ArrowUuid>() {
            return Ok(None);
        }

        // The Arrow canonical UUID extension type can only be applied to FixedSizeBinary[16], but
        // in Vortex we use a FixedSizeList<u8; 16>. We need to handle the conversion on our end.
        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = match executed.try_into_parts() {
            Ok(parts) => parts.slots[0].expect("ExtensionArray must have exactly 1 slot"),
            Err(array) => array.storage_array().clone(),
        };

        // Execute the storage array into FixedSizeList, then we convert it to FixedSizeBinary
        // (no copy).
        let values = ctx.session().arrow().execute_arrow(storage, todo!(), ctx)?;
        let fsl = values.as_fixed_size_list();
        let bytes = fsl
            .values()
            .as_primitive::<UInt8Type>()
            .values()
            .inner()
            .clone();

        Ok(Some(Arc::new(FixedSizeBinaryArray::new(
            fsl.value_length(),
            bytes,
            fsl.nulls().cloned(),
        ))))
    }

    // When we observe an Arrow FixedSizeBinary array with UUID extension metadata, we should
    // convert it into a Vortex FixedSizeList<u8; 16> which is how we store UUID data.
    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
    ) -> VortexResult<Option<ArrayRef>> {
        // Execute from an arrow array into a Vortex array. This should be doable
        // more or less without copying.
        if !field.has_valid_extension_type::<ArrowUuid>() {
            return Ok(None);
        }

        if !matches!(array.data_type(), DataType::FixedSizeBinary(_)) {
            return Ok(None);
        }

        // Cast the elements first
        let fsb = array.as_fixed_size_binary();

        let binary = fsb.values().clone();

        // TODO(aduffy): isn't this weird b/c we lose the alignment?
        let buffer = Buffer::from_arrow_buffer(binary, Alignment::none());

        // Capture values into nulls buffer

        let u8_array = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(buffer),
            PType::U8,
            Validity::NonNullable,
        );

        let validity = nulls(fsb.nulls(), field.is_nullable());

        Ok(Some(
            FixedSizeListArray::new(
                u8_array.into_array(),
                fsb.value_length() as u32,
                validity,
                fsb.len(),
            )
            .into_array(),
        ))
    }

    // The Arrow Field equivalent of a Vortex UUID is an Arrow UUID extension type.
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<Field>> {
        let DType::Extension(ext_dtype) = dtype else {
            return Ok(None);
        };

        if !ext_dtype.metadata_opt::<Uuid>().is_some() {
            return Ok(None);
        }

        let mut field = Field::new(
            name.to_string(),
            DataType::FixedSizeBinary(16),
            dtype.is_nullable(),
        );

        field
            .try_with_extension_type(ArrowUuid)
            .vortex_expect("FixedSizeBinary[16] is correct type for ArrowUuid");

        Ok(Some(field))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_to_arrow() {
        // Convert some of these other things to Arrow and make sure we can convert them back
        // again.
    }
}
