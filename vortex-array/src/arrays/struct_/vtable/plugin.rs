// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Struct`].

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::array::EmptyArrayData;
use crate::arrays::Struct;
use crate::arrays::struct_::array::struct_slots_with_capacity;
use crate::dtype::DType;
use crate::validity::Validity;

impl ArrayPlugin for Struct {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "Struct plugin cannot serialize an array of another encoding",
        );
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "Struct plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
        if !metadata.is_empty() {
            vortex_bail!(
                "StructArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (Validity::from(*nullability), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            (Validity::Array(validity), 1_usize)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let mut slots = struct_slots_with_capacity(&validity, len, struct_dtype.nfields());
        for i in 0..struct_dtype.nfields() {
            let child_dtype = struct_dtype
                .field_by_index(i)
                .vortex_expect("no out of bounds");
            slots.push(Some(children.get(
                non_data_children + i,
                &child_dtype,
                len,
            )?));
        }

        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            EmptyArrayData,
            slots,
        ))?
        .into_array())
    }
}
