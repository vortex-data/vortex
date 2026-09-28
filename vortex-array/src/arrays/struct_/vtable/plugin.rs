// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::EmptyArrayData;
use crate::array::VTable;
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
        vortex_ensure!(
            array.is::<Struct>(),
            "Struct plugin cannot serialize {}",
            array.encoding_id()
        );
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            serialized_id,
            dtype,
            len,
            metadata,
            buffers: _,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Struct plugin does not recognize serialized ID {serialized_id}"
        );
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

        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData).with_slots(slots),
        )?
        .into_array())
    }
}
