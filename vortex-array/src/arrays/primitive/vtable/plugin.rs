// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Alignment;
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
use crate::array::VTable;
use crate::arrays::Primitive;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::arrays::primitive::PrimitiveData;
use crate::dtype::PType;

impl ArrayPlugin for Primitive {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.is::<Primitive>(),
            "Primitive plugin cannot serialize {}",
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
            buffers,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "Primitive plugin does not recognize serialized ID {serialized_id}"
        );
        if !metadata.is_empty() {
            vortex_bail!(
                "PrimitiveArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let buffer = fixed_width::single_buffer(buffers)?;

        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;

        let ptype = PType::try_from(dtype)?;

        vortex_ensure!(
            buffer.is_aligned_to(Alignment::new(ptype.byte_width())),
            "Misaligned buffer cannot be used to build PrimitiveArray of {ptype}"
        );

        if buffer.len() != ptype.byte_width() * len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for {}, {}",
                buffer.len(),
                ptype.byte_width() * len,
                ptype.byte_width(),
                len,
            );
        }

        // SAFETY: the buffer length and alignment are checked above.
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = unsafe { PrimitiveData::new_unchecked_from_handle(buffer, ptype, validity) };
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
