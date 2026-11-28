// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Primitive`].

use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
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
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "Primitive plugin cannot serialize an array of another encoding",
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
            "Primitive plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
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

        let expected_len = ptype.byte_width().checked_mul(len).ok_or_else(|| {
            vortex_err!(
                "PrimitiveArray of {ptype} values has byte length overflowing usize for length {len}"
            )
        })?;
        if buffer.len() != expected_len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for {}, {}",
                buffer.len(),
                expected_len,
                ptype.byte_width(),
                len,
            );
        }

        // SAFETY: the buffer length and alignment are checked above.
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = unsafe { PrimitiveData::new_unchecked_from_handle(buffer, ptype, validity) };
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))?
        .into_array())
    }
}
