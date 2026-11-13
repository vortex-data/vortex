// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Masked`].

use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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
use crate::VortexSessionExecute;
use crate::array::validity_to_child;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedData;
use crate::validity::Validity;

impl ArrayPlugin for Masked {
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
            "Masked plugin cannot serialize an array of another encoding",
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
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "Masked plugin does not recognize the serialized ID",
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
                "MaskedArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffer, got {}", buffers.len());
        }

        vortex_ensure!(
            children.len() == 1 || children.len() == 2,
            "`MaskedArray::build` expects 1 or 2 children, got {}",
            children.len()
        );

        let child = children.get(0, &dtype.as_nonnullable(), len)?;

        let validity = if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            Validity::from(dtype.nullability())
        };

        let validity_child = validity_to_child(&validity, len);
        let data = MaskedData::try_new(
            len,
            child.all_valid(&mut session.create_execution_ctx())?,
            validity,
        )?;
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            smallvec![Some(child), validity_child],
        ))?
        .into_array())
    }
}
