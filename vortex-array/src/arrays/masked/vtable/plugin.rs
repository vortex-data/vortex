// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
use crate::array::validity_to_child;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedData;
use crate::legacy_session;
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
        vortex_ensure!(
            array.is::<Masked>(),
            "Masked plugin cannot serialize {}",
            array.encoding_id()
        );
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            vec![],
        )))
    }

    #[allow(clippy::disallowed_methods)]
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
            "Masked plugin does not recognize serialized ID {serialized_id}"
        );
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

        let validity_slot = validity_to_child(&validity, len);
        let data = MaskedData::try_new(
            len,
            child.all_valid(&mut legacy_session().create_execution_ctx())?,
            validity,
        )?;
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data)
                .with_slots(smallvec![Some(child), validity_slot]),
        )?
        .into_array())
    }
}
