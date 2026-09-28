// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
use crate::arrays::VarBinView;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::VarBinViewData;
use crate::validity::Validity;

impl ArrayPlugin for VarBinView {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.is::<VarBinView>(),
            "VarBinView plugin cannot serialize {}",
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
        session: &VortexSession,
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
            "VarBinView plugin does not recognize serialized ID {serialized_id}"
        );
        if !metadata.is_empty() {
            vortex_bail!(
                "VarBinViewArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let Some((views_handle, data_handles)) = buffers.split_last() else {
            vortex_bail!("Expected at least 1 buffer, got 0");
        };

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 children, got {}", children.len());
        };

        let views_nbytes = views_handle.len();
        let expected_views_nbytes = len
            .checked_mul(size_of::<BinaryView>())
            .ok_or_else(|| vortex_err!("views byte length overflow for len={len}"))?;
        if views_nbytes != expected_views_nbytes {
            vortex_bail!(
                "Expected views buffer length {} bytes, got {} bytes",
                expected_views_nbytes,
                views_nbytes
            );
        }

        // If any buffer is on device, skip host validation and use try_new_handle.
        if buffers.iter().any(|b| b.is_on_device()) {
            let data = VarBinViewData::try_new_handle(
                views_handle.clone(),
                Arc::from(data_handles.to_vec()),
                dtype.clone(),
                validity.clone(),
            )?;
            let slots = VarBinViewData::make_slots(&validity, len);
            return Ok(Array::try_from_parts(
                ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
            )?
            .into_array());
        }

        let data_buffers = data_handles
            .iter()
            .map(|b| b.as_host().clone())
            .collect::<Vec<_>>();
        let views = Buffer::<BinaryView>::from_byte_buffer(views_handle.clone().as_host().clone());

        let data = VarBinViewData::try_new(
            views,
            Arc::from(data_buffers),
            dtype.clone(),
            validity.clone(),
            &mut session.create_execution_ctx(),
        )?;
        let slots = VarBinViewData::make_slots(&validity, len);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
