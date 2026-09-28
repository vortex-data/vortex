// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ListViewMetadata;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
use crate::arrays::ListView;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewData;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

impl ArrayPlugin for ListView {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<ListView>().ok_or_else(|| {
            vortex_err!("ListView plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = ListViewMetadata {
            elements_len: view.elements().len() as u64,
            offset_ptype: PType::try_from(view.offsets().dtype())? as i32,
            size_ptype: PType::try_from(view.sizes().dtype())? as i32,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata,
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
            "ListView plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = ListViewMetadata::decode(metadata)?;
        vortex_ensure!(
            buffers.is_empty(),
            "`ListViewArray::build` expects no buffers"
        );

        let DType::List(element_dtype, _) = dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };

        let validity = if children.len() == 3 {
            Validity::from(dtype.nullability())
        } else if children.len() == 4 {
            let validity = children.get(3, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "`ListViewArray::build` expects 3 or 4 children, got {}",
                children.len()
            );
        };

        // Get elements with the correct length from metadata.
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.elements_len)?,
        )?;

        // Get offsets with proper type from metadata.
        let offsets = children.get(
            1,
            &DType::Primitive(metadata.offset_ptype(), Nullability::NonNullable),
            len,
        )?;

        // Get sizes with proper type from metadata.
        let sizes = children.get(
            2,
            &DType::Primitive(metadata.size_ptype(), Nullability::NonNullable),
            len,
        )?;

        ListViewData::validate(&elements, &offsets, &sizes, &validity)?;
        let data = ListViewData::try_new()?;
        let slots = ListViewData::make_slots(&elements, &offsets, &sizes, &validity, len);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
