// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::arrays::FixedSizeList;
use crate::arrays::fixed_size_list::FixedSizeListData;
use crate::dtype::DType;
use crate::validity::Validity;

impl ArrayPlugin for FixedSizeList {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.is::<FixedSizeList>(),
            "FixedSizeList plugin cannot serialize {}",
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
            "FixedSizeList plugin does not recognize serialized ID {serialized_id}"
        );
        if !metadata.is_empty() {
            vortex_bail!(
                "FixedSizeListArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        vortex_ensure!(
            buffers.is_empty(),
            "`FixedSizeList::build` expects no buffers"
        );

        let DType::FixedSizeList(element_dtype, list_size, _) = &dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {:?}", dtype);
        };

        let validity = {
            if children.len() > 2 {
                vortex_bail!("`FixedSizeList::build` method expected 1 or 2 children")
            }

            if children.len() == 2 {
                let validity = children.get(1, &Validity::DTYPE, len)?;
                Validity::Array(validity)
            } else {
                debug_assert_eq!(children.len(), 1);
                Validity::from(dtype.nullability())
            }
        };

        let num_elements = len * (*list_size as usize);
        let elements = children.get(0, element_dtype.as_ref(), num_elements)?;

        let data =
            FixedSizeListData::try_build(elements.clone(), *list_size, validity.clone(), len)?;
        let slots = FixedSizeListData::make_slots(&elements, &validity, len);
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
