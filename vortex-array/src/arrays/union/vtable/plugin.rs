// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Union`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::arrays::Union;
use crate::arrays::union::UnionSlots;
use crate::arrays::union::array::make_union_parts;
use crate::arrays::union::union_type_ids_dtype;
use crate::dtype::DType;

impl ArrayPlugin for Union {
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
            "Union plugin cannot serialize an array of another encoding",
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
            "Union plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        vortex_ensure!(metadata.is_empty(), "UnionArray expects empty metadata");
        vortex_ensure!(buffers.is_empty(), "UnionArray expects no buffers");
        let DType::Union(variants, nullability) = dtype else {
            vortex_bail!("Expected union dtype, found {dtype}")
        };
        vortex_ensure_eq!(children.len(), UnionSlots::CHILDREN_OFFSET + variants.len());

        let type_ids = children.get(
            UnionSlots::TYPE_IDS,
            &union_type_ids_dtype(*nullability),
            len,
        )?;
        let sparse_children = variants
            .variants()
            .enumerate()
            .map(|(index, dtype)| children.get(UnionSlots::CHILDREN_OFFSET + index, &dtype, len))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Array::try_from_parts(make_union_parts(
            type_ids,
            variants.clone(),
            sparse_children,
        ))?
        .into_array())
    }
}
