// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
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
        vortex_ensure!(
            array.is::<Union>(),
            "Union plugin cannot serialize {}",
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
            "Union plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(metadata.is_empty(), "UnionArray expects empty metadata");
        vortex_ensure!(buffers.is_empty(), "UnionArray expects no buffers");
        let DType::Union(variants, nullability) = dtype else {
            vortex_bail!("Expected union dtype, found {dtype}")
        };
        vortex_ensure_eq!(
            children.len(),
            UnionSlots::CHILDREN_OFFSET + variants.len(),
            "UnionArray expected {} children, found {}",
            UnionSlots::CHILDREN_OFFSET + variants.len(),
            children.len()
        );

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
