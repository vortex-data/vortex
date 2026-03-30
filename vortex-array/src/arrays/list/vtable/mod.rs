// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::arrays::list::ListData;
use crate::arrays::list::compute::PARENT_KERNELS;
use crate::arrays::list::compute::rules::PARENT_RULES;
use crate::arrays::listview::list_view_from_list;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::metadata::DeserializeMetadata;
use crate::metadata::SerializeMetadata;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;
mod operations;
mod validity;
vtable!(List, List, ListData);

#[derive(Clone, prost::Message)]
pub struct ListMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
}

impl VTable for List {
    type ArrayData = ListData;

    type Metadata = ProstMetadata<ListMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &ListData) -> &Self {
        &List
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ListData) -> usize {
        array.offsets.len().saturating_sub(1)
    }

    fn dtype(array: &ListData) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &Array<Self>, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.elements.array_hash(state, precision);
        array.offsets.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &Array<Self>, other: &Array<Self>, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.elements.array_eq(&other.elements, precision)
            && array.offsets.array_eq(&other.offsets, precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &Array<Self>) -> usize {
        0
    }

    fn buffer(_array: &Array<Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &Array<Self>, idx: usize) -> Option<String> {
        vortex_panic!("ListArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(array: &Array<Self>) -> usize {
        2 + validity_nchildren(&array.validity)
    }

    fn child(array: &Array<Self>, idx: usize) -> ArrayRef {
        match idx {
            0 => array.elements().clone(),
            1 => array.offsets().clone(),
            2 => validity_to_child(&array.validity, array.len())
                .vortex_expect("ListArray validity child out of bounds"),
            _ => vortex_panic!("ListArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &Array<Self>, idx: usize) -> String {
        match idx {
            0 => "elements".to_string(),
            1 => "offsets".to_string(),
            2 => "validity".to_string(),
            _ => vortex_panic!("ListArray child_name index {idx} out of bounds"),
        }
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn metadata(array: &Array<Self>) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ListMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(SerializeMetadata::serialize(metadata)))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<ListMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ListData> {
        let validity = if children.len() == 2 {
            Validity::from(dtype.nullability())
        } else if children.len() == 3 {
            let validity = children.get(2, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 2 or 3 children, got {}", children.len());
        };

        let DType::List(element_dtype, _) = &dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.0.elements_len)?,
        )?;

        let offsets = children.get(
            1,
            &DType::Primitive(metadata.0.offset_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        ListData::try_new(elements, offsets, validity)
    }

    fn with_children(array: &mut Self::ArrayData, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2 || children.len() == 3,
            "ListArray expects 2 or 3 children, got {}",
            children.len()
        );

        let mut iter = children.into_iter();
        let elements = iter
            .next()
            .vortex_expect("children length already validated");
        let offsets = iter
            .next()
            .vortex_expect("children length already validated");
        let validity = if let Some(validity_array) = iter.next() {
            Validity::Array(validity_array)
        } else {
            Validity::from(array.dtype.nullability())
        };

        let new_array = ListData::try_new(elements, offsets, validity)?;
        *array = new_array;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            list_view_from_list((*array).clone(), ctx)?.into_array(),
        ))
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct List;

impl List {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.list");
}
