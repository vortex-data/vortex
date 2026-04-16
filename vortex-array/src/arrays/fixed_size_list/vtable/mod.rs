// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::fixed_size_list::FixedSizeListData;
use crate::arrays::fixed_size_list::array::NUM_SLOTS;
use crate::arrays::fixed_size_list::array::SLOT_NAMES;
use crate::arrays::fixed_size_list::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
mod kernel;
mod operations;
mod validity;

/// A [`FixedSizeList`]-encoded Vortex array.
pub type FixedSizeListArray = Array<FixedSizeList>;

#[derive(Clone, Debug)]
pub struct FixedSizeList;

impl ArrayHash for FixedSizeListData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        let _precision = precision;
        self.degenerate_len.hash(state);
    }
}

impl ArrayEq for FixedSizeListData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        let _precision = precision;
        self.degenerate_len == other.degenerate_len
    }
}

impl VTable for FixedSizeList {
    type ArrayData = FixedSizeListData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fixed_size_list");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FixedSizeListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FixedSizeListArray buffer_name index {idx} out of bounds")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Self::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn validate(
        &self,
        data: &FixedSizeListData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "FixedSizeListArray expected {NUM_SLOTS} slots, found {}",
            slots.len()
        );
        let DType::FixedSizeList(_, list_size, nullability) = dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {dtype:?}");
        };
        let elements = slots[crate::arrays::fixed_size_list::array::ELEMENTS_SLOT]
            .as_ref()
            .vortex_expect("FixedSizeListArray elements slot");
        vortex_ensure!(
            if *list_size == 0 {
                data.degenerate_len == len
            } else {
                elements.len() / *list_size as usize == len
            },
            "FixedSizeListArray length {} does not match outer length {}",
            len,
            len
        );

        let actual_dtype =
            DType::FixedSizeList(Arc::new(elements.dtype().clone()), *list_size, *nullability);
        vortex_ensure!(
            &actual_dtype == dtype,
            "FixedSizeListArray dtype {} does not match outer dtype {}",
            actual_dtype,
            dtype
        );

        Ok(())
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
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
        Ok(crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}
