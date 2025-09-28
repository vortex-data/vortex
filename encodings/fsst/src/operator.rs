// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::compute::filter;
use vortex_array::operator::slice::SliceOperator;
use vortex_array::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, MaskExecution, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::vtable::PipelineVTable;
use vortex_array::{Array, Canonical};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{FSSTArray, FSSTVTable};

impl PipelineVTable<FSSTVTable> for FSSTVTable {
    fn to_operator(array: &FSSTArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for FSSTArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dtype().hash(state);
        self.symbols().operator_hash(state);
        self.symbol_lengths().operator_hash(state);
        self.codes().operator_hash(state);
        self.uncompressed_lengths().operator_hash(state);
    }
}

impl OperatorEq for FSSTArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dtype() == other.dtype()
            && self.symbols().operator_eq(other.symbols())
            && self.symbol_lengths().operator_eq(other.symbol_lengths())
            && self.codes().operator_eq(other.codes())
            && self
                .uncompressed_lengths()
                .operator_eq(other.uncompressed_lengths())
    }
}

impl Operator for FSSTArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        Array::dtype(self.as_ref())
    }

    fn len(&self) -> usize {
        Array::len(self.as_ref())
    }

    fn children(&self) -> &[OperatorRef] {
        // TODO(ngates): we have varbin child
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn reduce_parent(
        &self,
        parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        // if let Some(filter) = parent.as_any().downcast_ref::<FilterOperator>() {
        //     return Ok(Some(Arc::new(FilteredFSSTOperator {
        //         array: self.clone(),
        //         mask: filter.mask().clone(),
        //     })));
        // }

        if let Some(slice) = parent.as_any().downcast_ref::<SliceOperator>() {
            return Ok(Some(Arc::new(
                self.slice(slice.range().clone())
                    .as_::<FSSTVTable>()
                    .clone(),
            )));
        }

        Ok(None)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for FSSTArray {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        Ok(Box::new(FSSTExecution {
            array: self.clone(),
            mask: ctx.bind_mask(mask)?,
        }))
    }
}

// TODO(ngates): obviously we should inline the canonical logic here
struct FSSTExecution {
    array: FSSTArray,
    mask: MaskExecution,
}

#[async_trait]
impl BatchExecution for FSSTExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let mask = self.mask.await?;
        Ok(filter(self.array.as_ref(), &mask)?.to_canonical())
    }
}
