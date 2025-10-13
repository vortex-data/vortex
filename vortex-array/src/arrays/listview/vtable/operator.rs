// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::operator::bool_runs::BoolRunsOperator;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, Operator, OperatorEq,
    OperatorHash, OperatorId, OperatorRef,
};
use crate::vtable::PipelineVTable;
use crate::{Array, Canonical};
use async_trait::async_trait;
use futures::try_join;
use itertools::Itertools;
use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

impl PipelineVTable<ListViewVTable> for ListViewVTable {
    fn to_operator(array: &ListViewArray) -> VortexResult<Option<OperatorRef>> {
        let Some(elements) = array.elements().to_operator()? else {
            return Ok(None);
        };
        let Some(offsets) = array.offsets().to_operator()? else {
            return Ok(None);
        };
        let Some(sizes) = array.sizes().to_operator()? else {
            return Ok(None);
        };

        // TODO(ngates): handle validity

        Ok(Some(Arc::new(ListViewOperator {
            dtype: array.dtype.clone(),
            children: [elements, offsets, sizes],
        })))
    }
}

#[derive(Debug)]
pub struct ListViewOperator {
    dtype: DType,
    // The three children are: elements, offsets, and sizes
    children: [OperatorRef; 3],
}

impl ListViewOperator {
    pub fn elements(&self) -> &OperatorRef {
        &self.children[0]
    }

    pub fn offsets(&self) -> &OperatorRef {
        &self.children[1]
    }

    pub fn sizes(&self) -> &OperatorRef {
        &self.children[2]
    }
}

impl OperatorHash for ListViewOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        todo!()
    }
}

impl OperatorEq for ListViewOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Operator for ListViewOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.listview")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.offsets().len()
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let (elements, offsets, sizes) = children
            .into_iter()
            .tuples()
            .next()
            .vortex_expect("Expected 3 children for ListViewOperator");
        Ok(Arc::new(ListViewOperator {
            dtype: self.dtype.clone(),
            children: [elements, offsets, sizes],
        }))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for ListViewOperator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        // We have two options:
        //  1. If the mask is dense, we should probably canonicalize the full elements array
        //     and leave the offsets/sizes alone.
        //  2. If the mask is sparse, we should push the mask down to the elements array, have
        //     a dense offsets array, and compute a new sizes array.
        //
        // We could construct the operators required for both options and switch at execution-time.
        // For now, we always do option 2.

        // We cannot push the mask as-is into the elements array.
        // We need to construct a new mask that is dependent on the offsets/sizes arrays.
        let elements_mask: OperatorRef = Arc::new(BoolRunsOperator::new(
            self.elements().len(),
            self.offsets().clone(),
            self.sizes().clone(),
        ));
        let elements = ctx.bind_project(self.elements(), Some(&elements_mask))?;

        // Once we have constructed an elements mask, we end up with a contiguous elements array.
        // Therefore, we only need to apply the mask to the sizes.
        let sizes = ctx.bind_project(self.sizes(), Some(mask))?;

        Ok(Box::new(ListViewProjection { sizes, elements }))
    }
}

struct ListViewProjection {
    sizes: BatchExecutionRef,
    elements: BatchExecutionRef,
}

#[async_trait]
impl BatchExecution for ListViewProjection {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let (sizes, elements) = try_join!(self.sizes.execute(), self.elements.execute())?;

        // We compute offsets as cumulative sum of sizes.
        todo!()
        // Ok(Canonical::List())
    }
}
