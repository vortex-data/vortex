// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

use crate::arrays::{StructArray, StructVTable};
use crate::operator::getitem::GetItemOperator;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, MaskExecution, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::validity::Validity;
use crate::vtable::PipelineVTable;
use crate::{Array, ArrayRef, Canonical, IntoArray};

impl PipelineVTable<StructVTable> for StructVTable {
    fn to_operator(array: &StructArray) -> VortexResult<Option<OperatorRef>> {
        let mut children = Vec::with_capacity(array.fields.len());
        for field in array.fields().iter() {
            if let Some(operator) = field.to_operator()? {
                children.push(operator);
            } else {
                // If any of the children can't be converted, bail out.
                return Ok(None);
            }
        }

        Ok(Some(Arc::new(StructOperator {
            dtype: array.dtype().clone(),
            len: array.len(),
            children,
            // validity: array.validity.clone(),
        })))
    }
}

/// An operator for a struct array.
#[derive(Debug)]
struct StructOperator {
    dtype: DType,
    children: Vec<OperatorRef>,
    len: usize,
    // FIXME(ngates): validity should be an operator too...
    // validity: Validity,
}

impl OperatorHash for StructOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.len.hash(state);
        for child in &self.children {
            child.operator_hash(state);
        }
    }
}

impl OperatorEq for StructOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.len == other.len
            && self.children.len() == other.children.len()
            && self
                .children
                .iter()
                .zip(other.children.iter())
                .all(|(a, b)| a.operator_eq(b))
    }
}

impl Operator for StructOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.struct")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.len
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(StructOperator {
            dtype: self.dtype.clone(),
            // validity: self.validity.clone(),
            children,
            len: self.len,
        }))
    }

    fn reduce_parent(
        &self,
        parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        // The only real things we know how to push-down are things that exclusively operate on
        // validity, or operate on a single field.
        if let Some(getitem) = parent.as_any().downcast_ref::<GetItemOperator>() {
            let field_idx = self
                .dtype
                .as_struct_fields_opt()
                .vortex_expect("Struct dtype must have fields")
                .find(getitem.field_name())
                .ok_or_else(|| {
                    vortex_err!(
                        "Field {} not found in struct {}",
                        getitem.field_name(),
                        &self.dtype
                    )
                })?;

            // FIXME(ngates): intersect validity
            return Ok(Some(self.children[field_idx].clone()));
        }

        Ok(None)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for StructOperator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        let children = self
            .children
            .iter()
            .map(|child| ctx.bind_project(child, Some(mask)))
            .try_collect()?;

        let mask = ctx.bind_mask(mask)?;

        // TODO(ngates): we need custom push down logic for selection over a struct array in case
        //  there are no children. Because in this case, we need to hold onto the selection mask
        //  to know the true length.

        Ok(Box::new(StructExecution {
            dtype: self.dtype.clone(),
            mask,
            children,
            // validity: self.validity.clone(),
        }))
    }
}

struct StructExecution {
    dtype: DType,
    mask: MaskExecution,
    children: Vec<BatchExecutionRef>,
    // validity: Validity,
}

#[async_trait]
impl BatchExecution for StructExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let children: Vec<_> =
            try_join_all(self.children.into_iter().map(|child| child.execute())).await?;
        let children: Vec<ArrayRef> = children
            .into_iter()
            .map(|canonical| canonical.into_array())
            .collect();

        // TODO(ngates): join at the same time as the children? Although we only need this if
        //  we have no children
        let mask = self.mask.await?;

        let array = StructArray::new(
            self.dtype
                .as_struct_fields_opt()
                .vortex_expect("Struct dtype must have fields")
                .names()
                .clone(),
            children,
            mask.true_count(),
            // self.validity,
            Validity::AllValid,
        );

        Ok(Canonical::Struct(array))
    }
}
