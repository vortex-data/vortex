// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::arrays::{StructArray, StructVTable};
use crate::operator::getitem::GetItemOperator;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::validity::Validity;
use crate::vtable::PipelineVTable;
use crate::{Array, Canonical, IntoArray};

impl PipelineVTable<StructVTable> for StructVTable {
    fn to_operator(array: &StructArray) -> VortexResult<Option<OperatorRef>> {
        let mut children = Vec::with_capacity(array.fields.len());
        for field in array.fields() {
            if let Some(operator) = field.to_operator()? {
                children.push(operator);
            } else {
                // If any of the children can't be converted, bail out.
                return Ok(None);
            }
        }

        Ok(Some(Arc::new(StructOperator {
            dtype: array.dtype().clone(),
            bounds: array.len().into(),
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
    bounds: LengthBounds,
    // FIXME(ngates): validity should be an operator too...
    // validity: Validity,
}

impl OperatorHash for StructOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.bounds.hash(state);
        for child in &self.children {
            child.operator_hash(state);
        }
    }
}

impl OperatorEq for StructOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.bounds == other.bounds
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

    fn bounds(&self) -> LengthBounds {
        self.bounds
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let bounds = LengthBounds::intersect_all(children.iter().map(|c| c.bounds()));
        Ok(Arc::new(StructOperator {
            dtype: self.dtype.clone(),
            // validity: self.validity.clone(),
            children,
            bounds,
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
}

impl BatchOperator for StructOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        let children = (0..self.children.len())
            .map(|i| ctx.child(i))
            .collect::<VortexResult<Vec<_>>>()?;

        // TODO(ngates): we need custom push down logic for selection over a struct array in case
        //  there are no children. Because in this case, we need to hold onto the selection mask
        //  to know the true length.

        Ok(Box::new(StructExecution {
            len: self
                .bounds
                .maybe_len()
                .ok_or_else(|| vortex_err!("StructOperator must have a known length"))?,
            dtype: self.dtype.clone(),
            children,
            // validity: self.validity.clone(),
        }))
    }
}

struct StructExecution {
    len: usize,
    dtype: DType,
    children: Vec<BatchExecutionRef>,
    // validity: Validity,
}

#[async_trait]
impl BatchExecution for StructExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let children: Vec<_> =
            try_join_all(self.children.into_iter().map(|child| child.execute())).await?;
        let children = children
            .into_iter()
            .map(|canonical| canonical.into_array())
            .collect();

        let array = StructArray::new(
            self.dtype
                .as_struct_fields_opt()
                .vortex_expect("Struct dtype must have fields")
                .names()
                .clone(),
            children,
            self.len,
            // self.validity,
            Validity::AllValid,
        );

        Ok(Canonical::Struct(array))
    }
}
