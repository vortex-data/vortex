// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{StructArray, StructVTable};
use crate::operator::getitem::GetItemOperator;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchOperator, Operator, OperatorId, OperatorRef,
};
use crate::validity::Validity;
use crate::vtable::PipelineVTable;
use crate::{Array, Canonical, IntoArray};
use async_trait::async_trait;
use futures::future::try_join_all;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

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
            len: array.len(),
            children,
            // validity: array.validity.clone(),
        })))
    }
}

/// An operator for a struct array.
#[derive(Debug, Hash)]
struct StructOperator {
    dtype: DType,
    len: usize,
    children: Vec<OperatorRef>,
    // FIXME(ngates): validity should be an operator too...
    // validity: Validity,
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
            len: self.len,
            dtype: self.dtype.clone(),
            // validity: self.validity.clone(),
            children,
        }))
    }

    fn reduce_parent(&self, parent: OperatorRef, _child_idx: usize) -> VortexResult<OperatorRef> {
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
            return Ok(self.children[field_idx].clone());
        }

        Ok(parent)
    }
}

impl BatchOperator for StructOperator {
    fn bind(&self, ctx: &dyn BatchBindCtx) -> VortexResult<Box<dyn BatchExecution>> {
        let children = (0..self.children.len())
            .map(|i| ctx.child(i))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Box::new(StructExecution {
            len: self.len,
            dtype: self.dtype.clone(),
            children,
            // validity: self.validity.clone(),
        }))
    }
}

struct StructExecution {
    len: usize,
    dtype: DType,
    children: Vec<Box<dyn BatchExecution>>,
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
