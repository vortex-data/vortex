// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::StructArray;
use crate::operator::ops::getitem::GetItemOperator;
use crate::operator::{ArrayOperator, BatchBindCtx, BatchExecution, BatchOperator};
use crate::validity::Validity;
use crate::{Canonical, IntoArray};
use async_trait::async_trait;
use futures::future::try_join_all;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};

/// An operator for a struct array.
struct StructOperator {
    dtype: DType,
    len: usize,
    children: Vec<Arc<dyn ArrayOperator>>,
    validity: Validity,
}

impl ArrayOperator for StructOperator {
    fn id(&self) -> Arc<str> {
        Arc::from("vortex.struct")
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

    fn children(&self) -> &[Arc<dyn ArrayOperator>] {
        &self.children
    }

    fn with_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ArrayOperator>>,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
        Ok(Arc::new(StructOperator {
            len: self.len,
            dtype: self.dtype.clone(),
            validity: self.validity.clone(),
            children,
        }))
    }

    fn optimize(
        &self,
        parent: Arc<dyn ArrayOperator>,
        _child_idx: usize,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
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
            validity: self.validity.clone(),
        }))
    }
}

struct StructExecution {
    len: usize,
    dtype: DType,
    children: Vec<Box<dyn BatchExecution>>,
    validity: Validity,
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
            self.validity,
        );

        Ok(Canonical::Struct(array))
    }
}
