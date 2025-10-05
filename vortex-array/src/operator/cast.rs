// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, Operator, OperatorEq,
    OperatorHash, OperatorId, OperatorRef,
};
use crate::Canonical;
use async_trait::async_trait;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

/// An operator that casts an array to a different dtype.
#[derive(Debug)]
pub struct CastOperator {
    pub target_dtype: DType,
    pub child: OperatorRef,
}

impl OperatorHash for CastOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.target_dtype.hash(state);
        self.child.operator_hash(state);
    }
}

impl OperatorEq for CastOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.target_dtype == other.target_dtype && self.child.operator_eq(&other.child)
    }
}

impl CastOperator {
    pub fn new(target_dtype: DType, child: OperatorRef) -> Self {
        Self {
            target_dtype,
            child,
        }
    }
}

impl Operator for CastOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.cast")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.target_dtype
    }

    fn len(&self) -> usize {
        self.child.len()
    }

    fn children(&self) -> &[OperatorRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let child = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("CastOperator requires one child"))?;
        Ok(Arc::new(CastOperator {
            target_dtype: self.target_dtype.clone(),
            child,
        }))
    }

    fn reduce_children(&self) -> VortexResult<Option<OperatorRef>> {
        // Remove the cast if the child already has the target dtype.
        if self.child.dtype() == &self.target_dtype {
            return Ok(Some(self.child.clone()));
        }
        Ok(None)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }

    // TODO(ngates): some primitive casts can be done as a pipelined operator.
}

impl BatchOperator for CastOperator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        let child = ctx.bind_project(&self.child, Some(mask))?;

        // TODO(ngates): here we should match on the source and target dtypes and implement the
        //  casting logic we wish to see for canonical arrays.
        //  But first, we implement some common short-cut cases.

        // If the child already has the target dtype, just project the child.
        if self.child.dtype() == &self.target_dtype {
            return Ok(child);
        }

        // If the child is non-nullable, and the target is nullable, we have a simple executor.
        if self.target_dtype.is_nullable()
            && self.child.dtype().eq_ignore_nullability(&self.target_dtype)
        {
            return Ok(Box::new(CastToNullableExecution(child)));
        }

        vortex_bail!(
            "Casting from {} to {} is not implemented",
            self.child.dtype(),
            self.target_dtype
        );
    }
}

/// Execution that casts a non-nullable array to a nullable array.
pub struct CastToNullableExecution(BatchExecutionRef);
#[async_trait]
impl BatchExecution for CastToNullableExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let child = self.0.execute().await?;
        Ok(match child {
            Canonical::Null(arr) => Canonical::Null(arr),
            Canonical::Bool(arr) => {
                todo!("casting Bool to nullable not implemented yet")
            }
            Canonical::Primitive(_) => {
                todo!("casting Primitive to nullable not implemented yet")
            }
            Canonical::Decimal(_) => {
                todo!("casting Decimal to nullable not implemented yet")
            }
            Canonical::VarBinView(_) => {
                todo!("casting VarBinView to nullable not implemented yet")
            }
            Canonical::List(_) => {
                todo!("casting List to nullable not implemented yet")
            }
            Canonical::FixedSizeList(_) => {
                todo!("casting FixedSizeList to nullable not implemented yet")
            }
            Canonical::Struct(_) => {
                todo!("casting Struct to nullable not implemented yet")
            }
            Canonical::Extension(_) => {
                todo!("casting Extension to nullable not implemented yet")
            }
        })
    }
}
