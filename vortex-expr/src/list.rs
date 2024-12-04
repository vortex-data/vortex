use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::ArrayData;
use vortex_array::compute::list_mean;
use vortex_dtype::field::Field;
use vortex_error::VortexResult;

use crate::{unbox_any, ExprRef, VortexExpr};

#[derive(Debug, Clone)]
pub struct ListMean {
    child: ExprRef,
}

impl ListMean {
    pub fn new_expr(child: ExprRef) -> ExprRef {
        Arc::new(Self { child })
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

impl Display for ListMean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mean({})", self.child)
    }
}

impl VortexExpr for ListMean {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        list_mean(&self.child.evaluate(batch)?)
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        self.child.collect_references(references)
    }
}

impl PartialEq<dyn Any> for ListMean {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x.child.eq(&self.child))
            .unwrap_or(false)
    }
}
