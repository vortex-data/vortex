use std::ops::Range;

use async_trait::async_trait;
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::compute::{filter, slice};
use vortex_array::Array;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

/// A blocking task that can be spawned by a [`crate::LayoutReader`].
#[derive(Debug, Clone)]
pub enum ScanTask {
    Filter(Mask),
    Expr(ExprRef),
    Slice(Range<usize>),
    Canonicalize,
}

impl ScanTask {
    pub fn execute(&self, array: &Array) -> VortexResult<Array> {
        match self {
            ScanTask::Filter(mask) => filter(array, mask),
            ScanTask::Expr(expr) => expr.evaluate(array),
            ScanTask::Slice(range) => slice(array, range.start, range.end),
            ScanTask::Canonicalize => {
                // TODO(ngates): replace this with into_canonical. We want a fully recursive
                //  canonicalize here, so we pretend by converting via Arrow.
                let is_nullable = array.dtype().is_nullable();
                Ok(Array::from_arrow(
                    array.clone().into_arrow_preferred()?,
                    is_nullable,
                ))
            }
        }
    }
}

/// A trait used to spawn and execute blocking tasks.
#[async_trait]
pub trait TaskExecutor: 'static + Send + Sync {
    async fn execute(&self, array: &Array, tasks: &[ScanTask]) -> VortexResult<Array>;
}

pub struct InlineTaskExecutor;

#[async_trait]
impl TaskExecutor for InlineTaskExecutor {
    async fn execute(&self, array: &Array, tasks: &[ScanTask]) -> VortexResult<Array> {
        let mut array = array.clone();
        for task in tasks {
            array = task.execute(&array)?;
        }
        Ok(array)
    }
}
