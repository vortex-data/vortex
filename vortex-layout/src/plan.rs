// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use crate::ArrayFuture;
use crate::LayoutReaderRef;
use crate::layouts::struct_::reader::StructReader;

pub trait ProjectionPlan: Send + Sync {
    fn evaluate(&self, row_range: &Range<u64>, mask: MaskFuture) -> VortexResult<ArrayFuture>;
}

pub type ProjectionPlanRef = Arc<dyn ProjectionPlan>;

pub fn build_projection_plan(
    reader: LayoutReaderRef,
    expr: Expression,
) -> VortexResult<ProjectionPlanRef> {
    let any_reader: Arc<dyn Any + Send + Sync> = reader.clone();
    if let Ok(struct_reader) = Arc::downcast::<StructReader>(any_reader) {
        let plan = struct_reader.projection_plan(expr)?;
        return Ok(Arc::new(StructProjectionPlan {
            reader: struct_reader,
            plan,
        }));
    }

    Ok(Arc::new(DefaultProjectionPlan {
        reader,
        expr: Arc::new(expr),
    }))
}

struct DefaultProjectionPlan {
    reader: LayoutReaderRef,
    expr: Arc<Expression>,
}

impl ProjectionPlan for DefaultProjectionPlan {
    fn evaluate(&self, row_range: &Range<u64>, mask: MaskFuture) -> VortexResult<ArrayFuture> {
        self.reader
            .projection_evaluation(row_range, &self.expr, mask)
    }
}

struct StructProjectionPlan {
    reader: Arc<StructReader>,
    plan: Arc<crate::layouts::struct_::reader::StructProjectionPlan>,
}

impl ProjectionPlan for StructProjectionPlan {
    fn evaluate(&self, row_range: &Range<u64>, mask: MaskFuture) -> VortexResult<ArrayFuture> {
        self.reader.projection_with_plan(self.plan.clone(), row_range, mask)
    }
}
