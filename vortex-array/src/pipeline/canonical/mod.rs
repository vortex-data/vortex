// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod primitive;

use vortex_dtype::{DType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::Canonical;
use crate::pipeline::Kernel;
use crate::pipeline::bits::{BitAlignedChunkedIterator, TrueSliceIterator};
use crate::pipeline::canonical::bool::export_bool_nonnull_masked;
use crate::pipeline::canonical::primitive::{export_primitive_nonnull, export_primitive_null};
use crate::pipeline::operators::Operator;
use crate::pipeline::query::QueryPlan;

/// Export canonical data from a pipeline kernel with the given mask.
pub fn export_canonical_pipeline(
    dtype: &DType,
    len: usize,
    pipeline: &mut dyn Kernel,
    mask: &Mask,
) -> VortexResult<Canonical> {
    match dtype {
        DType::Bool(Nullability::NonNullable) => {
            if mask.all_true() {
                export_bool_nonnull_masked(mask, pipeline).map(Canonical::Bool)
            } else {
                export_bool_nonnull_masked(mask, pipeline).map(Canonical::Bool)
            }
        }
        DType::Primitive(ptype, Nullability::NonNullable) => {
            if mask.all_true() {
                match_each_native_ptype!(ptype, |T| {
                    export_primitive_nonnull::<T, _>(TrueSliceIterator::new(len), pipeline)
                        .map(Canonical::Primitive)
                })
            } else {
                match_each_native_ptype!(ptype, |T| {
                    export_primitive_nonnull::<T, _>(
                        BitAlignedChunkedIterator::new(&mask.to_boolean_buffer()),
                        pipeline,
                    )
                    .map(Canonical::Primitive)
                })
            }
        }
        DType::Primitive(ptype, Nullability::Nullable) => {
            if mask.all_true() {
                return match_each_native_ptype!(ptype, |T| {
                    export_primitive_null::<T, _>(TrueSliceIterator::new(len), pipeline)
                        .map(Canonical::Primitive)
                });
            } else {
                return match_each_native_ptype!(ptype, |T| {
                    export_primitive_null::<T, _>(BitAlignedChunkedIterator::new(&mask.to_boolean_buffer()), pipeline)
                        .map(Canonical::Primitive)
                });
            }
        }
        _ => vortex_bail!("Expected a primitive array, got: {}", dtype),
    }
}

/// Export canonical data from an operator expression with a starting offset and mask.
pub fn export_canonical_pipeline_expr_offset(
    dtype: &DType,
    offset: usize,
    len: usize,
    expression: &dyn Operator,
    mask: &Mask,
) -> VortexResult<Canonical> {
    let plan = QueryPlan::new(expression)?;
    let mut pipeline = plan.executable_plan()?;
    pipeline.seek(offset)?;
    export_canonical_pipeline(dtype, len, &mut pipeline, mask)
}

/// Export canonical data from an operator expression with the given mask.
pub fn export_canonical_pipeline_expr(
    dtype: &DType,
    len: usize,
    expression: &dyn Operator,
    mask: &Mask,
) -> VortexResult<Canonical> {
    let plan = QueryPlan::new(expression)?;
    let mut pipeline = plan.executable_plan()?;
    export_canonical_pipeline(dtype, len, &mut pipeline, mask)
}
