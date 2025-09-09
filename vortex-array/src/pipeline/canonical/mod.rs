// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod primitive;

use vortex_dtype::{DType, Nullability, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::Canonical;
use crate::pipeline::Kernel;
use crate::pipeline::bits::{
    AlignedBitSink, BitAlignedChunkedIterator, EmptyBitSink, TrueSliceIterator, UnalignedBitSink,
};
use crate::pipeline::canonical::bool::{export_bool, export_bool_nonnull_masked};
use crate::pipeline::canonical::primitive::export_primitive;
use crate::pipeline::operators::Operator;
use crate::pipeline::query::QueryPlan;

/// Export canonical data from a pipeline kernel with the given mask.
pub fn export_canonical_pipeline(
    dtype: &DType,
    len: usize,
    pipeline: &mut dyn Kernel,
    mask: &Mask,
) -> VortexResult<Canonical> {
    if mask.all_false() {
        return Ok(Canonical::empty(dtype));
    }

    match dtype {
        DType::Bool(nullability) => {
            export_bool(*nullability, mask, pipeline).map(Canonical::Bool)
        }
        DType::Primitive(ptype, nullability) => {
            export_primitive(*ptype, *nullability, mask, pipeline).map(Canonical::Primitive)
        }
        _ => vortex_bail!("Expected a bool or primitive array, got: {}", dtype),
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
