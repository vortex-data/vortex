// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::Operator;
use crate::pipeline::nodes::plan::PlanNode;
use crate::vtable::{NotSupported, VTable};
use vortex_error::{VortexResult, vortex_bail};

pub trait PipelineVTable<V: VTable> {
    /// Convert the current array into a [`PlanNode`].
    fn to_pipeline_plan(array: &V::Array) -> VortexResult<Box<dyn PlanNode>>;

    fn to_pipeline(array: &V::Array) -> VortexResult<Box<dyn Operator>>;
}

impl<V: VTable> PipelineVTable<V> for NotSupported {
    fn to_pipeline_plan(array: &V::Array) -> VortexResult<Box<dyn PlanNode>> {
        todo!()
    }

    fn to_pipeline(array: &V::Array) -> VortexResult<Box<dyn Operator>> {
        vortex_bail!(
            "PipelineVTable::pipeline is not supported for this array type: {}",
            array.encoding_id()
        );
    }
}
