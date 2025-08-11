// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexResult, vortex_bail};

use crate::pipeline::Kernel;
use crate::pipeline::operators::Operator;
use crate::vtable::{NotSupported, VTable};

pub trait PipelineVTable<V: VTable> {
    /// Convert the current array into a [`Operator`].
    fn to_operator(array: &V::Array) -> VortexResult<Box<dyn Operator>>;

    fn to_pipeline(array: &V::Array) -> VortexResult<Box<dyn Kernel>>;
}

impl<V: VTable> PipelineVTable<V> for NotSupported {
    fn to_operator(array: &V::Array) -> VortexResult<Box<dyn Operator>> {
        todo!()
    }

    fn to_pipeline(array: &V::Array) -> VortexResult<Box<dyn Kernel>> {
        vortex_bail!(
            "PipelineVTable::pipeline is not supported for this array type: {}",
            array.encoding_id()
        );
    }
}
