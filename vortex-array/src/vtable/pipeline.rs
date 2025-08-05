// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::Pipeline;
use crate::vtable::{NotSupported, VTable};
use vortex_error::{VortexResult, vortex_bail};

pub trait PipelineVTable<V: VTable> {
    /// Convert the current array into a [`Pipeline`].
    fn to_pipeline(array: &V::Array) -> VortexResult<Box<dyn Pipeline>>;
}

impl<V: VTable> PipelineVTable<V> for NotSupported {
    fn to_pipeline(array: &V::Array) -> VortexResult<Box<dyn Pipeline>> {
        vortex_bail!(
            "PipelineVTable::pipeline is not supported for this array type: {}",
            array.encoding_id()
        );
    }
}
