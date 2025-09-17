// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::operator::OperatorRef;
use crate::vtable::{NotSupported, VTable};

pub trait PipelineVTable<V: VTable> {
    /// Convert the current array into a [`Operator`].
    /// Returns `None` if the array cannot be converted to an operator.
    fn to_operator(array: &V::Array) -> VortexResult<Option<OperatorRef>>;
}

impl<V: VTable> PipelineVTable<V> for NotSupported {
    fn to_operator(_array: &V::Array) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
    }
}
