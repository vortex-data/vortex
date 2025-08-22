// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pipeline module re-exports from vortex-vector
//!
//! This module provides a compatibility layer for existing code that expects
//! pipeline functionality to be available through vortex-array.

use std::rc::Rc;

use vortex_error::VortexResult;

use crate::vector::*;
use crate::vtable::{NotSupported, VTable};

pub mod canonical;

pub trait PipelineVTable<V: VTable> {
    /// Convert the current array into a [`Operator`].
    /// Returns `None` if the array cannot be converted to an operator.
    fn to_operator(array: &V::Array) -> VortexResult<Option<Rc<dyn Operator>>>;
}

impl<V: VTable> PipelineVTable<V> for NotSupported {
    fn to_operator(_array: &V::Array) -> VortexResult<Option<Rc<dyn Operator>>> {
        Ok(None)
    }
}
