// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kernels represent the CPU physical plan for array execution.

mod closure;
mod ready;
mod validate;

use std::fmt::Debug;

pub use closure::*;
pub use ready::*;
pub use validate::*;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;

/// A boxed reference to a kernel.
pub type KernelRef = Box<dyn Kernel>;

/// A trait representing the physical CPU execution of an array tree.
pub trait Kernel: 'static + Send + Debug {
    /// Execute the kernel and produce a vector result.
    fn execute(self: Box<Self>) -> VortexResult<Vector>;

    /// Report an estimated cost for computing this kernel over the given filter mask.
    ///
    /// This is obviously a very rough estimate, but is used to decide when a filter should be
    /// pushed through a kernel using [`Kernel::push_down_filter`].
    ///
    /// Return [`f64::INFINITY`] if the kernel has unknown cost, meaning filters will _always_
    /// be pushed through the kernel if possible.
    fn cost_estimate(&self, selection: &Mask) -> f64 {
        _ = selection;
        f64::INFINITY
    }

    /// Push a selection mask through this kernel.
    ///
    /// Return `Ok(None)` if the filter cannot be pushed down.
    fn push_down_filter(&self, selection: &Mask) -> VortexResult<Option<KernelRef>> {
        _ = selection;
        Ok(None)
    }
}
