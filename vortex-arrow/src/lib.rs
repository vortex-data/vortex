// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for Vortex arrays.
//!
//! This crate provides conversions between Vortex and Apache Arrow arrays, along with
//! the Arrow-based fallback implementations of scalar compute kernels. `vortex-array`
//! itself does not depend on any Arrow crate - all Arrow integration lives here.
//!
//! Simply linking this crate into a binary is enough: the fallback compute kernels
//! register themselves during static initialisation, and the arrow executor
//! registrations are picked up via `inventory`.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_array::ArrayRef;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::i256 as VortexI256;
use vortex_error::VortexResult;

/// Register the Arrow-backed compute fallbacks with `vortex-array`'s runtime hooks.
///
/// Call this once at program start-up (or at the beginning of any test) to make
/// the Arrow-backed implementations of `numeric`, `compare`, `boolean`, `like`,
/// `zip`, and VarBinView filtering available through
/// [`vortex_array::arrow_hooks::arrow_compute`].
///
/// Calling multiple times is harmless - subsequent registrations are ignored.
pub fn init() {
    compute::register();
}

/// Convert a Vortex `i256` into an Arrow [`arrow_buffer::i256`].
pub fn vortex_i256_to_arrow(value: VortexI256) -> arrow_buffer::i256 {
    let (low, high) = value.to_parts();
    arrow_buffer::i256::from_parts(low, high)
}

/// Convert an Arrow [`arrow_buffer::i256`] into a Vortex `i256`.
pub fn arrow_i256_to_vortex(value: arrow_buffer::i256) -> VortexI256 {
    let (low, high) = value.to_parts();
    VortexI256::from_parts(low, high)
}

mod buffer_ext;
mod compute;
mod convert;
mod datum;
mod dtype;
mod executor;
mod iter;
mod null_buffer;
mod record_batch;
mod scalar;

pub use buffer_ext::*;
pub use convert::*;
pub use datum::*;
pub use dtype::*;
pub use executor::*;
pub use iter::*;
pub use null_buffer::to_arrow_null_buffer;
pub use null_buffer::to_null_buffer;
pub use scalar::*;

/// Trait for producing a Vortex array from an Arrow array.
pub trait FromArrowArray<A> {
    /// Build a Vortex array from the provided Arrow array, with the given nullability.
    fn from_arrow(array: A, nullable: bool) -> VortexResult<Self>
    where
        Self: Sized;
}

/// Trait for converting a Vortex array into an Arrow array.
pub trait IntoArrowArray {
    /// Convert this array into an Arrow array using the array's preferred (cheapest)
    /// Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef>;

    /// Convert this array into an Arrow array with the requested Arrow [`DataType`].
    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef>;
}

impl IntoArrowArray for ArrayRef {
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow(None, &mut LEGACY_SESSION.create_execution_ctx())
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow(Some(data_type), &mut LEGACY_SESSION.create_execution_ctx())
    }
}

