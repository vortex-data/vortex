// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

mod convert;
mod datum;
mod executor;
mod iter;
mod null_buffer;
mod session;

pub(crate) use convert::nulls;
pub use datum::*;
pub use executor::*;
pub use iter::*;
pub use null_buffer::to_arrow_null_buffer;
pub use null_buffer::to_null_buffer;
pub use session::*;

use crate::ArrayRef;
#[expect(deprecated)]
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;

/// Construct a Vortex array from an Arrow array (or other Arrow container) of type `A`.
///
/// Implementations reuse the underlying Arrow buffers without copying wherever the Arrow and
/// Vortex memory layouts allow it.
pub trait FromArrowArray<A> {
    /// Convert `array` into a Vortex array whose [`DType`](crate::dtype::DType) has the requested
    /// `nullable` [`Nullability`](crate::dtype::Nullability).
    ///
    /// An Arrow array can carry a validity (null) buffer regardless of whether its schema declares
    /// the field nullable, so the desired nullability is supplied separately by the caller
    /// (typically from the corresponding Arrow `Field`'s `is_nullable`). This flag is reconciled
    /// with the array's physical nulls as follows:
    ///
    /// - `nullable == true`: the resulting validity is derived from the array's null buffer, or
    ///   all-valid when the array has none.
    /// - `nullable == false`: the array must contain no nulls, and the result is non-nullable.
    ///
    /// # Errors
    ///
    /// Returns an error if `nullable` is `false` but `array` physically contains one or more nulls
    /// (including an Arrow `NullArray`, which is entirely null), or if the Arrow data type is not
    /// supported.
    fn from_arrow(array: A, nullable: bool) -> VortexResult<Self>
    where
        Self: Sized;
}

#[deprecated(note = "Use `execute_arrow(None, ctx)` or `execute_arrow(Some(dt), ctx)` instead")]
pub trait IntoArrowArray {
    #[deprecated(note = "Use `execute_arrow(None, ctx)` instead")]
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef>;

    #[deprecated(note = "Use `execute_arrow(Some(data_type), ctx)` instead")]
    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef>;
}

#[allow(deprecated)]
impl IntoArrowArray for ArrayRef {
    /// Convert this [`crate::ArrayRef`] into an Arrow [`crate::ArrayRef`] by using the array's
    /// preferred (cheapest) Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow(None, &mut LEGACY_SESSION.create_execution_ctx())
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow(Some(data_type), &mut LEGACY_SESSION.create_execution_ctx())
    }
}
