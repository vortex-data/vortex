// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Apache Arrow interoperability for Vortex.
//!
//! This crate owns every conversion between Vortex and Arrow: importing Arrow arrays, record
//! batches, schemas, and fields into Vortex ([`FromArrowArray`], [`FromArrowType`],
//! [`TryFromArrowType`]), and executing Vortex arrays into Arrow ([`ArrowSession::execute_arrow`]
//! and the [`ArrowArrayExecutor`] convenience trait).
//!
//! The [`ArrowSession`] session variable is created lazily on first use of
//! [`ArrowSessionExt::arrow`], so no explicit registration is required for plain conversions.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::legacy_session;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

mod convert;
mod datum;
pub mod dtype;
mod executor;
mod iter;
mod null_buffer;
mod run_end_import;
mod scalar;
mod session;
mod uuid;

pub use convert::IntoVortexArray;
pub(crate) use convert::nulls;
pub use datum::*;
pub use dtype::FromArrowType;
pub use dtype::ToArrowType;
pub use dtype::TryFromArrowType;
pub use executor::*;
pub use iter::*;
pub use null_buffer::to_arrow_null_buffer;
pub use null_buffer::to_null_buffer;
pub use scalar::ToArrowDatum;
pub use session::*;

/// Register vortex-arrow's session extensions into `session`.
///
/// This eagerly registers the [`ArrowSession`], which is otherwise created lazily on first use.
pub fn initialize(session: &VortexSession) {
    session.register(ArrowSession::default());
}

/// Construct a Vortex array from an Arrow array (or other Arrow container) of type `A`.
///
/// Implementations reuse the underlying Arrow buffers without copying wherever the Arrow and
/// Vortex memory layouts allow it.
pub trait FromArrowArray<A> {
    /// Convert `array` into a Vortex array whose [`DType`](vortex_array::dtype::DType) has the requested
    /// `nullable` [`Nullability`](vortex_array::dtype::Nullability).
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
    /// Convert this [`vortex_array::ArrayRef`] into an Arrow [`ArrowArrayRef`] by using the array's
    /// preferred (cheapest) Arrow [`DataType`].
    #[allow(clippy::disallowed_methods)]
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow(None, &mut legacy_session().create_execution_ctx())
    }

    #[allow(clippy::disallowed_methods)]
    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        self.execute_arrow(
            Some(data_type),
            &mut legacy_session().create_execution_ctx(),
        )
    }
}
