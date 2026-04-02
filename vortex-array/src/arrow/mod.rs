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
mod record_batch;

pub use datum::*;
pub use executor::*;
pub use iter::*;
pub use null_buffer::to_arrow_null_buffer;
pub use null_buffer::to_null_buffer;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;

pub trait FromArrowArray<A> {
    fn from_arrow(array: A, nullable: bool) -> VortexResult<Self>
    where
        Self: Sized;
}

pub trait IntoArrowArray {
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef>;

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef>;
}

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
