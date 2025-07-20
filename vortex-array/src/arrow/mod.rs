// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

mod array;
/// Arrow compute operations and conversion utilities.
pub mod compute;
mod convert;
mod datum;
mod iter;
mod record_batch;

pub use datum::*;
pub use iter::*;

use crate::arrow::compute::ToArrowOptions;

/// Trait for converting from Arrow arrays to Vortex arrays.
pub trait FromArrowArray<A> {
    /// Converts an Arrow array to a Vortex array.
    fn from_arrow(array: A, nullable: bool) -> Self;
}

/// Trait for converting Vortex arrays to Arrow arrays.
pub trait IntoArrowArray {
    /// Converts to Arrow using the array's preferred data type.
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef>;

    /// Converts to Arrow using the specified data type.
    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef>;
}

impl IntoArrowArray for crate::ArrayRef {
    /// Convert this [`crate::ArrayRef`] into an Arrow [`crate::ArrayRef`] by using the array's preferred
    /// Arrow [`DataType`].
    fn into_arrow_preferred(self) -> VortexResult<ArrowArrayRef> {
        compute::to_arrow_opts(&self, &ToArrowOptions { arrow_type: None })
    }

    fn into_arrow(self, data_type: &DataType) -> VortexResult<ArrowArrayRef> {
        compute::to_arrow_opts(
            &self,
            &ToArrowOptions {
                arrow_type: Some(data_type.clone()),
            },
        )
    }
}
