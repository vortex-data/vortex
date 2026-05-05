// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities to work with `Arrow` data and types.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

pub mod canonical;
mod convert;
mod datum;
pub mod decoder;
pub mod decoders;
pub mod dtype_converter;
pub mod encoder;
pub mod encoders;
mod executor;
mod iter;
mod null_buffer;
mod record_batch;
mod session;

pub use datum::*;
pub use decoder::ArrowDecoder;
pub use decoder::ArrowDecoderRef;
pub use dtype_converter::ArrowDTypeConverter;
pub use dtype_converter::ArrowDTypeConverterRef;
pub use dtype_converter::ArrowDTypeReader;
pub use dtype_converter::ArrowDTypeReaderRef;
pub use encoder::ArrowEncoder;
pub use encoder::ArrowEncoderRef;
pub use executor::*;
pub use iter::*;
pub use null_buffer::to_arrow_null_buffer;
pub use null_buffer::to_null_buffer;
pub use session::ArrowSession;
pub use session::ArrowSessionExt;

use crate::ArrayRef;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;

/// Convert an Arrow array into a Vortex [`ArrayRef`].
///
/// Prefer the porcelain on [`ArrowSession`] (`session.arrow().from_arrow_array(...)`),
/// which dispatches through registered [`ArrowDecoder`] plugins. This trait is retained
/// as the underlying canonical implementation that the default decoder delegates to.
pub trait FromArrowArray<A> {
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
