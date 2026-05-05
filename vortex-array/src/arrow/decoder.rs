// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrowDecoder`] — pluggable Arrow → Vortex array conversion.

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;

/// Reference-counted pointer to an [`ArrowDecoder`].
pub type ArrowDecoderRef = Arc<dyn ArrowDecoder>;

/// Plugin trait that converts an Arrow array into a Vortex [`ArrayRef`].
///
/// Decoders are registered as a chain on [`crate::arrow::ArrowSession`] and walked in
/// registration order, with user-registered decoders running before the built-in canonical
/// decoders so external crates can override built-in behavior.
///
/// Returning [`Ok(None)`] passes the request to the next decoder in the chain. Returning
/// [`Ok(Some(_))`] short-circuits the chain. The dispatcher hard-fails if no decoder claims
/// the request.
pub trait ArrowDecoder: 'static + Send + Sync + Debug {
    /// Try to decode `array` into a Vortex [`ArrayRef`].
    ///
    /// `field` carries the Arrow extension-name metadata (`ARROW:extension:name`) that lets
    /// extension-aware decoders dispatch on logical type rather than physical [`arrow_schema::DataType`].
    ///
    /// `session` is the active [`VortexSession`], available so decoders can recurse into child
    /// arrays via [`crate::arrow::ArrowSession`] porcelain.
    fn try_decode(
        &self,
        array: &dyn ArrowArray,
        field: &Field,
        session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>>;
}
