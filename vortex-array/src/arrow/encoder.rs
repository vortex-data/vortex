// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrowEncoder`] — pluggable Vortex → Arrow array conversion.

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrow::ArrowSession;

/// Reference-counted pointer to an [`ArrowEncoder`].
pub type ArrowEncoderRef = Arc<dyn ArrowEncoder>;

/// Plugin trait that converts a Vortex [`ArrayRef`] into an Arrow array.
///
/// Encoders are registered against an [`crate::array::ArrayId`] (encoding-keyed) or an
/// [`crate::dtype::extension::ExtId`] (extension-keyed) on the [`crate::arrow::ArrowSession`].
/// Returning [`None`] from [`ArrowEncoder::to_arrow_array`] tells the dispatcher to fall through
/// to the canonical encoder. This is the only way an encoder can decline a request.
pub trait ArrowEncoder: 'static + Send + Sync + Debug {
    /// Returns the Arrow [`DataType`] this encoder would prefer to emit for `array`.
    ///
    /// `session` is provided so encoders for nested types (e.g.
    /// [`crate::arrays::List`]) can recursively resolve the preferred Arrow type of their
    /// children via [`ArrowSession::resolve_preferred_arrow_type`].
    ///
    /// Returning [`None`] defers to the canonical encoder's preference (e.g.
    /// [`DataType::Utf8View`] for [`crate::dtype::DType::Utf8`]). Implementations should only
    /// override the canonical preference when they can produce a cheaper Arrow representation
    /// (for example, [`crate::arrays::VarBin`] preferring offset-based [`DataType::Utf8`] over
    /// [`DataType::Utf8View`]).
    fn preferred_arrow_type(
        &self,
        array: &ArrayRef,
        session: &ArrowSession,
    ) -> VortexResult<Option<DataType>>;

    /// Convert `array` into an Arrow array of type `target`.
    ///
    /// Returning [`Ok(None)`] tells the dispatcher to canonicalize the array and re-dispatch
    /// through the canonical encoder. Encoders may decline requests they don't recognize but
    /// must not silently mis-convert.
    fn to_arrow_array(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>>;
}
