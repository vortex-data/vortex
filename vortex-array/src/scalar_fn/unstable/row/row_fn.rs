// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`RowFn`] contract for scalar functions whose natural kernel computes one row at a time.
//!
//! Implementations declare their arity and fallibility, then use [`RowFn::dispatch`] to select the
//! typed row signature for each supported dtype combination. Optional methods provide
//! serialization without putting persistence plumbing in the row kernel.

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::visitor::RowVisitor;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnId;

/// A strict scalar function whose row kernel cannot produce null from valid inputs.
///
/// This is stronger than
/// [`ScalarFnVTable::is_strict`](crate::scalar_fn::ScalarFnVTable::is_strict), which requires null
/// propagation but permits valid inputs to produce null. The framework derives output validity
/// only from input validity.
///
/// Declare argument names and use [`dispatch`](Self::dispatch) to select element and output types.
/// Every implementation receives the standard [`ScalarFnVTable`]. A public type that needs custom
/// vtable hooks can delegate its row kernel through [`row_fn_return_dtype`] and [`execute_rows`].
///
/// [`ScalarFnVTable`]: crate::scalar_fn::ScalarFnVTable
/// [`execute_rows`]: crate::scalar_fn::unstable::row::execute_rows
/// [`row_fn_return_dtype`]: crate::scalar_fn::unstable::row::row_fn_return_dtype
pub trait RowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, or [`EmptyOptions`](crate::scalar_fn::EmptyOptions) for none.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The arguments in display order. Its length is the function's exact arity.
    const ARG_NAMES: &'static [&'static str];

    /// Whether any dispatch can raise a semantic error.
    ///
    /// See [`ScalarFnVTable::is_fallible`](crate::scalar_fn::ScalarFnVTable::is_fallible) for a
    /// more detailed explanation of semantic errors.
    ///
    /// The framework checks dispatched element and result types. A conservative `true` is allowed.
    const FALLIBLE: bool;

    /// Returns the ID of the scalar function.
    fn id(&self) -> ScalarFnId;

    /// Serialize this function's options, or return `None` when the function is not serializable.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Restore options written by [`serialize`](Self::serialize).
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("Expression {} is not deserializable", self.id())
    }

    /// Choose element types for these input dtypes and visit the framework with them.
    ///
    /// Planning and execution both call this method, so its result **must** depend only on
    /// `options` and `args`. Cross-argument dtype validation belongs here.
    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult>;
}
