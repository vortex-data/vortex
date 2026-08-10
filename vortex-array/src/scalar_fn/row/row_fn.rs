// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions computed one row at a time.

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::visitor::RowVisitor;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::RowExecution;
use crate::scalar_fn::ScalarFnId;

/// A scalar function computed one row at a time.
///
/// Declare the argument names and use [`dispatch`](Self::dispatch) to choose concrete element and
/// sink types for each accepted dtype combination. Implement
/// [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) directly for columnar kernels.
pub trait RowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, if any. Use [`EmptyOptions`](crate::scalar_fn::EmptyOptions)
    /// for none.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The arguments in display order. Its length is the function's exact arity.
    const ARG_NAMES: &'static [&'static str];

    /// Whether any legal dispatch can raise a semantic error as defined by
    /// [`ScalarFnVTable::is_fallible`](crate::scalar_fn::ScalarFnVTable::is_fallible).
    ///
    /// The framework checks this at compile time for every fallible dispatched element or result.
    /// A conservative `true` is allowed when only some dtype choices are fallible.
    const FALLIBLE: bool = false;

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
    /// Plan time and run time both call this method, so the choice **must** be a pure function of
    /// `options` and `args`. Cross-argument dtype validation belongs here.
    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult>;

    /// Try an encoding-aware implementation before decoding the inputs into row elements.
    ///
    /// `None` continues to the dispatched row loop. [`Output`](RowExecution::Output) skips that
    /// loop and may remain encoded or lazy. [`DeferredError`](RowExecution::DeferredError) reruns
    /// only valid rows when null payloads may have caused the failure. For non-nullary functions,
    /// batch execution calls this hook at most once with the original, unfiltered arrays; slices
    /// and compacted retries do not reach it.
    ///
    /// Like a dense row closure, this hook must be total over every stored payload, including
    /// payloads behind null rows. An `Err` is immediately user-visible and is never suppressed or
    /// retried through the row layer.
    ///
    /// # Requirements
    ///
    /// - `output.len()` **must** equal `args[0].len()`.
    /// - The output dtype **must** match the planned dtype when ignoring nullability.
    /// - The output **must not** introduce a null where every input is valid.
    ///
    /// The framework skips this hook for nullary functions.
    fn reduce_encoded(
        &self,
        options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        _ = (options, args, ctx);
        Ok(None)
    }
}
