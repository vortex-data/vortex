// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions computed one row at a time.

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::SinkResult;

/// A scalar function computed one row at a time.
///
/// An implementor declares its argument names, then [`dispatch`](Self::dispatch) picks the concrete
/// element and sink types for a batch. The planning visit reads dense safety, decode fallibility,
/// and decode cost from that concrete choice; no representative element types are needed.
///
/// A function whose kernel is columnar rather than row-at-a-time (negating a whole bit buffer, a
/// zero-copy unwrap) is not a `RowFn`, and implements
/// [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) directly.
pub trait RowFn: 'static + Sized + Clone + Send + Sync {
    /// Options for this function, if any. Use [`EmptyOptions`](crate::scalar_fn::EmptyOptions)
    /// for none.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The arguments in display order. Its length is the function's exact arity.
    const ARG_NAMES: &'static [&'static str];

    /// Whether any legal dispatch can fail while decoding or computing a row.
    ///
    /// The framework verifies that every fallible dispatched element or result implies this value.
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
    /// This is where a per-batch width match lives (`match_each_float_ptype!` and friends panic
    /// outside their width class, so check the class first), and where cross-argument dtype
    /// constraints belong, since per-argument validation runs inside the visit. Plan time and run
    /// time both come through here, so the choice **must** be a pure function of `options` and
    /// `args`.
    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out>;

    /// An encoding-aware rewrite, tried on the input arrays before the row loop.
    ///
    /// `Some` skips the row loop entirely, which makes this the escape hatch for a function that is
    /// row-shaped in general but has a bulk answer for some encodings: reading stored values back out
    /// of a wrapper encoding, or handing back a child array whole. The result may be lazy and
    /// nullable, but its nulls **must** be a subset of the rows the lifting will mask, and it
    /// **must** have one row per row of `args`, which on the filter strategy is the _filtered_ count
    /// rather than the original one. Size the result from `args`, which are filtered to match, and
    /// never from a length captured elsewhere.
    ///
    /// Whether the arrays still carry their original encoding depends on the execution path.
    /// Dense execution always passes them through untouched. Valid-only execution does too when
    /// no row is null; for a mixed mask, branch-and-skip also passes them through untouched (full
    /// length, with the result masked afterwards), while filtering hands over filtered copies,
    /// which are canonical and so match no encoding fast path.
    ///
    /// A non-nullable operand therefore reaches an encoding fast path under either. Note also that
    /// filtering a constant yields a constant, so a fast path keyed on
    /// [`as_constant`](ArrayRef::as_constant) still fires even for a filtered batch.
    fn reduce_encoded(
        &self,
        options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (options, args, ctx);
        Ok(None)
    }
}

/// One use of a [`RowFn`] at concrete element types.
///
/// The framework hands a visitor to [`RowFn::dispatch`], which calls one of the visit methods with
/// the element types it chose: at plan time the visit validates dtypes, at run time it executes the row
/// loop. Only the framework implements this trait, and a function only ever _calls_ a visit.
///
/// The function names one output sink and one preparation step. Passing `|_| ()` is the no-prepare
/// case.
pub trait RowVisitor: private::Sealed {
    /// What this visit produces.
    type Out;

    /// Visit at argument tuple `A`, preparing shared state once and writing every output row into
    /// sink `S`.
    ///
    /// `prepare` receives [`A::ConstElems`](ElementTuple::ConstElems): the element value of every
    /// argument whose operand is constant for the batch, and `None` for each one that varies by
    /// row. Whatever it returns is handed to every `apply` call by shared reference.
    ///
    /// `A` **must** have the arity declared by [`RowFn::ARG_NAMES`]. A fallible element or result
    /// also requires [`RowFn::FALLIBLE`] to be `true`; the reverse is not required. A deferred result
    /// must be paired with a sink whose [`OutputSink::ERRORS_ARE_DEFERRED`] is `true`.
    fn visit_prepared_into<A: ElementTuple, S: OutputSink, P, R: SinkResult>(
        self,
        prepare: impl FnOnce(A::ConstElems<'_>) -> P,
        apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
    ) -> VortexResult<Self::Out>;
}

pub(super) mod private {
    pub trait Sealed {}
}
