// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Visits that plan or execute the concrete row signature selected by [`RowFn::dispatch`].
//!
//! [`RowFn::dispatch`]: crate::scalar_fn::RowFn::dispatch

use std::ops::BitOrAssign;

use vortex_error::VortexResult;

use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::IndexedElementTuple;
use crate::scalar_fn::OutputElement;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::SinkResult;

mod check;
pub(super) use check::assert_owned_output_needs_no_drop;

mod execute;
pub(super) use execute::ExecuteRows;
pub(super) use execute::ExecuteValidRows;

mod plan;
pub(super) use plan::PlanRows;

/// A planning or execution visit at concrete input and output types.
///
/// Only the framework implements this trait. The `visit_prepared*` methods derive shared state
/// from constant arguments before visiting any rows.
pub trait RowVisitor: private::Sealed + Sized {
    /// The framework result of visiting one concrete row signature.
    ///
    /// This is a batch plan or execution result, not the per-row `Out` returned by
    /// [`RowVisitor::visit`] and [`RowVisitor::visit_deferred`].
    type VisitResult;

    /// Visit an infallible row computation that returns one independent output value.
    ///
    /// `apply` must be total over every stored element value: it must not panic or have side
    /// effects. Dense execution can pass unspecified values from null rows.
    ///
    /// # Prerequisites
    ///
    /// The framework checks these at compile time:
    ///
    /// - `Args::ARITY` **must** equal the length of
    ///   [`RowFn::ARG_NAMES`](crate::scalar_fn::RowFn::ARG_NAMES).
    /// - [`RowFn::FALLIBLE`](crate::scalar_fn::RowFn::FALLIBLE) **must** be `true` when decoding
    ///   `Args` can fail.
    /// - `Out` **must not** require drop glue.
    fn visit<Args, Out>(
        self,
        apply: impl Fn(Args::Elems<'_>) -> Out,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
    {
        self.visit_prepared::<Args, Out, ()>(|_| (), move |&(), args| apply(args))
    }

    /// The prepared form of [`visit`](Self::visit), with the same prerequisites.
    fn visit_prepared<Args, Out, Prepared>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> Out,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement;

    /// Visit a row computation that writes through a sink-provided row handle.
    ///
    /// `apply` must be total over every stored input value: it must not panic or cause side effects
    /// other than writing the supplied row handle. Dense execution can pass unspecified values
    /// from null rows.
    ///
    /// # Prerequisites
    ///
    /// The framework checks these at compile time:
    ///
    /// - `Args::ARITY` **must** equal the length of
    ///   [`RowFn::ARG_NAMES`](crate::scalar_fn::RowFn::ARG_NAMES).
    /// - [`RowFn::FALLIBLE`](crate::scalar_fn::RowFn::FALLIBLE) **must** be `true` when decoding
    ///   `Args` or computing the result can fail.
    fn visit_into<Args, Sink, ApplyResult>(
        self,
        apply: impl Fn(Args::Elems<'_>, Sink::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink,
        ApplyResult: SinkResult<WriteToken = Sink::WriteToken>,
    {
        self.visit_prepared_into::<Args, Sink, (), ApplyResult>(
            |_| (),
            move |&(), args, row| apply(args, row),
        )
    }

    /// The prepared form of [`visit_into`](Self::visit_into), with the same prerequisites.
    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>, Sink::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink,
        ApplyResult: SinkResult<WriteToken = Sink::WriteToken>;

    /// Visit a row computation that returns an owned output and deferred failure evidence.
    ///
    /// `apply` must be total over every stored element value: it must not panic or have side
    /// effects. Dense execution can pass unspecified values from null rows.
    ///
    /// `Fail` is OR-reduced across rows and handed to `finish_failure`. The value from
    /// [`Default::default`] **must** mean success, including for an empty batch. The compiler
    /// cannot check this semantic requirement.
    ///
    /// # Prerequisites
    ///
    /// The framework checks these at compile time:
    ///
    /// - `Args::ARITY` **must** equal the length of
    ///   [`RowFn::ARG_NAMES`](crate::scalar_fn::RowFn::ARG_NAMES).
    /// - [`RowFn::FALLIBLE`](crate::scalar_fn::RowFn::FALLIBLE) **must** be `true`.
    /// - `Out` **must not** require drop glue.
    /// - `Out` **must** be at least as wide as `Fail` so failure tracking does not reduce the
    ///   vector width.
    fn visit_deferred<Args, Out, Fail>(
        self,
        apply: impl Fn(Args::Elems<'_>) -> (Out, Fail),
        finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
        Fail: Copy + Default + BitOrAssign,
    {
        self.visit_prepared_deferred::<Args, Out, (), Fail>(
            |_| (),
            move |&(), args| apply(args),
            finish_failure,
        )
    }

    /// The prepared form of [`visit_deferred`](Self::visit_deferred), with the same prerequisites.
    fn visit_prepared_deferred<Args, Out, Prepared, Fail>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
        finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
        Fail: Copy + Default + BitOrAssign;
}

mod private {
    pub trait Sealed {}
}
