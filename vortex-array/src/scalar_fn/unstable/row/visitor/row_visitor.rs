// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The typed dispatch interface implemented by [`RowFn`] planning and execution.
//!
//! [`RowVisitor`] lets a function select its concrete input and output capabilities without
//! exposing framework-specific planning or execution state.
//!
//! [`RowFn`]: crate::scalar_fn::unstable::row::RowFn

use vortex_error::VortexResult;

use crate::scalar_fn::unstable::row::ElementTuple;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::IndexedElementTuple;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::SinkResult;

/// A planning or execution visit at concrete input and output types.
///
/// Only the framework implements this trait. The `visit_prepared*` methods derive shared state
/// from constant arguments before visiting any rows. Every visit verifies that the argument tuple
/// matches [`RowFn::ARG_NAMES`] and that fallible decoding agrees with [`RowFn::FALLIBLE`].
///
/// [`RowFn::ARG_NAMES`]: crate::scalar_fn::unstable::row::RowFn::ARG_NAMES
/// [`RowFn::FALLIBLE`]: crate::scalar_fn::unstable::row::RowFn::FALLIBLE
pub trait RowVisitor<Options>: private::Sealed + Sized {
    /// The framework result of visiting one concrete row signature.
    ///
    /// This is a batch plan or execution result, not a per-row output.
    type VisitResult;

    /// Visit an infallible row computation that returns one output value per row.
    ///
    /// `apply` must not panic or have side effects. Dense execution can pass unspecified values
    /// from null rows.
    ///
    /// The framework verifies that `Out` does not require drop glue.
    ///
    /// # Examples
    ///
    /// Apply infallible wrapping arithmetic.
    ///
    /// ```ignore
    /// visitor.visit::<(i64, i64), i64>(|(lhs, rhs)| lhs.wrapping_add(rhs))
    /// ```
    ///
    /// Dispatch an equality helper over its primitive element type.
    ///
    /// ```ignore
    /// fn visit_equal<T, Options, V>(visitor: V) -> VortexResult<V::VisitResult>
    /// where
    ///     T: NativePType,
    ///     V: RowVisitor<Options>,
    /// {
    ///     visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_eq(rhs))
    /// }
    /// ```
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
    ///
    /// # Examples
    ///
    /// Test whether each string occurs in its allowed-values list. The prepare closure builds one
    /// lookup table for a batch-constant list. The row closure scans the current list from the input
    /// column directly.
    ///
    /// ```ignore
    /// visitor.visit_prepared::<
    ///     (StringRow, StringListRow),
    ///     bool,
    ///     Option<PreparedAllowedValues>,
    /// >(
    ///     |(_value, allowed_values)| allowed_values.map(PreparedAllowedValues::new),
    ///     |prepared_allowed_values, (value, allowed_values)| {
    ///         match prepared_allowed_values {
    ///             Some(allowed_values) => allowed_values.contains(value),
    ///             None => allowed_values.iter().any(|allowed| allowed == value),
    ///         }
    ///     },
    /// )
    /// ```
    fn visit_prepared<Args, Out, Prepared>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> Out,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement;

    /// Visit a row computation that writes through a row handle from an output sink.
    ///
    /// `apply` must not panic or have side effects except for writes to the supplied row handle.
    /// Dense execution can pass unspecified values from null rows.
    ///
    /// On success, `apply` must return the write token for the supplied row handle. A token from
    /// another row, sink, or local cell can violate the safety contract of [`OutputSink::finish`].
    ///
    /// A fallible `ApplyResult` requires
    /// [`RowFn::FALLIBLE`](crate::scalar_fn::unstable::row::RowFn::FALLIBLE) to be `true`.
    ///
    /// # Examples
    ///
    /// Checked integer division reports errors immediately and writes successful rows into
    /// uninitialized output. The cold, non-inlined helper keeps error construction out of the row
    /// callback.
    ///
    /// ```ignore
    /// #[cold]
    /// #[inline(never)]
    /// fn integer_division_error() -> VortexError {
    ///     vortex_err!(InvalidArgument: "integer division by zero or overflow")
    /// }
    ///
    /// visitor.visit_into::<(i64, i64), UninitElementSink<i64>, _>(
    ///     |(lhs, rhs), output| {
    ///         let Some(value) = lhs.checked_div(rhs) else {
    ///             return Err(integer_division_error());
    ///         };
    ///
    ///         // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
    ///         Ok(unsafe { InitializedElement::write(output, value) })
    ///     },
    /// )
    /// ```
    fn visit_into<Args, Sink, ApplyResult>(
        self,
        apply: impl Fn(Args::Elems<'_>, <Sink as OutputSink<Options>>::Row<'_>) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink<Options>,
        ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>,
    {
        self.visit_prepared_into::<Args, Sink, (), ApplyResult>(
            |_| (),
            move |&(), args, row| apply(args, row),
        )
    }

    /// The prepared form of [`visit_into`](Self::visit_into), with the same prerequisites.
    ///
    /// # Examples
    ///
    /// Compute the cosine similarity of each vector pair: their dot product divided by their
    /// magnitudes. The prepare closure computes each batch-constant vector's magnitude once.
    ///
    /// ```ignore
    /// visitor.visit_prepared_into::<
    ///     (TensorRow<T>, TensorRow<T>),
    ///     UninitElementSink<T>,
    ///     ConstVectorMagnitudes<T>,
    ///     InitializedElement,
    /// >(
    ///     |(lhs, rhs)| ConstVectorMagnitudes {
    ///         lhs: lhs.map(vector_magnitude),
    ///         rhs: rhs.map(vector_magnitude),
    ///     },
    ///     |const_magnitudes, (lhs, rhs), output| {
    ///         let similarity =
    ///             cosine_similarity_with_const_magnitudes(const_magnitudes, lhs, rhs);
    ///
    ///         // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
    ///         unsafe { InitializedElement::write(output, similarity) }
    ///     },
    /// )
    /// ```
    fn visit_prepared_into<Args, Sink, Prepared, ApplyResult>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(
            &Prepared,
            Args::Elems<'_>,
            <Sink as OutputSink<Options>>::Row<'_>,
        ) -> ApplyResult,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: ElementTuple,
        Sink: OutputSink<Options>,
        ApplyResult: SinkResult<WriteToken = <Sink as OutputSink<Options>>::WriteToken>;

    /// Visit a row computation that returns an owned output value and deferred failure evidence.
    ///
    /// `apply` must not panic or have side effects. Dense execution can pass unspecified values
    /// from null rows.
    ///
    /// The executor OR-reduces [`FailureEvidence`] across rows and passes the result to
    /// `finish_failure`.
    ///
    /// [`RowFn::FALLIBLE`](crate::scalar_fn::unstable::row::RowFn::FALLIBLE) **must** be `true`.
    /// `Out` must not require drop glue. `Fail` must be no wider than `Out`, or failure tracking
    /// reduces the vector width. The framework checks these requirements.
    ///
    /// # Examples
    ///
    /// Checked addition returns a wrapping value and compact overflow flag without branching. The
    /// executor reduces the flags after the loop. The cold, non-inlined helper keeps error
    /// construction out of the row loop.
    ///
    /// ```ignore
    /// #[cold]
    /// #[inline(never)]
    /// fn integer_addition_error() -> VortexError {
    ///     vortex_err!(InvalidArgument: "integer overflow in checked add")
    /// }
    ///
    /// visitor.visit_deferred::<(i64, i64), i64, bool>(
    ///     // `overflowing_add` returns `(i64, bool)`.
    ///     |(lhs, rhs)| lhs.overflowing_add(rhs),
    ///     |overflowed| {
    ///         if overflowed {
    ///             return Err(integer_addition_error());
    ///         }
    ///
    ///         Ok(())
    ///     },
    /// )
    /// ```
    fn visit_deferred<Args, Out, Fail>(
        self,
        apply: impl Fn(Args::Elems<'_>) -> (Out, Fail),
        finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
        Fail: FailureEvidence,
    {
        self.visit_prepared_deferred::<Args, Out, (), Fail>(
            |_| (),
            move |&(), args| apply(args),
            finish_failure,
        )
    }

    /// The prepared form of [`visit_deferred`](Self::visit_deferred), with the same prerequisites.
    ///
    /// # Examples
    ///
    /// Rescale each unscaled decimal by multiplying it by `10^scale`. The prepare closure computes
    /// the multiplier once for a batch-constant scale. Each row returns the rescaled value and an
    /// overflow flag, which the executor reduces after the loop.
    ///
    /// ```ignore
    /// #[cold]
    /// #[inline(never)]
    /// fn decimal_rescaling_overflow() -> VortexError {
    ///     vortex_err!(InvalidArgument: "decimal rescaling overflowed")
    /// }
    ///
    /// visitor.visit_prepared_deferred::<
    ///     (i64, DecimalScale),
    ///     i64,
    ///     Option<PreparedDecimalScale>,
    ///     bool,
    /// >(
    ///     |(_value, scale)| scale.map(PreparedDecimalScale::new),
    ///     |prepared_scale, (value, scale)| match prepared_scale {
    ///         Some(scale) => scale.apply_checked(value),
    ///         None => PreparedDecimalScale::new(scale).apply_checked(value),
    ///     },
    ///     |overflowed| {
    ///         if overflowed {
    ///             return Err(decimal_rescaling_overflow());
    ///         }
    ///
    ///         Ok(())
    ///     },
    /// )
    /// ```
    fn visit_prepared_deferred<Args, Out, Prepared, Fail>(
        self,
        prepare: impl FnOnce(Args::ConstElems<'_>) -> Prepared,
        apply: impl Fn(&Prepared, Args::Elems<'_>) -> (Out, Fail),
        finish_failure: impl FnOnce(Fail) -> VortexResult<()>,
    ) -> VortexResult<Self::VisitResult>
    where
        Args: IndexedElementTuple,
        Out: OutputElement,
        Fail: FailureEvidence;
}

pub(super) mod private {
    pub trait Sealed {}
}
