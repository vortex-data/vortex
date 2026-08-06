// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Equivalence and selection tests for filtering and branch-and-skip null execution.

use std::sync::Arc;

use vortex_buffer::ByteBuffer;

use super::*;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::Nullability;
use crate::validity::Validity;

/// Executes `scalar_fn` over `args` with `strategy` forced, canonicalized like [`apply`].
fn apply_forced<F: RowFn<Options = EmptyOptions>>(
    scalar_fn: &F,
    args: &[ArrayRef],
    strategy: NullStrategy,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let rows = args.first().map_or(0, |arg| arg.len());

    Ok(
        execute_row_fn_with_strategy(scalar_fn, &EmptyOptions, args.to_vec(), rows, strategy, ctx)?
            .execute::<Canonical>(ctx)?
            .into_array(),
    )
}

/// Runs `scalar_fn` under forced filter, forced branch-and-skip, and the automatic per-batch
/// selection, and asserts all three produce identical arrays.
fn assert_strategies_agree<F: RowFn<Options = EmptyOptions>>(
    scalar_fn: F,
    args: Vec<ArrayRef>,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    let filtered = apply_forced(&scalar_fn, &args, NullStrategy::Filter, &mut ctx)?;
    let branched = apply_forced(&scalar_fn, &args, NullStrategy::BranchAndSkip, &mut ctx)?;
    let auto = apply(scalar_fn, args, &mut ctx)?;

    assert_arrays_eq!(branched, filtered, &mut ctx);
    assert_arrays_eq!(auto, filtered, &mut ctx);
    Ok(())
}

/// A `Utf8` column whose null rows carry views naming a buffer that does not exist, at
/// offsets far out of bounds. Resolving such a row's bytes panics, so strategy agreement
/// proves the branch loop never calls `get` behind a null.
fn hostile_nullable_strings() -> VortexResult<ArrayRef> {
    let views = buffer![
        BinaryView::make_view(b"a longer string here", 0, 0),
        BinaryView::new_ref(64, *b"junk", 9, 4096),
        BinaryView::make_view(b"another non-inlined string", 1, 0),
        BinaryView::new_ref(64, *b"junk", 7, 1 << 20),
    ];

    Ok(VarBinViewArray::try_new(
        views,
        Arc::from([
            ByteBuffer::copy_from(b"a longer string here"),
            ByteBuffer::copy_from(b"another non-inlined string"),
        ]),
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false, true, false]),
    )?
    .into_array())
}

/// `Bytes` is not dense-safe, so `Shout` uses valid-only execution; both strategies must produce
/// the same array without resolving the hostile views behind the nulls.
#[test]
fn branch_matches_filter_for_bytes() -> VortexResult<()> {
    assert_strategies_agree(Shout, vec![hostile_nullable_strings()?])
}

/// A fallible kernel with a poison value (zero divisor) behind every null: the branch loop
/// must skip those rows rather than spuriously failing on them.
#[test]
fn branch_never_applies_a_fallible_kernel_behind_nulls() -> VortexResult<()> {
    let lhs = buffer![10i64, 10, 12, 9].into_array();
    let rhs = PrimitiveArray::new(
        buffer![2i64, 0, 3, 0],
        Validity::from_iter([true, false, true, false]),
    )
    .into_array();

    assert_strategies_agree(CheckedDiv, vec![lhs, rhs])
}

/// Nulls in both operands: the branch loop must honor the _conjoined_ mask, not either
/// input's own validity.
#[test]
fn branch_conjoins_validities() -> VortexResult<()> {
    let lhs =
        PrimitiveArray::from_option_iter([Some(10i64), None, Some(12), Some(9), None]).into_array();
    let rhs = PrimitiveArray::new(
        buffer![2i64, 0, 3, 0, 0],
        Validity::from_iter([true, true, true, false, false]),
    )
    .into_array();

    assert_strategies_agree(CheckedDiv, vec![lhs, rhs])
}

/// A constant operand under the branch strategy still hoists through the stride-0 decode.
#[test]
fn branch_handles_constant_operands() -> VortexResult<()> {
    let lhs = PrimitiveArray::from_option_iter([Some(10i64), None, Some(12)]).into_array();
    let rhs = ConstantArray::new(Scalar::from(2i64), 3).into_array();

    assert_strategies_agree(CheckedDiv, vec![lhs, rhs])
}

/// An error from a _valid_ row still propagates under the branch strategy.
#[test]
fn branch_propagates_real_errors() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10, 12].into_array();
    let rhs = PrimitiveArray::new(
        buffer![2i64, 3, 0],
        Validity::from_iter([true, false, true]),
    )
    .into_array();

    let error = apply_forced(
        &CheckedDiv,
        &[lhs, rhs],
        NullStrategy::BranchAndSkip,
        &mut ctx,
    )
    .expect_err("a zero divisor in a valid row must fail");

    assert!(
        error.to_string().contains("division by zero"),
        "unexpected error: {error}"
    );
}

/// The automatic per-batch selection, observed through elements that record which decode ran
/// on how many rows: the branch strategy decodes null-tolerantly at full length, the filter
/// strategy decodes ordinarily over the survivors.
mod selection {
    use std::cell::Cell;
    use std::cell::RefCell;

    use vortex_buffer::Buffer;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::*;
    use crate::scalar_fn::row::lift::branch_beats_filter;

    thread_local! {
        /// What the last varying-column decode did: `(null_tolerant, rows)`. Thread-local so
        /// concurrent tests in one process cannot race it; execution runs on the calling
        /// thread.
        static LAST_DECODE: Cell<Option<(bool, usize)>> = const { Cell::new(None) };
    }

    /// An i64 element that records its decodes and reports `COST` units of filtered decode work.
    /// It is not dense-safe, so strategy selection actually happens.
    struct TrackedI64<const COST: usize>;

    impl<const COST: usize> InputElement for TrackedI64<COST> {
        type Column = Buffer<i64>;
        type Varying<'a> = <i64 as InputElement>::Varying<'a>;
        type Elem<'a> = i64;

        const DENSE_SAFE: bool = false;
        const DECODE_FALLIBLE: bool = false;
        const FILTERED_DECODE_COST: usize = COST;

        fn validate(dtype: &DType) -> VortexResult<()> {
            <i64 as InputElement>::validate(dtype)
        }

        fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
            LAST_DECODE.set(Some((false, array.len())));
            <i64 as InputElement>::decode(array, ctx)
        }

        fn decode_null_tolerant(
            array: ArrayRef,
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<Self::Column>> {
            LAST_DECODE.set(Some((true, array.len())));
            <i64 as InputElement>::decode(array, ctx).map(Some)
        }

        fn get(column: &Self::Column, index: usize) -> i64 {
            <i64 as InputElement>::get(column, index)
        }

        fn varying(column: &Self::Column) -> Self::Varying<'_> {
            <i64 as InputElement>::varying(column)
        }

        fn varying_len(column: &Self::Varying<'_>) -> usize {
            <i64 as InputElement>::varying_len(column)
        }

        fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
        where
            Self: 'a,
        {
            <i64 as InputElement>::get_varying(column, index)
        }
    }

    /// Negation over one tracked column.
    #[derive(Clone)]
    struct TrackedNegate<const COST: usize>;

    impl<const COST: usize> RowFn for TrackedNegate<COST> {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &["input"];

        fn id(&self) -> ScalarFnId {
            if COST == 0 {
                static ID: CachedId = CachedId::new("vortex.test.tracked_negate.bulk");
                *ID
            } else {
                static ID: CachedId = CachedId::new("vortex.test.tracked_negate.per_row");
                *ID
            }
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_prepared_into::<(TrackedI64<COST>,), ElementSink<i64>, _, _>(
                |_| (),
                |&(), (value,), output| *output = -value,
            )
        }
    }

    /// A 32-row nullable i64 column whose first `valid_count` rows are valid.
    fn column_with_survivors(valid_count: usize) -> ArrayRef {
        PrimitiveArray::from_option_iter(
            (0..32u16).map(|i| (usize::from(i) < valid_count).then_some(i64::from(i))),
        )
        .into_array()
    }

    /// Executes the tracked function through the full pipeline and returns what the decode
    /// recorded: whether it was null-tolerant, and how many rows it saw.
    fn run<const COST: usize>(valid_count: usize) -> VortexResult<(bool, usize)> {
        let mut ctx = array_session().create_execution_ctx();
        LAST_DECODE.set(None);

        apply(
            TrackedNegate::<COST>,
            [column_with_survivors(valid_count)],
            &mut ctx,
        )?;

        LAST_DECODE
            .get()
            .ok_or_else(|| vortex_err!("no decode ran"))
    }

    /// A bulk-decoded element takes branch-and-skip on a mixed mask however sparse the
    /// survivors: the decode is null-tolerant and full length.
    #[test]
    fn bulk_decode_branches_at_any_density() -> VortexResult<()> {
        assert_eq!(run::<0>(31)?, (true, 32));
        assert_eq!(run::<0>(4)?, (true, 32));
        Ok(())
    }

    /// One per-row decode still branches when half the rows survive, matching the measured
    /// single-nullable-input crossover.
    #[test]
    fn per_row_decode_filters_when_sparse() -> VortexResult<()> {
        // 30/32 surviving: branch, full-length null-tolerant decode.
        assert_eq!(run::<1>(30)?, (true, 32));
        // 16/32 = 50% surviving sits exactly on the threshold: still branch.
        assert_eq!(run::<1>(16)?, (true, 32));
        // Below 50% surviving: filter, ordinary decode over the survivors.
        assert_eq!(run::<1>(15)?, (false, 15));
        Ok(())
    }

    /// An all-true mask short-circuits to the plain kernel and an all-false mask to an
    /// all-null constant, before any strategy is selected.
    #[test]
    fn degenerate_masks_bypass_the_selection() -> VortexResult<()> {
        assert_eq!(run::<1>(32)?, (false, 32));

        let mut ctx = array_session().create_execution_ctx();
        LAST_DECODE.set(None);
        apply(TrackedNegate::<1>, [column_with_survivors(0)], &mut ctx)?;
        assert_eq!(LAST_DECODE.get(), None);
        Ok(())
    }

    /// An i64 element that omits `decode_null_tolerant`: the conservative default refuses, so
    /// the batch must fall back to the filter strategy even though the selection preferred
    /// branch.
    struct RefusesNullTolerant;

    impl InputElement for RefusesNullTolerant {
        type Column = Buffer<i64>;
        type Varying<'a> = <i64 as InputElement>::Varying<'a>;
        type Elem<'a> = i64;

        const DENSE_SAFE: bool = false;
        const DECODE_FALLIBLE: bool = false;

        fn validate(dtype: &DType) -> VortexResult<()> {
            <i64 as InputElement>::validate(dtype)
        }

        fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
            LAST_DECODE.set(Some((false, array.len())));
            <i64 as InputElement>::decode(array, ctx)
        }

        fn get(column: &Self::Column, index: usize) -> i64 {
            <i64 as InputElement>::get(column, index)
        }

        fn varying(column: &Self::Column) -> Self::Varying<'_> {
            <i64 as InputElement>::varying(column)
        }

        fn varying_len(column: &Self::Varying<'_>) -> usize {
            <i64 as InputElement>::varying_len(column)
        }

        fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
        where
            Self: 'a,
        {
            <i64 as InputElement>::get_varying(column, index)
        }
    }

    #[derive(Clone)]
    struct RefusingNegate;

    impl RowFn for RefusingNegate {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &["input"];

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.refusing_negate");
            *ID
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_prepared_into::<(RefusesNullTolerant,), ElementSink<i64>, _, _>(
                |_| (),
                |&(), (value,), output| *output = -value,
            )
        }
    }

    /// The fallback is silent and correct: the ordinary decode runs over the survivors and
    /// the result matches the expected negation.
    #[test]
    fn missing_null_tolerant_decode_falls_back_to_filter() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        LAST_DECODE.set(None);

        let result = apply(
            RefusingNegate,
            [PrimitiveArray::from_option_iter([Some(3i64), None, Some(5)]).into_array()],
            &mut ctx,
        )?;

        assert_eq!(LAST_DECODE.get(), Some((false, 2)));
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(-3i64), None, Some(-5)]),
            &mut ctx
        );
        Ok(())
    }

    thread_local! {
        /// Every `row_count` `reduce_encoded` was handed, in call order.
        static REDUCE_ROW_COUNTS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    /// [`RefusingNegate`] with an encoding-aware rewrite that declines, recording the row count it
    /// was offered.
    #[derive(Clone)]
    struct ProbingNegate;

    impl RowFn for ProbingNegate {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &["input"];

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("vortex.test.probing_negate");
            *ID
        }

        fn dispatch<V: RowVisitor>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::Out> {
            visitor.visit_prepared_into::<(RefusesNullTolerant,), ElementSink<i64>, _, _>(
                |_| (),
                |&(), (value,), output| *output = -value,
            )
        }

        fn reduce_encoded(
            &self,
            _options: &Self::Options,
            args: &[ArrayRef],
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>> {
            REDUCE_ROW_COUNTS.with_borrow_mut(|counts| counts.push(args[0].len()));
            Ok(None)
        }
    }

    /// A `reduce_encoded` rewrite must be sized from the arrays it was handed, which under the
    /// filter strategy hold the surviving rows rather than the whole batch. This is the only
    /// mixed-mask path where the two differ, so nothing else would catch a rewrite sized from a
    /// length captured elsewhere.
    ///
    /// This also pins the double probe as deliberate. The first call sees the original arrays at
    /// full length, and is the only one that does; the second sees filtered, canonical copies. An
    /// "optimization" that skipped the first because the batch will end up filtering would take an
    /// encoding-aware rewrite away from every function whose sink cannot skip rows.
    #[test]
    fn reduce_encoded_is_probed_before_and_after_filtering() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        REDUCE_ROW_COUNTS.with_borrow_mut(Vec::clear);

        let result = apply(
            ProbingNegate,
            [PrimitiveArray::from_option_iter([Some(3i64), None, Some(5)]).into_array()],
            &mut ctx,
        )?;

        assert_eq!(
            REDUCE_ROW_COUNTS.with_borrow(|counts| counts.clone()),
            vec![3, 2],
            "expected an unfiltered probe at the batch length, then a filtered one at the \
             surviving count",
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(-3i64), None, Some(-5)]),
            &mut ctx
        );
        Ok(())
    }

    /// The rule itself, at and around the threshold, without going through an execution.
    #[rstest]
    #[case::bulk_dense_mask(0, 99, 100, true)]
    #[case::bulk_sparse_mask(0, 1, 100, true)]
    #[case::one_decode_dense_mask(1, 99, 100, true)]
    #[case::one_decode_at_threshold(1, 50, 100, true)]
    #[case::one_decode_below_threshold(1, 49, 100, false)]
    #[case::two_decodes_at_old_boolean_choice(2, 81, 100, false)]
    #[case::two_decodes_dense_mask(2, 90, 100, true)]
    fn selects_branch_per_the_measured_rule(
        #[case] filtered_decode_cost: usize,
        #[case] true_count: usize,
        #[case] len: usize,
        #[case] expect_branch: bool,
    ) {
        let valid = Mask::from_indices(len, 0..true_count);
        assert_eq!(
            branch_beats_filter(filtered_decode_cost, &valid),
            expect_branch,
        );
    }

    #[test]
    fn planning_adds_decode_cost_across_arguments() {
        assert_eq!(
            RowPolicy::for_dispatch::<(TrackedI64<1>, TrackedI64<1>), ()>(),
            RowPolicy::ValidOnly {
                filtered_decode_cost: 2
            }
        );
    }
}
