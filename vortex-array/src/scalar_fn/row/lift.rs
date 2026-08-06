// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lifting a kernel over non-null values into a full [`ScalarFnVTable::execute`].
//!
//! A [`RowFn`] hands the framework a kernel that only ever computes rows valid in every argument.
//! Everything between that kernel and [`ScalarFnVTable::execute`] lives here: null propagation,
//! constant folding, nullability widening, output dtype reconciliation, and the per-batch choice
//! between dense execution and the two mechanisms that execute only valid rows.
//!
//! This is machinery, not an interface. It takes the kernel as a pair of closures rather than a
//! trait because the one trait that ever occupied the slot (a public `StrictScalarFnVTable`, with
//! [`RowFn`] blanket-implementing it) never found a second implementor, and the indirection cost
//! more than it explained. Extract a trait if and when a non-row user appears.
//!
//! [`RowFn`]: crate::scalar_fn::RowFn
//! [`ScalarFnVTable::execute`]: crate::scalar_fn::ScalarFnVTable::execute

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::SinkResult;
use crate::scalar_fn::row::element::batch_constant;
use crate::scalar_fn::row::execute::RowExecution;
use crate::validity::Validity;

struct BorrowedExecutionArgs<'a> {
    inputs: &'a [ArrayRef],
    row_count: usize,
}

impl<'a> BorrowedExecutionArgs<'a> {
    fn new(inputs: &'a [ArrayRef], row_count: usize) -> Self {
        Self { inputs, row_count }
    }
}

impl ExecutionArgs for BorrowedExecutionArgs<'_> {
    fn get(&self, index: usize) -> VortexResult<ArrayRef> {
        self.inputs.get(index).cloned().ok_or_else(|| {
            vortex_error::vortex_err!(
                "Input index {} out of bounds (num_inputs={})",
                index,
                self.inputs.len()
            )
        })
    }

    fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }
}

/// The arguments handed to one kernel invocation.
///
/// `arrays` may be filtered or sliced, while `dtypes` and `sink_dtype` always describe the original
/// planned batch. Keeping them together prevents an execution path from accidentally pairing an
/// input view with unrelated planning metadata.
#[derive(Clone, Copy)]
pub(super) struct KernelArgs<'a> {
    /// The executor-facing view, including the row count for this invocation.
    pub(super) execution: &'a dyn ExecutionArgs,

    /// The same inputs as concrete arrays for encoding-aware rewrites.
    pub(super) arrays: &'a [ArrayRef],

    /// The original input dtypes used to select the row implementation.
    pub(super) dtypes: &'a [DType],

    /// The non-nullable dtype allocated by the selected output sink.
    pub(super) sink_dtype: &'a DType,
}

/// The execution policy and output dtype selected by a planning visit.
pub(super) struct BatchPlan {
    /// The non-nullable dtype built by the selected sink.
    pub(super) sink_dtype: DType,

    /// How this concrete dispatch executes nullable rows.
    pub(super) policy: RowPolicy,
}

/// The nullable execution policy derived from one concrete dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowPolicy {
    /// Evaluate all rows and mask the result.
    Dense,

    /// Evaluate all rows, retrying only valid rows if a deferred error is raised.
    DenseWithRetry,

    /// Execute only valid rows, choosing branch-and-skip or filtering from the mask and decode
    /// cost.
    ValidOnly { filtered_decode_cost: usize },
}

impl RowPolicy {
    /// The policy one concrete dispatch executes nullable rows under.
    ///
    /// Note what is deliberately **not** read here: [`OutputSink::SUPPORTS_SKIPPED_ROWS`]. Hoisting
    /// it into the plan so that a non-skipping sink never enters the branch path looks like a free
    /// win, and #9130 records it as one, but it is not: the branch path probes
    /// [`reduce_encoded`](crate::scalar_fn::RowFn::reduce_encoded) against the _original_ arrays
    /// before it ever consults the sink, and that is the only probe that sees them still encoded.
    /// Skipping the path early would leave such a function with only the filtered probe, whose
    /// canonical arrays match no encoding fast path. For a function whose reduction is defined to
    /// answer differently from its row loop, that is a wrong answer rather than a slow one.
    ///
    /// [`OutputSink::SUPPORTS_SKIPPED_ROWS`]: crate::scalar_fn::OutputSink::SUPPORTS_SKIPPED_ROWS
    pub(super) const fn for_dispatch<A: ElementTuple, R: SinkResult>() -> Self {
        if A::DENSE_SAFE && !A::DECODE_FALLIBLE && !R::FALLIBLE {
            if R::DEFERRED {
                Self::DenseWithRetry
            } else {
                Self::Dense
            }
        } else {
            Self::ValidOnly {
                filtered_decode_cost: A::FILTERED_DECODE_COST,
            }
        }
    }
}

/// How far [`Batch::resolve_validity`] got before a mixed-mask strategy became necessary.
enum ResolvedMask {
    /// The batch was answered without one: every row valid, or every row null.
    Decided(ArrayRef),

    /// A mask with both set and unset bits, which a strategy must now execute.
    Mixed(Mask),
}

/// One batch of inputs, with everything the lifting reads off them before the kernel runs.
pub(super) struct Batch<'a> {
    /// The function being executed, named in the errors this raises.
    id: ScalarFnId,

    /// The arguments as the execution layer handed them over. Every path but the filter strategy
    /// gives the kernel these untouched, so it sees the original encodings.
    args: &'a dyn ExecutionArgs,

    /// The input columns, collected once: constant folding inspects them and the filter strategy
    /// filters them.
    inputs: SmallVec<[ArrayRef; 4]>,

    /// The input dtypes, collected with the columns and reused by both planning and execution.
    arg_dtypes: SmallVec<[DType; 4]>,

    /// The conjoined input validity, so a row of the output is valid iff it is valid in every
    /// input. Conjoining is lazy, and nothing materializes it unless the null handling asks.
    validity: Validity,

    /// The dtype the function declares for these inputs, which the kernel's output is reconciled
    /// against. Already widened to nullable if any input is nullable.
    result_dtype: DType,

    /// The non-nullable dtype the dispatched sink builds, computed once while planning.
    sink_dtype: DType,

    /// How the concrete dispatch executes nullable rows.
    policy: RowPolicy,
}

impl<'a> Batch<'a> {
    /// Collect `args` and read the lifting's facts off them, `return_dtype` being the function's
    /// declared return dtype for the input dtypes it is handed.
    ///
    /// **Not** for a nullary function: with no inputs there is no validity to propagate and no
    /// per-row work to fold, and the all-constant check below would vacuously pass.
    pub(super) fn new(
        id: ScalarFnId,
        args: &'a dyn ExecutionArgs,
        plan: impl FnOnce(&[DType]) -> VortexResult<BatchPlan>,
    ) -> VortexResult<Self> {
        let inputs: SmallVec<[ArrayRef; 4]> = (0..args.num_inputs())
            .map(|i| args.get(i))
            .collect::<VortexResult<_>>()?;

        let arg_dtypes: SmallVec<[DType; 4]> =
            inputs.iter().map(|input| input.dtype().clone()).collect();
        let plan = plan(&arg_dtypes)?;
        let nullability = plan.sink_dtype.nullability()
            | Nullability::from(arg_dtypes.iter().any(DType::is_nullable));
        let result_dtype = plan.sink_dtype.with_nullability(nullability);

        let mut validity = Validity::NonNullable;
        for input in &inputs {
            validity = validity.and(input.validity()?)?;
        }

        Ok(Self {
            id,
            args,
            inputs,
            arg_dtypes,
            validity,
            result_dtype,
            sink_dtype: plan.sink_dtype,
            policy: plan.policy,
        })
    }

    /// Run `kernel` over this batch, adding everything the kernel does not do: the null-constant
    /// short circuit, the all-constant fold, and the null handling.
    ///
    /// `kernel` computes the whole column from the arguments it is handed. Those are this batch's
    /// arguments untouched, except under the filter strategy, where they are filtered copies, and
    /// in the all-constant fold, where they are one row each. What it may assume:
    ///
    /// - No input is a null constant, and the inputs are not all constant.
    /// - Under valid-only execution, every row of every input is valid.
    /// - Under dense execution, rows behind nulls hold arbitrary values, and their results are
    ///   discarded.
    ///
    /// Either way the kernel can ignore input validity, and its output **must** equal
    /// `return_dtype` up to nullability. A kernel that returns nulls of its own keeps them, unioned
    /// with the ones the lifting applies, which requires its declared dtype to be nullable.
    ///
    /// `branch` computes only the rows set in the conjoined mask, over the _unfiltered_ arguments,
    /// writing an arbitrary placeholder everywhere else; `Ok(None)` means it cannot for these
    /// inputs, which sends the batch to the filter strategy. It is only ever called with a mixed
    /// mask, and it **must not** run its row computation (nor any per-row fallible decode) on an
    /// unset row, since those rows hold arbitrary values and a fallible kernel would spuriously
    /// fail on them.
    pub(super) fn execute(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        branch: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Strictness: any null-constant input forces an all-null result without evaluating the
        // kernel.
        if self
            .inputs
            .iter()
            .any(|input| input.as_constant().is_some_and(|scalar| scalar.is_null()))
        {
            return Ok(self.all_null());
        }

        // All inputs constant, and their conjoined validity proves every row non-null. This sees
        // through extension and masked wrappers just like argument decoding does.
        if self.args.row_count() > 0
            && self.validity.definitely_no_nulls()
            && self
                .inputs
                .iter()
                .all(|input| batch_constant(input).is_some())
        {
            return self.broadcast_one_row(kernel, ctx);
        }

        match self.policy {
            RowPolicy::Dense => self.execute_dense(kernel, false, ctx),
            RowPolicy::DenseWithRetry => self.execute_dense(kernel, true, ctx),
            RowPolicy::ValidOnly {
                filtered_decode_cost,
            } => self.execute_filtered(kernel, branch, filtered_decode_cost, ctx),
        }
    }

    /// Evaluate a single row of all-constant inputs and broadcast its value.
    ///
    /// Reconciling the row's dtype before reading the scalar keeps this path on the same
    /// kernel/declaration agreement check as the dense and filter paths, rather than letting `cast`
    /// paper over a disagreement.
    fn broadcast_one_row(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let one_row: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.slice(0..1))
            .collect::<VortexResult<_>>()?;

        let args = BorrowedExecutionArgs::new(&one_row, 1);
        let result = kernel(self.kernel_args(&args, &one_row), ctx)?.into_result()?;
        let scalar = self.with_return_dtype(result, 1)?.execute_scalar(0, ctx)?;

        Ok(ConstantArray::new(scalar, self.args.row_count()).into_array())
    }

    /// Run the kernel over every row, including the rows behind nulls, then mask its result.
    ///
    /// The arguments reach the kernel untouched, so the inputs keep their original encoding, and
    /// the conjoined validity is handed to `mask` as an array rather than materialized into a
    /// [`Mask`] first.
    fn execute_dense(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        retry_deferred_error: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Every row is null, so the kernel has nothing to contribute.
        if matches!(self.validity, Validity::AllInvalid) {
            return Ok(self.all_null());
        }

        let values = match kernel(self.kernel_args(self.args, &self.inputs), ctx)? {
            RowExecution::Output(values) => values,
            RowExecution::DeferredError(error) if retry_deferred_error => {
                let valid = self
                    .validity
                    .clone()
                    .execute_mask(self.args.row_count(), ctx)?;

                // The same shortcut pair as `resolve_validity`, with different outcomes: every
                // row valid means some valid row genuinely failed, and no row valid means every
                // failure was behind a null. An empty mask is both all-true and all-false, but
                // cannot reach this arm: a zero-row loop accumulates no evidence, so a zero-row
                // batch never reports a deferred error.
                if valid.all_true() {
                    return Err(error);
                }
                if valid.all_false() {
                    return Ok(self.all_null());
                }

                // Filtering unconditionally, rather than consulting `branch_beats_filter`. Not
                // because branch-and-skip is unavailable in principle: `ERRORS_ARE_DEFERRED` and
                // `SUPPORTS_SKIPPED_ROWS` are independent, and a sink may legally set both. It is
                // that `execute_dense` is not handed the `branch` closure at all, so filtering is
                // the only strategy reachable from here. This is the cold path, taken only after a
                // batch has already reported an error, so the choice has not been worth plumbing
                // for.
                return self.filter_and_scatter(kernel, &valid, ctx);
            }
            RowExecution::DeferredError(error) => return Err(error),
        };

        match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.with_return_dtype(values, self.args.row_count())
            }
            Validity::Array(valid) => {
                self.with_return_dtype(values.mask(valid)?, self.args.row_count())
            }
            // Handled by the guard above, before the kernel ran.
            Validity::AllInvalid => Ok(self.all_null()),
        }
    }

    /// Materialize the conjoined validity and resolve everything that does not need a mixed-mask
    /// strategy, so that the production selector and the forced-strategy test seam cannot drift
    /// apart on the shortcuts they share. The deferred-error retry in
    /// [`execute_dense`](Self::execute_dense) repeats the same materialize-then-shortcut shape
    /// with different outcomes — all-true is an error there, all-false is all-null — so it stays
    /// open-coded, with its own note on why the ordering is safe.
    fn resolve_validity(
        &self,
        kernel: &impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ResolvedMask> {
        let valid = self
            .validity
            .clone()
            .execute_mask(self.args.row_count(), ctx)?;

        // Check all-true before all-false: an empty mask is both, and must not be treated as
        // all-null (a zero-length non-nullable execution keeps its non-nullable dtype).
        if valid.all_true() {
            return self
                .with_return_dtype(
                    kernel(self.kernel_args(self.args, &self.inputs), ctx)?.into_result()?,
                    self.args.row_count(),
                )
                .map(ResolvedMask::Decided);
        }

        if valid.all_false() {
            return Ok(ResolvedMask::Decided(self.all_null()));
        }

        Ok(ResolvedMask::Mixed(valid))
    }

    /// Materialize the conjoined validity once, take the all-true and all-false shortcuts, and
    /// pick a strategy per batch for a mixed mask.
    ///
    /// Two strategies can execute a mixed mask, and neither is visible to the kernel:
    ///
    /// - **Branch-and-skip** ([`execute_branched`](Self::execute_branched)): hand the _unfiltered_
    ///   arguments plus the mask to `branch`, which computes only the valid rows, then mask the
    ///   full-length result exactly as the dense path does. This skips the filter and the scatter
    ///   entirely, at the price of decoding full-length columns.
    /// - **Filter** ([`filter_and_scatter`](Self::filter_and_scatter)): filter every input down to
    ///   the conjoined-valid rows, run the kernel over those, and scatter its results back into a
    ///   null-padded output. Always available, never encoding-preserving.
    ///
    /// Branch-and-skip is preferred whenever [`branch_beats_filter`] says so, and the filter
    /// strategy is also the fallback for a kernel with no branch execution.
    fn execute_filtered(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        branch: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        filtered_decode_cost: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid = match self.resolve_validity(&kernel, ctx)? {
            ResolvedMask::Decided(result) => return Ok(result),
            ResolvedMask::Mixed(valid) => valid,
        };

        if branch_beats_filter(filtered_decode_cost, &valid)
            && let Some(result) = self.execute_branched(branch, &valid, ctx)?
        {
            return Ok(result);
        }

        self.filter_and_scatter(kernel, &valid, ctx)
    }

    /// Try the branch-and-skip strategy for a mixed mask: the kernel computes only the rows set in
    /// `valid` over the unfiltered inputs, and the full-length result is masked exactly as the
    /// dense path masks. `Ok(None)` means the kernel has no branch execution for these inputs, and
    /// the caller falls back to [`filter_and_scatter`](Self::filter_and_scatter).
    fn execute_branched(
        &self,
        branch: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(values) = branch(self.kernel_args(self.args, &self.inputs), valid, ctx)? else {
            return Ok(None);
        };
        let values = values.into_result()?;

        let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
        self.with_return_dtype(values.mask(mask)?, valid.len())
            .map(Some)
    }

    /// The filter strategy for a mixed mask: filter every input down to the rows set in `valid`,
    /// run the kernel over those, and scatter its results back into a null-padded output.
    fn filter_and_scatter(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let filtered: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.filter(valid.clone()))
            .collect::<VortexResult<_>>()?;

        let args = BorrowedExecutionArgs::new(&filtered, valid.true_count());
        let values = kernel(self.kernel_args(&args, &filtered), ctx)?.into_result()?;

        self.with_return_dtype(self.scatter_valid(values, valid)?, valid.len())
    }

    /// An all-null result of the function's declared return dtype.
    fn all_null(&self) -> ArrayRef {
        ConstantArray::new(
            Scalar::null(self.result_dtype.clone()),
            self.args.row_count(),
        )
        .into_array()
    }

    /// Pair an input view with this batch's planning metadata.
    fn kernel_args<'b>(
        &'b self,
        execution: &'b dyn ExecutionArgs,
        arrays: &'b [ArrayRef],
    ) -> KernelArgs<'b> {
        KernelArgs {
            execution,
            arrays,
            dtypes: &self.arg_dtypes,
            sink_dtype: &self.sink_dtype,
        }
    }

    /// Reconcile the kernel's output dtype with the function's declared return dtype.
    ///
    /// The kernel may ignore nullability, so a nullability difference is cast away. Any other
    /// difference means the declared dtype and the kernel disagree, which is a bug worth naming
    /// rather than silently casting away.
    fn with_return_dtype(&self, values: ArrayRef, expected_len: usize) -> VortexResult<ArrayRef> {
        reconcile_return(self.id, &self.result_dtype, expected_len, values)
    }

    /// Scatter `values` (one per set bit of `valid`, in order) back to the positions of the set
    /// bits, producing an array of length `valid.len()` that is null at every unset position.
    fn scatter_valid(&self, values: ArrayRef, valid: &Mask) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            values.len(),
            valid.true_count(),
            "the {} kernel produced {} rows for {} filtered rows",
            self.id,
            values.len(),
            valid.true_count(),
        );

        let AllOr::Some(slices) = valid.slices() else {
            // The caller handles the all-true and all-false masks.
            vortex_bail!("scatter_valid requires a mixed mask");
        };

        // Gather indices: row i of the output reads values[rank(i)]. Rows behind nulls read index
        // 0, and any in-bounds index would do since they are masked out below (values is non-empty
        // here).
        let mut indices = vec![0u64; valid.len()];
        let mut rank = 0u64;
        for &(start, end) in slices {
            for index in &mut indices[start..end] {
                *index = rank;
                rank += 1;
            }
        }
        let indices = PrimitiveArray::new(indices, Validity::NonNullable).into_array();

        let scattered = values.take(indices)?;

        // A kernel that produced nulls of its own (only `reduce_encoded` may) cannot be wrapped,
        // since a `Masked` child must be all valid. Those nulls have to be unioned with the
        // lifting's, which is what the general masking pass does.
        if scattered.dtype().is_nullable() {
            let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
            return scattered.mask(mask);
        }

        // Attaching the mask as validity rather than masking again: the gathered values are
        // already all valid, so recording which rows survive is the whole job and a `Masked`
        // wrapper says exactly that. Worth 1.13-1.53x here, growing with null density
        // (`null_strategy_bytes`, 65536 rows, divan fastest and median of 100 samples, best of two
        // runs, Apple M4 Max). The same substitution on the dense path measured no difference, so
        // it is deliberately confined to the scatter.
        Ok(MaskedArray::try_new(
            scattered,
            Validity::from_mask(valid.clone(), Nullability::Nullable),
        )?
        .into_array())
    }
}

/// Validate the row count and reconcile nullability against a row function's declared dtype.
pub(super) fn reconcile_return(
    id: ScalarFnId,
    result_dtype: &DType,
    expected_len: usize,
    values: ArrayRef,
) -> VortexResult<ArrayRef> {
    vortex_ensure_eq!(
        values.len(),
        expected_len,
        "the {id} kernel produced {} rows for {expected_len} input rows",
        values.len(),
    );
    vortex_ensure!(
        values.dtype().eq_ignore_nullability(result_dtype),
        "the {id} kernel produced {} but the function declares {result_dtype}",
        values.dtype(),
    );

    if values.dtype() == result_dtype {
        Ok(values)
    } else {
        values.cast(result_dtype.clone())
    }
}

/// The minimum surviving-row fraction (`true_count / len` of the conjoined mask) at which
/// branch-and-skip is still chosen for one filtered decode unit.
///
/// From the branch-and-skip measurements (65536 rows, divan fastest of 100 samples, two runs on a
/// shared 4-vCPU VM). A kernel with a _bulk_ decode never lost under branch: `byte_length` over
/// a byte-string element ran 1.8-5.9x faster than filter at every null density from 1% to 90%, so
/// such kernels skip this check entirely. A kernel with a _per-row_ decode (geo `contains`, which
/// arrow-exports and parses one geometry per row) pays that decode over the full column under
/// branch but only over the survivors under filter, so filter wins once validity is sparse:
///
/// - polygons CONTAINS constant point: branch won 1.07-1.18x at 1-50% nulls; filter won 1.38x at
///   90% nulls (10% of rows surviving).
/// - polygons CONTAINS points, independent nulls on both: branch won up to ~10% null density
///   (~81% surviving); filter won 1.2x at ~56% surviving, 1.9x at ~25%, 11.3x at ~1%.
///
/// A single nullable operand still favored branch at 50% surviving, while two independent nullable
/// operands favored filtering at 81% surviving. Keep those cases distinct instead of collapsing
/// every per-row decode into one boolean. There is not yet enough evidence to distinguish two from
/// three or more decode units, so they share the conservative multi-decode threshold.
pub(super) const ONE_DECODE_BRANCH_MIN_SURVIVING_FRACTION: f64 = 0.50;
pub(super) const MULTI_DECODE_BRANCH_MIN_SURVIVING_FRACTION: f64 = 0.85;

/// Whether the branch-and-skip strategy should be preferred over filtering for the mixed mask
/// `valid`. A zero cost always branches; otherwise the survivor threshold grows when filtering
/// avoids more than one unit of per-row decode work.
pub(super) fn branch_beats_filter(filtered_decode_cost: usize, valid: &Mask) -> bool {
    if filtered_decode_cost == 0 {
        return true;
    }

    let minimum = if filtered_decode_cost == 1 {
        ONE_DECODE_BRANCH_MIN_SURVIVING_FRACTION
    } else {
        MULTI_DECODE_BRANCH_MIN_SURVIVING_FRACTION
    };
    valid.true_count() as f64 >= valid.len() as f64 * minimum
}

/// Which null strategy a forced execution takes for a mixed validity mask.
///
/// A test and benchmark seam: pinning a strategy is how the two are compared and how their
/// agreement is asserted. Production execution selects per batch inside the lifting and never
/// names one. See [`execute_row_fn_with_strategy`](super::execute_row_fn_with_strategy).
#[cfg(any(test, feature = "_test-harness"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullStrategy {
    /// Filter the inputs down to the conjoined-valid rows, run the kernel, and scatter back.
    Filter,

    /// Decode the unfiltered inputs null-tolerantly, compute only the conjoined-valid rows, and
    /// mask the full-length result.
    BranchAndSkip,
}

#[cfg(any(test, feature = "_test-harness"))]
impl Batch<'_> {
    /// Execute this batch with a forced null strategy, bypassing the per-batch selection.
    ///
    /// A test and benchmark seam only. It mirrors [`execute_filtered`](Self::execute_filtered)
    /// (conjoined validity, the all-true and all-false shortcuts, output dtype reconciliation) but
    /// takes the strategy from the caller instead of the selection rule, and it skips the
    /// null-constant and all-constant folds, so do not pass such inputs. `Ok(None)` means
    /// [`NullStrategy::BranchAndSkip`] was forced on a kernel with no branch execution, which the
    /// caller reports rather than silently falling back.
    pub(super) fn execute_with_strategy(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        branch: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        strategy: NullStrategy,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let valid = match self.resolve_validity(&kernel, ctx)? {
            ResolvedMask::Decided(result) => return Ok(Some(result)),
            ResolvedMask::Mixed(valid) => valid,
        };

        match strategy {
            NullStrategy::Filter => self.filter_and_scatter(kernel, &valid, ctx).map(Some),
            NullStrategy::BranchAndSkip => self.execute_branched(branch, &valid, ctx),
        }
    }
}
