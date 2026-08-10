// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Null propagation, constant folding, and strategy execution for one columnar batch.

use smallvec::SmallVec;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::args::KernelArgs;
use super::policy::BatchPlan;
use super::policy::RowPolicy;
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
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::row::execute::RowExecution;
use crate::scalar_fn::row::types::batch_constant;
use crate::validity::Validity;

/// How far [`Batch::resolve_validity`] got before a mixed-mask strategy became necessary.
enum ResolvedMask {
    /// The all-valid or all-null batch was answered without a mixed-mask strategy.
    Decided(ArrayRef),

    /// A mask with both set and unset bits, which a strategy must now execute.
    Mixed(Mask),
}

/// One batch of inputs and the metadata needed before its row kernel runs.
pub struct Batch {
    /// The function being executed, named in the errors this raises.
    id: ScalarFnId,

    /// The number of rows in the original execution scope.
    row_count: usize,

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

    /// The non-nullable dtype the dispatched output capability builds, computed while planning.
    output_dtype: DType,

    /// How the concrete dispatch executes nullable rows.
    policy: RowPolicy,
}

impl Batch {
    /// Collect the inputs and derive their dtype, validity, and execution policy.
    ///
    /// **Not** for a nullary function: with no inputs there is no validity to propagate and no
    /// per-row work to fold, and the all-constant check below would vacuously pass.
    pub fn new(
        id: ScalarFnId,
        args: &dyn ExecutionArgs,
        plan: impl FnOnce(&[DType]) -> VortexResult<BatchPlan>,
    ) -> VortexResult<Self> {
        let row_count = args.row_count();
        let inputs: SmallVec<[ArrayRef; 4]> = (0..args.num_inputs())
            .map(|index| args.get(index))
            .collect::<VortexResult<_>>()?;

        for (index, input) in inputs.iter().enumerate() {
            vortex_ensure_eq!(
                input.len(),
                row_count,
                "the {id} input {index} must have {row_count} rows, got {}",
                input.len(),
            );
        }

        let arg_dtypes: SmallVec<[DType; 4]> =
            inputs.iter().map(|input| input.dtype().clone()).collect();
        let plan = plan(&arg_dtypes)?;
        let result_dtype = plan.result_dtype(&arg_dtypes);

        let mut validity = Validity::NonNullable;
        for input in &inputs {
            validity = validity.and(input.validity()?)?;
        }

        Ok(Self {
            id,
            row_count,
            inputs,
            arg_dtypes,
            validity,
            result_dtype,
            output_dtype: plan.output_dtype,
            policy: plan.policy,
        })
    }

    /// Add null propagation, constant folding, and strategy selection around `kernel`.
    ///
    /// The kernel may ignore input validity. It receives valid-only rows when required, and its
    /// output **must** match the planned dtype up to nullability. `reduce` receives the original
    /// inputs exactly once, before the generic all-constant broadcast, so a function-owned encoded
    /// implementation takes precedence. `try_unfiltered` receives the originals plus a mixed
    /// validity mask; `Ok(None)` selects filter-and-scatter.
    pub fn execute(
        &self,
        reduce: impl FnOnce(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<Option<RowExecution>>,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        try_unfiltered: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Strictness: an all-null batch has no observable row work. Keep the literal-constant
        // check explicit alongside the conjoined validity invariant.
        if matches!(self.validity, Validity::AllInvalid)
            || self
                .inputs
                .iter()
                .any(|input| input.as_constant().is_some_and(|scalar| scalar.is_null()))
        {
            return Ok(self.all_null());
        }

        // The function-owned encoded path takes precedence over the generic all-constant
        // broadcast and sees the original inputs before slicing or filtering changes them.
        if let Some(execution) = reduce(self.kernel_args(&self.inputs, self.row_count), ctx)? {
            match execution {
                RowExecution::Output(values) => return self.finalize_reduced(values),
                RowExecution::DeferredError(error) => {
                    return self.resolve_reduced_error(error, kernel, try_unfiltered, ctx);
                }
            }
        }

        // All inputs constant, and their conjoined validity proves every row non-null. This sees
        // through extension and masked wrappers just like argument decoding does.
        if self.row_count > 0
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
            RowPolicy::ValidOnly => self.execute_valid_only(kernel, try_unfiltered, ctx),
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

        let result = VortexResult::from(kernel(self.kernel_args(&one_row, 1), ctx)?)?;
        let scalar = self.finalize_output(result, 1)?.execute_scalar(0, ctx)?;

        Ok(ConstantArray::new(scalar, self.row_count).into_array())
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
        let values = match kernel(self.kernel_args(&self.inputs, self.row_count), ctx)? {
            RowExecution::Output(values) => values,
            RowExecution::DeferredError(error) if retry_deferred_error => {
                let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

                // Unlike `resolve_validity`, all-true preserves the deferred error and all-false
                // suppresses evidence that came entirely from null rows. An empty loop cannot
                // produce deferred evidence, so the ambiguous empty mask cannot reach this arm.
                if valid.all_true() {
                    return Err(error);
                }
                if valid.all_false() {
                    return Ok(self.all_null());
                }

                // Deferred retry receives only the dense kernel. Filter first so the retry cannot
                // evaluate null rows again.
                return self.filter_and_scatter(kernel, &valid, ctx);
            }
            RowExecution::DeferredError(error) => return Err(error),
        };

        match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.row_count)
            }
            Validity::Array(valid) => self.finalize_output(values.mask(valid)?, self.row_count),
            // Handled by the guard above, before the kernel ran.
            Validity::AllInvalid => Ok(self.all_null()),
        }
    }

    /// Materialize validity and answer all-valid or all-null batches before selecting a mixed-mask
    /// strategy.
    fn resolve_validity(
        &self,
        kernel: &impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ResolvedMask> {
        let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

        // Check all-true before all-false: an empty mask is both, and must not be treated as
        // all-null (a zero-length non-nullable execution keeps its non-nullable dtype).
        if valid.all_true() {
            return self
                .finalize_output(
                    VortexResult::from(kernel(
                        self.kernel_args(&self.inputs, self.row_count),
                        ctx,
                    )?)?,
                    self.row_count,
                )
                .map(ResolvedMask::Decided);
        }

        if valid.all_false() {
            return Ok(ResolvedMask::Decided(self.all_null()));
        }

        Ok(ResolvedMask::Mixed(valid))
    }

    /// Resolve validity, try unfiltered execution, then fall back to filtering.
    fn execute_valid_only(
        &self,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        try_unfiltered: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid = match self.resolve_validity(&kernel, ctx)? {
            ResolvedMask::Decided(result) => return Ok(result),
            ResolvedMask::Mixed(valid) => valid,
        };

        if let Some(result) = self.try_execute_unfiltered(try_unfiltered, &valid, ctx)? {
            return Ok(result);
        }

        self.filter_and_scatter(kernel, &valid, ctx)
    }

    /// Try execution against the original inputs, then mask a returned full-length result.
    fn try_execute_unfiltered(
        &self,
        try_unfiltered: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(execution) =
            try_unfiltered(self.kernel_args(&self.inputs, self.row_count), valid, ctx)?
        else {
            return Ok(None);
        };
        let values = VortexResult::from(execution)?;

        let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
        self.finalize_output(values.mask(mask)?, valid.len())
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

        let values = VortexResult::from(kernel(
            self.kernel_args(&filtered, valid.true_count()),
            ctx,
        )?)?;

        self.finalize_output(self.scatter_valid(values, valid)?, valid.len())
    }

    /// An all-null result of the function's declared return dtype.
    fn all_null(&self) -> ArrayRef {
        ConstantArray::new(Scalar::null(self.result_dtype.clone()), self.row_count).into_array()
    }

    /// Reconcile an encoding-aware result and apply the batch's strict input validity.
    fn finalize_reduced(&self, values: ArrayRef) -> VortexResult<ArrayRef> {
        match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.row_count)
            }
            Validity::Array(valid) => self.finalize_output(values.mask(valid)?, self.row_count),
            // Handled before the encoding-aware hook runs.
            Validity::AllInvalid => Ok(self.all_null()),
        }
    }

    /// Resolve deferred evidence from the encoded path by executing only observable rows.
    fn resolve_reduced_error(
        &self,
        error: VortexError,
        kernel: impl Fn(KernelArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowExecution>,
        try_unfiltered: impl FnOnce(
            KernelArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<RowExecution>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

        if valid.all_true() {
            return Err(error);
        }
        if valid.all_false() {
            return Ok(self.all_null());
        }

        if let Some(result) = self.try_execute_unfiltered(try_unfiltered, &valid, ctx)? {
            return Ok(result);
        }

        self.filter_and_scatter(kernel, &valid, ctx)
    }

    /// Pair an input view with this batch's planning metadata.
    fn kernel_args<'b>(&'b self, arrays: &'b [ArrayRef], row_count: usize) -> KernelArgs<'b> {
        KernelArgs {
            arrays,
            row_count,
            dtypes: &self.arg_dtypes,
            output_dtype: &self.output_dtype,
        }
    }

    /// Finalize an output against this batch's expected length and declared return dtype.
    fn finalize_output(&self, values: ArrayRef, expected_len: usize) -> VortexResult<ArrayRef> {
        finalize_kernel_output(self.id, &self.result_dtype, expected_len, values)
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
        // batch validity, which is what the general masking pass does.
        if scattered.dtype().is_nullable() {
            let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
            return scattered.mask(mask);
        }

        // The gathered values are all valid, so attaching validity is sufficient.
        Ok(MaskedArray::try_new(
            scattered,
            Validity::from_mask(valid.clone(), Nullability::Nullable),
        )?
        .into_array())
    }
}

/// Validate a kernel output, then cast it to the row function's declared nullability.
///
/// `values` **must** contain `expected_len` rows. Its dtype must match `result_dtype` when ignoring
/// nullability. The kernel may omit nullability because batch execution owns strict null
/// propagation, so a nullability-only difference is cast to `result_dtype`.
pub fn finalize_kernel_output(
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
