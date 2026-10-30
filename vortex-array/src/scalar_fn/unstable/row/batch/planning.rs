// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inspects borrowed columns before pairing collected inputs with a concrete execution plan.
//!
//! Input lengths, constants, and validity are properties of the batch. Planning derives the output
//! dtype and execution policy from the row signature selected by dispatch.

use std::borrow::Cow;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::BatchInputs;
use super::BatchPlan;
use super::BatchProbe;
use super::RowFnExecutionArgs;
use super::RowInputs;
use super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::arrays::Constant;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::unstable::row::types::is_batch_const;
use crate::validity::Validity;

impl BatchProbe {
    /// Inspect the borrowed inputs: check their row counts, classify batch constants, and conjoin
    /// their validity.
    ///
    /// This **must not** be used for a nullary function. With no input columns, there is no
    /// validity to propagate and the all-constant check would pass vacuously.
    ///
    /// `id` names the function in the errors this raises. It is called only on an error.
    pub(crate) fn new<I: RowInputs + ?Sized>(
        id: impl Fn() -> ScalarFnId,
        inputs: &I,
    ) -> VortexResult<Self> {
        let row_count = inputs.row_count();
        let num_inputs = inputs.num_inputs();
        vortex_ensure!(
            num_inputs <= u32::BITS as usize,
            "the {} function has {num_inputs} inputs, more than a row function supports",
            id(),
        );

        let mut const_mask = 0u32;
        let mut has_null_const = false;
        let mut nullability = Nullability::NonNullable;
        for index in 0..num_inputs {
            let input = inputs.input(index);
            vortex_ensure_eq!(
                input.len(),
                row_count,
                "the {} input {index} has the wrong row count",
                id(),
            );

            if is_batch_const(input) {
                const_mask |= 1 << index;
                has_null_const |= input
                    .as_opt::<Constant>()
                    .is_some_and(|constant| constant.scalar().is_null());
            }

            if input.dtype().is_nullable() {
                nullability = Nullability::Nullable;
            }
        }

        // A null literal makes every row invalid, so skip conjoining the other validities: each
        // conjunction of two array-backed validities builds and optimizes an expression. A
        // non-nullable input has no invalid row, and `NonNullable` is the identity of `and`.
        let validity = if has_null_const {
            Validity::AllInvalid
        } else {
            (0..num_inputs)
                .map(|index| inputs.input(index))
                .filter(|input| input.dtype().is_nullable())
                .try_fold(Validity::NonNullable, |validity, input| {
                    validity.and(input.validity()?)
                })?
        };

        Ok(Self {
            row_count,
            validity,
            const_mask,
            has_null_const,
            nullability,
        })
    }

    /// Return the output validity of a _direct_ batch, or return the probe unchanged for any other
    /// batch.
    ///
    /// A visit finishes a direct batch from its kernel output alone, once that output carries
    /// this validity. A direct batch has no null literal, at least one input that is not
    /// batch-constant, and at least one possibly valid row. It therefore needs no all-null or
    /// all-constant handling. Set `dense` when the visit plans
    /// [`RowPolicy::Dense`](super::RowPolicy::Dense). Without it, only a batch that provably has
    /// no null row is direct, because the kernel visits every row.
    pub(crate) fn into_direct_validity(
        self,
        num_inputs: usize,
        dense: bool,
    ) -> Result<Validity, Self> {
        if self.has_null_const
            || self.const_mask.count_ones() as usize == num_inputs
            || matches!(self.validity, Validity::AllInvalid)
            || !(dense || self.validity.definitely_no_nulls())
        {
            return Err(self);
        }

        Ok(self.validity.union_nullability(self.nullability))
    }

    /// Return the batch-constant classification of the inputs: bit `i` marks input `i`.
    pub(crate) fn const_mask(&self) -> u32 {
        self.const_mask
    }
}

impl<'a> BatchInputs<'a> {
    /// Keep the probed inputs of the function `id`.
    ///
    /// The columns are borrowed when [`RowInputs::as_slice`] provides them, so a caller that
    /// already collected them does not pay for a second copy.
    pub(crate) fn new<I: RowInputs + ?Sized>(
        id: ScalarFnId,
        probe: BatchProbe,
        inputs: &'a I,
    ) -> Self {
        let columns = match inputs.as_slice() {
            Some(columns) => Cow::Borrowed(columns),
            None => Cow::Owned(
                (0..inputs.num_inputs())
                    .map(|index| inputs.input(index).clone())
                    .collect(),
            ),
        };

        Self {
            id,
            inputs: columns,
            probe,
        }
    }

    /// Whether every input is batch-constant.
    pub(super) fn all_batch_const(&self) -> bool {
        self.probe.const_mask.count_ones() as usize == self.inputs.len()
    }

    /// Attach the plan a dispatch selected for these inputs.
    pub(super) fn with_plan<'b>(
        &'b self,
        arg_dtypes: &'b [DType],
        plan: BatchPlan,
    ) -> RowFnExecutionArgs<'b> {
        let result_dtype = plan.result_dtype(arg_dtypes);

        RowFnExecutionArgs {
            batch: self,
            arg_dtypes,
            result_dtype,
            plan,
        }
    }
}

impl RowFnExecutionArgs<'_> {
    /// Return the plan, discarding the batch view.
    pub(super) fn into_plan(self) -> BatchPlan {
        self.plan
    }

    /// Pair an input view with this batch's planning metadata.
    pub(super) fn execution_args<'b>(
        &'b self,
        arrays: &'b [ArrayRef],
        row_count: usize,
    ) -> BorrowedRowFnArgs<'b> {
        BorrowedRowFnArgs::new(arrays, row_count, self.arg_dtypes, &self.plan)
    }
}
