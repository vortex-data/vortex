// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::Batch;
use super::BatchPlan;
use super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::validity::Validity;

impl Batch {
    /// Collect the inputs and derive their dtypes, validity, and execution policy.
    ///
    /// This constructor **must not** be used for a nullary function. With no input columns, there
    /// is no validity to propagate and the all-constant check would pass vacuously.
    pub(crate) fn new(
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

    /// Pair an input view with this batch's planning metadata.
    pub(super) fn execution_args<'b>(
        &'b self,
        arrays: &'b [ArrayRef],
        row_count: usize,
    ) -> BorrowedExecutionArgs<'b> {
        BorrowedExecutionArgs::new(
            arrays,
            row_count,
            &self.arg_dtypes,
            &self.output_dtype,
            self.policy,
        )
    }
}
