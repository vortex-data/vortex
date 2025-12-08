// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::Vector;

use crate::arrays::FilterKernel;
use crate::expr::ExecutionArgs;
use crate::expr::ScalarFn;
use crate::kernel::Kernel;
use crate::kernel::KernelRef;
use crate::kernel::PushDownResult;

/// A CPU kernel for executing scalar functions.
#[derive(Debug)]
pub struct ScalarFnKernel {
    /// The scalar function to apply.
    pub(super) scalar_fn: ScalarFn,

    /// Inputs to the kernel, either constants or other kernels.
    pub(super) inputs: Vec<KernelInput>,
    /// The input data types
    pub(super) input_dtypes: Vec<DType>,
    /// The row count for vector inputs
    pub(super) row_count: usize,
    /// The return data type
    pub(super) return_dtype: DType,
}

#[derive(Debug)]
pub(super) enum KernelInput {
    Scalar(Scalar),
    Vector(KernelRef),
}

impl Kernel for ScalarFnKernel {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let datums: Vec<_> = self
            .inputs
            .into_iter()
            .map(|input| match input {
                KernelInput::Scalar(s) => Ok(Datum::Scalar(s)),
                KernelInput::Vector(k) => k.execute().map(Datum::Vector),
            })
            .try_collect()?;

        let args = ExecutionArgs {
            datums,
            dtypes: self.input_dtypes,
            row_count: self.row_count,
            return_dtype: self.return_dtype,
        };

        Ok(self.scalar_fn.execute(args)?.ensure_vector(self.row_count))
    }

    fn cost_estimate(&self, selection: &Mask) -> f64 {
        let self_cost = self.scalar_fn.cost_estimate(selection);
        let child_cost = self
            .inputs
            .iter()
            .map(|input| match input {
                KernelInput::Scalar(_) => 0.0,
                KernelInput::Vector(k) => k.cost_estimate(selection),
            })
            .sum::<f64>();

        self_cost + child_cost
    }

    fn push_down_filter(self: Box<Self>, selection: &Mask) -> VortexResult<PushDownResult> {
        let mut new_inputs = Vec::with_capacity(self.inputs.len());
        for (input, dtype) in self.inputs.into_iter().zip(&self.input_dtypes) {
            match input {
                KernelInput::Scalar(s) => {
                    new_inputs.push(KernelInput::Scalar(s.clone()));
                }
                KernelInput::Vector(k) => match k.push_down_filter(selection)? {
                    PushDownResult::Pushed(new_k) => {
                        new_inputs.push(KernelInput::Vector(new_k));
                    }
                    PushDownResult::NotPushed(k) => {
                        let new_k = FilterKernel::new(k, selection.clone(), dtype.clone());
                        new_inputs.push(KernelInput::Vector(Box::new(new_k)));
                    }
                },
            }
        }

        Ok(PushDownResult::Pushed(Box::new(ScalarFnKernel {
            scalar_fn: self.scalar_fn,
            inputs: new_inputs,
            input_dtypes: self.input_dtypes,
            row_count: self.row_count,
            return_dtype: self.return_dtype,
        })))
    }
}
