// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::Vector;

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
        let mut datums: Vec<Datum> = Vec::with_capacity(self.inputs.len());
        for input in self.inputs {
            match input {
                KernelInput::Scalar(s) => {
                    datums.push(Datum::Scalar(s));
                }
                KernelInput::Vector(kernel) => {
                    datums.push(Datum::Vector(kernel.execute()?));
                }
            }
        }

        let args = ExecutionArgs {
            datums,
            dtypes: self.input_dtypes,
            row_count: self.row_count,
            return_dtype: self.return_dtype,
        };

        Ok(self.scalar_fn.execute(args)?.ensure_vector(self.row_count))
    }

    fn push_down_filter(self: Box<Self>, selection: &Mask) -> VortexResult<PushDownResult> {
        let mut new_inputs = Vec::with_capacity(self.inputs.len());
        for input in self.inputs {
            match input {
                KernelInput::Scalar(s) => {
                    new_inputs.push(KernelInput::Scalar(s.clone()));
                }
                KernelInput::Vector(k) => {
                    new_inputs.push(KernelInput::Vector(k.force_push_down_filter(selection)?));
                }
            }
        }

        Ok(PushDownResult::Pushed(Box::new(ScalarFnKernel {
            scalar_fn: self.scalar_fn,
            inputs: new_inputs,
            input_dtypes: self.input_dtypes,
            row_count: selection.true_count(),
            return_dtype: self.return_dtype,
        })))
    }
}
