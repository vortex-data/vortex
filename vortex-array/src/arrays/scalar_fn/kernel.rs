// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::Vector;

use crate::expr::ExecutionArgs;
use crate::expr::ScalarFn;
use crate::kernel::Kernel;
use crate::kernel::KernelRef;

/// A CPU kernel for executing scalar functions.
#[derive(Debug)]
pub struct ScalarFnKernel {
    /// The scalar function to apply.
    scalar_fn: ScalarFn,

    /// Inputs to the kernel, either constants or other kernels.
    inputs: Vec<KernelInput>,
    /// The input data types
    input_dtypes: Vec<DType>,
    /// The row count for vector inputs
    row_count: usize,
    /// The return data type
    return_dtype: DType,
}

#[derive(Debug)]
enum KernelInput {
    Scalar(Scalar),
    Vector(KernelRef),
}

impl Kernel for ScalarFnKernel {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let datums: Vec<_> = self
            .inputs
            .into_iter()
            .map(|input| {
                Ok(match input {
                    KernelInput::Scalar(s) => Datum::Scalar(s),
                    KernelInput::Vector(k) => Datum::Vector(k.execute()?),
                })
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

    fn children(&self) -> Vec<&KernelRef> {
        self.inputs
            .iter()
            .filter_map(|input| match input {
                KernelInput::Vector(k) => Some(k),
                KernelInput::Scalar(_) => None,
            })
            .collect()
    }
}
