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
}
