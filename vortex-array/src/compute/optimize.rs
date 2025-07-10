// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output};
use crate::vtable::VTable;
use crate::{Array, ArrayRef};

pub fn optimize(array: &dyn Array) -> VortexResult<ArrayRef> {
    OPTIMIZE_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub static OPTIMIZE_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("optimize".into(), ArcRef::new_ref(&Optimize));
    for kernel in inventory::iter::<OptimizeKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct Optimize;

impl ComputeFnVTable for Optimize {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let OptimizeArgs { array } = OptimizeArgs::try_from(args)?;

        let optimized = optimize_impl(array, kernels)?;
        Ok(optimized.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let OptimizeArgs { array } = OptimizeArgs::try_from(args)?;
        Ok(array.dtype().clone())
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let OptimizeArgs { array } = OptimizeArgs::try_from(args)?;
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

fn optimize_impl(array: &dyn Array, kernels: &[ArcRef<dyn Kernel>]) -> VortexResult<ArrayRef> {
    let args = InvocationArgs {
        inputs: &[array.into()],
        options: &(),
    };

    // Look for an Optimize kernel
    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return output.unwrap_array();
        }
    }
    if let Some(output) = array.invoke(&OPTIMIZE_FN, &args)? {
        return output.unwrap_array();
    }

    // If no kernel is defined, just return the original array.
    Ok(array.to_array())
}

struct OptimizeArgs<'a> {
    array: &'a dyn Array,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for OptimizeArgs<'a> {
    type Error = vortex_error::VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 1 {
            vortex_bail!("Expected 1 input, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_error::vortex_err!("Expected first input to be an array"))?;
        Ok(Self { array })
    }
}

pub trait OptimizeKernel: VTable {
    /// Create an optimized version of the array, typically by compacting buffers
    /// or removing unused data.
    ///
    /// For most array types, this will be a no-op and return the original array.
    /// For variable-length arrays with multiple buffers, this can significantly
    /// reduce memory usage by consolidating referenced data.
    fn optimize(&self, array: &Self::Array) -> VortexResult<ArrayRef>;
}

/// A kernel that implements the optimize function.
pub struct OptimizeKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(OptimizeKernelRef);

#[derive(Debug)]
pub struct OptimizeKernelAdapter<V: VTable>(pub V);

impl<V: VTable + OptimizeKernel> OptimizeKernelAdapter<V> {
    pub const fn lift(&'static self) -> OptimizeKernelRef {
        OptimizeKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + OptimizeKernel> Kernel for OptimizeKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = OptimizeArgs::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(Some(V::optimize(&self.0, array)?.into()))
    }
}
