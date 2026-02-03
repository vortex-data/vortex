// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::compute::UnaryArgs;
use crate::vtable::VTable;

static INVERT_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("invert".into(), ArcRef::new_ref(&Invert));
    for kernel in inventory::iter::<InvertKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    INVERT_FN.kernels().len()
}

/// Logically invert a boolean array, preserving its validity.
pub fn invert(array: &dyn Array) -> VortexResult<ArrayRef> {
    INVERT_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &(),
        })?
        .unwrap_array()
}

struct Invert;

impl ComputeFnVTable for Invert {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;

        // PREAMBLE

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&INVERT_FN, args)? {
            return Ok(output);
        }

        // Otherwise, we canonicalize into a boolean array and invert.
        tracing::debug!(
            "No invert implementation found for encoding {}",
            array.encoding_id(),
        );
        if array.is_canonical() {
            vortex_panic!("Canonical bool array does not implement invert");
        }
        Ok(invert(&array.to_bool().into_array())?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;

        if !matches!(array.dtype(), DType::Bool(..)) {
            vortex_bail!("Expected boolean array, got {}", array.dtype());
        }
        Ok(array.dtype().clone())
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct InvertArgs<'a> {
    array: &'a dyn Array,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for InvertArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 1 {
            vortex_bail!("Invert expects exactly one argument",);
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Invert expects an array argument"))?;
        Ok(InvertArgs { array })
    }
}

pub struct InvertKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(InvertKernelRef);

pub trait InvertKernel: VTable {
    /// Logically invert a boolean array. Converts true -> false, false -> true, null -> null.
    fn invert(&self, array: &Self::Array) -> VortexResult<ArrayRef>;
}

#[derive(Debug)]
pub struct InvertKernelAdapter<V: VTable>(pub V);

impl<V: VTable + InvertKernel> InvertKernelAdapter<V> {
    pub const fn lift(&'static self) -> InvertKernelRef {
        InvertKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + InvertKernel> Kernel for InvertKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let args = InvertArgs::try_from(args)?;
        let Some(array) = args.array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(Some(V::invert(&self.0, array)?.into()))
    }
}
