// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayRef;
use crate::compute::ComputeFn;
use crate::compute::ComputeFnVTable;
use crate::compute::InvocationArgs;
use crate::compute::Kernel;
use crate::compute::Output;
use crate::vtable::VTable;

static CAST_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("cast".into(), ArcRef::new_ref(&Cast));
    for kernel in inventory::iter::<CastKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub(crate) fn warm_up_vtable() -> usize {
    CAST_FN.kernels().len()
}

/// Attempt to cast an array to a desired DType.
///
/// Some array support the ability to narrow or upcast.
/// Struct DType casting is nominal and will resolve fields by their name and not their position.
///
/// Casting allows you:
/// 1. Cast array/field to a different dtype.
/// 2. Add fields to a struct with null value.
/// 3. Reorder fields in a struct.
/// 4. Remove fields from a struct.
/// 5. Cast top level struct nullability
pub fn cast(array: &dyn Array, dtype: &DType) -> VortexResult<ArrayRef> {
    CAST_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), dtype.into()],
            options: &(),
        })?
        .unwrap_array()
}

struct Cast;

impl ComputeFnVTable for Cast {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let CastArgs { array, dtype } = CastArgs::try_from(args)?;

        if array.dtype() == dtype {
            return Ok(array.to_array().into());
        }

        // TODO(ngates): check for null_count if dtype is non-nullable

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }

        // Otherwise, we fall back to the canonical implementations.
        tracing::debug!(
            "Falling back to canonical cast for encoding {} and dtype {} to {}",
            array.encoding_id(),
            array.dtype(),
            dtype
        );

        if array.is_canonical() {
            vortex_bail!(
                "No compute kernel to cast array {} with dtype {} to {}",
                array.encoding_id(),
                array.dtype(),
                dtype
            );
        }

        Ok(cast(array.to_canonical()?.as_ref(), dtype)?.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let CastArgs { dtype, .. } = CastArgs::try_from(args)?;
        Ok(dtype.clone())
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let CastArgs { array, .. } = CastArgs::try_from(args)?;
        Ok(array.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

struct CastArgs<'a> {
    array: &'a dyn Array,
    dtype: &'a DType,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for CastArgs<'a> {
    type Error = VortexError;

    fn try_from(args: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if args.inputs.len() != 2 {
            vortex_bail!(
                "Cast function requires 2 arguments, but got {}",
                args.inputs.len()
            );
        }
        let array = args.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Missing array argument"))?;
        let dtype = args.inputs[1]
            .dtype()
            .ok_or_else(|| vortex_err!("Missing dtype argument"))?;

        Ok(CastArgs { array, dtype })
    }
}

pub struct CastKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(CastKernelRef);

pub trait CastKernel: VTable {
    fn cast(&self, array: &Self::Array, dtype: &DType) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Debug)]
pub struct CastKernelAdapter<V: VTable>(pub V);

impl<V: VTable + CastKernel> CastKernelAdapter<V> {
    pub const fn lift(&'static self) -> CastKernelRef {
        CastKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + CastKernel> Kernel for CastKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let CastArgs { array, dtype } = CastArgs::try_from(args)?;
        let Some(array) = array.as_opt::<V>() else {
            return Ok(None);
        };

        Ok(V::cast(&self.0, array, dtype)?.map(|o| o.into()))
    }
}
