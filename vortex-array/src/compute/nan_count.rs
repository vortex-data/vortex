use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output, UnaryArgs};
use crate::stats::{Precision, Stat};
use crate::vtable::VTable;
use crate::{Array, ArrayExt};

/// Computes the number of NaN values in the array.
pub fn nan_count(array: &dyn Array) -> VortexResult<usize> {
    Ok(NAN_COUNT_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into()],
            options: &(),
        })?
        .unwrap_scalar()?
        .as_primitive()
        .as_::<usize>()?
        .vortex_expect("NaN count should not return null"))
}

struct NaNCount;

impl ComputeFnVTable for NaNCount {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;

        let nan_count = nan_count_impl(array, kernels)?;

        // Update the stats set with the computed NaN count
        array.statistics().set(
            Stat::NaNCount,
            Precision::Exact(ScalarValue::from(nan_count as u64)),
        );

        Ok(Scalar::from(nan_count as u64).into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;
        Stat::NaNCount
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Cannot compute NaN count for dtype {}", array.dtype()))
    }

    fn return_len(&self, _args: &InvocationArgs) -> VortexResult<usize> {
        Ok(1)
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

/// Computes the min and max of an array, returning the (min, max) values
pub trait NaNCountKernel: VTable {
    fn nan_count(&self, array: &Self::Array) -> VortexResult<usize>;
}

pub static NAN_COUNT_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("nan_count".into(), ArcRef::new_ref(&NaNCount));
    for kernel in inventory::iter::<NaNCountKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct NaNCountKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(NaNCountKernelRef);

#[derive(Debug)]
pub struct NaNCountKernelAdapter<V: VTable>(pub V);

impl<V: VTable + NaNCountKernel> NaNCountKernelAdapter<V> {
    pub const fn lift(&'static self) -> NaNCountKernelRef {
        NaNCountKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + NaNCountKernel> Kernel for NaNCountKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let UnaryArgs { array, .. } = UnaryArgs::<()>::try_from(args)?;
        let Some(array) = array.as_opt::<V>() else {
            return Ok(None);
        };
        let nan_count = V::nan_count(&self.0, array)?;
        Ok(Some(Scalar::from(nan_count as u64).into()))
    }
}

fn nan_count_impl(array: &dyn Array, kernels: &[ArcRef<dyn Kernel>]) -> VortexResult<usize> {
    if array.is_empty() || array.valid_count()? == 0 {
        return Ok(0);
    }

    if let Some(nan_count) = array
        .statistics()
        .get_as::<usize>(Stat::NaNCount)
        .and_then(Precision::as_exact)
    {
        // If the NaN count is already computed, return it
        return Ok(nan_count);
    }

    let args = InvocationArgs {
        inputs: &[array.into()],
        options: &(),
    };

    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return output
                .unwrap_scalar()?
                .as_primitive()
                .as_::<usize>()?
                .ok_or_else(|| vortex_err!("NaN count should not return null"));
        }
    }
    if let Some(output) = array.invoke(&NAN_COUNT_FN, &args)? {
        return output
            .unwrap_scalar()?
            .as_primitive()
            .as_::<usize>()?
            .ok_or_else(|| vortex_err!("NaN count should not return null"));
    }

    if !array.is_canonical() {
        let canonical = array.to_canonical()?;
        return nan_count(canonical.as_ref());
    }

    vortex_bail!(
        "No NaN count kernel found for array type: {}",
        array.dtype()
    )
}
