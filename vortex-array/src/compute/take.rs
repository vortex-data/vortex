use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output};
use crate::stats::{Precision, Stat, StatsProviderExt, StatsSet};
use crate::vtable::VTable;
use crate::{Array, ArrayExt, ArrayRef, IntoArray};

pub fn take(array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
    TAKE_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), indices.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub static TAKE_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("take".into(), ArcRef::new_ref(&Take));
    for kernel in inventory::iter::<TakeKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct Take;

impl ComputeFnVTable for Take {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let TakeArgs { array, indices } = TakeArgs::try_from(args)?;

        // TODO(ngates): if indices are sorted and unique (strict-sorted), then we should delegate to
        //  the filter function since they're typically optimised for this case.
        // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
        //  such that canonicalize does less work.

        if indices.all_invalid()? {
            return Ok(ConstantArray::new(
                Scalar::null(array.dtype().as_nullable()),
                indices.len(),
            )
            .into_array()
            .into());
        }

        // We know that constant array don't need stats propagation, so we can avoid the overhead of
        // computing derived stats and merging them in.
        let derived_stats = (!array.is_constant()).then(|| derive_take_stats(array));

        let taken = take_impl(array, indices, kernels)?;

        if let Some(derived_stats) = derived_stats {
            let mut stats = taken.statistics().to_owned();
            stats.combine_sets(&derived_stats, array.dtype())?;
            for (stat, val) in stats.into_iter() {
                taken.statistics().set(stat, val)
            }
        }

        Ok(taken.into())
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let TakeArgs { array, indices } = TakeArgs::try_from(args)?;

        if !indices.dtype().is_int() {
            vortex_bail!(
                "Take indices must be an integer type, got {}",
                indices.dtype()
            );
        }

        // If either the indices or the array are nullable, the result should be nullable.
        let expected_nullability = indices.dtype().nullability() | array.dtype().nullability();

        Ok(array.dtype().with_nullability(expected_nullability))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        let TakeArgs { indices, .. } = TakeArgs::try_from(args)?;
        Ok(indices.len())
    }

    fn is_elementwise(&self) -> bool {
        false
    }
}

fn derive_take_stats(arr: &dyn Array) -> StatsSet {
    let stats = arr.statistics().to_owned();

    let is_constant = stats.get_as::<bool>(Stat::IsConstant);

    let mut stats = stats.keep_inexact_stats(&[
        // Cannot create values smaller than min or larger than max
        Stat::Min,
        Stat::Max,
    ]);

    if is_constant == Some(Precision::Exact(true)) {
        // Any combination of elements from a constant array is still const
        stats.set(Stat::IsConstant, Precision::exact(true));
    }

    stats
}

fn take_impl(
    array: &dyn Array,
    indices: &dyn Array,
    kernels: &[ArcRef<dyn Kernel>],
) -> VortexResult<ArrayRef> {
    let args = InvocationArgs {
        inputs: &[array.into(), indices.into()],
        options: &(),
    };

    // First look for a TakeFrom specialized on the indices.
    for kernel in TAKE_FROM_FN.kernels() {
        if let Some(output) = kernel.invoke(&args)? {
            return output.unwrap_array();
        }
    }
    if let Some(output) = indices.invoke(&TAKE_FROM_FN, &args)? {
        return output.unwrap_array();
    }

    // Then look for a Take kernel
    for kernel in kernels {
        if let Some(output) = kernel.invoke(&args)? {
            return output.unwrap_array();
        }
    }
    if let Some(output) = array.invoke(&TAKE_FN, &args)? {
        return output.unwrap_array();
    }

    // Otherwise, canonicalize and try again.
    if !array.is_canonical() {
        log::debug!("No take implementation found for {}", array.encoding_id());
        let canonical = array.to_canonical()?;
        return take(canonical.as_ref(), indices);
    }

    vortex_bail!("No take implementation found for {}", array.encoding_id());
}

struct TakeArgs<'a> {
    array: &'a dyn Array,
    indices: &'a dyn Array,
}

impl<'a> TryFrom<&InvocationArgs<'a>> for TakeArgs<'a> {
    type Error = VortexError;

    fn try_from(value: &InvocationArgs<'a>) -> Result<Self, Self::Error> {
        if value.inputs.len() != 2 {
            vortex_bail!("Expected 2 inputs, found {}", value.inputs.len());
        }
        let array = value.inputs[0]
            .array()
            .ok_or_else(|| vortex_err!("Expected first input to be an array"))?;
        let indices = value.inputs[1]
            .array()
            .ok_or_else(|| vortex_err!("Expected second input to be an array"))?;
        Ok(Self { array, indices })
    }
}

pub trait TakeKernel: VTable {
    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Panics
    ///
    /// Using `indices` that are invalid for the given `array` will cause a panic.
    fn take(&self, array: &Self::Array, indices: &dyn Array) -> VortexResult<ArrayRef>;
}

/// A kernel that implements the filter function.
pub struct TakeKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(TakeKernelRef);

#[derive(Debug)]
pub struct TakeKernelAdapter<V: VTable>(pub V);

impl<V: VTable + TakeKernel> TakeKernelAdapter<V> {
    pub const fn lift(&'static self) -> TakeKernelRef {
        TakeKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + TakeKernel> Kernel for TakeKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = TakeArgs::try_from(args)?;
        let Some(array) = inputs.array.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(Some(V::take(&self.0, array, inputs.indices)?.into()))
    }
}

pub static TAKE_FROM_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("take_from".into(), ArcRef::new_ref(&TakeFrom));
    for kernel in inventory::iter::<TakeFromKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

pub struct TakeFrom;

impl ComputeFnVTable for TakeFrom {
    fn invoke(
        &self,
        _args: &InvocationArgs,
        _kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        vortex_bail!(
            "TakeFrom should not be invoked directly. Its kernels are used to accelerated the Take function"
        )
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        Take.return_dtype(args)
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        Take.return_len(args)
    }

    fn is_elementwise(&self) -> bool {
        Take.is_elementwise()
    }
}

pub trait TakeFromKernel: VTable {
    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    fn take_from(&self, indices: &Self::Array, array: &dyn Array)
    -> VortexResult<Option<ArrayRef>>;
}

pub struct TakeFromKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(TakeFromKernelRef);

#[derive(Debug)]
pub struct TakeFromKernelAdapter<V: VTable>(pub V);

impl<V: VTable + TakeFromKernel> TakeFromKernelAdapter<V> {
    pub const fn lift(&'static self) -> TakeFromKernelRef {
        TakeFromKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + TakeFromKernel> Kernel for TakeFromKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let inputs = TakeArgs::try_from(args)?;
        let Some(indices) = inputs.indices.as_opt::<V>() else {
            return Ok(None);
        };
        Ok(V::take_from(&self.0, indices, inputs.array)?.map(Output::from))
    }
}
