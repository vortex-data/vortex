// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::compute::{ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Output};
use crate::stats::{Precision, Stat, StatsProviderExt, StatsSet};
use crate::vtable::VTable;
use crate::{Array, ArrayRef, Canonical, IntoArray};

/// Take values from an array at the given indices.
///
/// Returns a new array containing the values from `array` at the positions
/// specified by `indices`. The result has the same length as `indices`.
///
/// # Arguments
///
/// * `array` - The array to take values from
/// * `indices` - Integer array specifying which indices to take
///
/// # Errors
///
/// Returns an error if:
/// - The indices array is not an integer type
/// - Any index is out of bounds for the array
/// - The operation fails during execution
pub fn take(array: &dyn Array, indices: &dyn Array) -> VortexResult<ArrayRef> {
    if indices.is_empty() {
        return Ok(Canonical::empty(
            &array
                .dtype()
                .union_nullability(indices.dtype().nullability()),
        )
        .into_array());
    }

    TAKE_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), indices.into()],
            options: &(),
        })?
        .unwrap_array()
}

/// The global take compute function.
///
/// This function is initialized with all registered take kernels
/// and provides the main entry point for take operations.
pub static TAKE_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("take".into(), ArcRef::new_ref(&Take));
    for kernel in inventory::iter::<TakeKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

/// Implementation of the take compute function.
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

        Ok(array
            .dtype()
            .union_nullability(indices.dtype().nullability()))
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

/// Trait for implementing take operations on specific array types.
///
/// This trait allows array encodings to provide optimized implementations
/// of the take operation.
pub trait TakeKernel: VTable {
    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Panics
    ///
    /// Using `indices` that are invalid for the given `array` will cause a panic.
    fn take(&self, array: &Self::Array, indices: &dyn Array) -> VortexResult<ArrayRef>;
}

/// A reference to a take kernel implementation.
///
/// This type is used in the inventory collection system to register take kernels.
pub struct TakeKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(TakeKernelRef);

/// Adapter to convert a VTable implementing TakeKernel into a Kernel.
#[derive(Debug)]
pub struct TakeKernelAdapter<V: VTable>(pub V);

impl<V: VTable + TakeKernel> TakeKernelAdapter<V> {
    /// Convert this adapter into a TakeKernelRef for registration.
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

/// The global take_from compute function.
///
/// This function provides specialized kernels for take operations
/// that are optimized based on the indices array type.
pub static TAKE_FROM_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("take_from".into(), ArcRef::new_ref(&TakeFrom));
    for kernel in inventory::iter::<TakeFromKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

/// Implementation of the take_from compute function.
///
/// This function is used internally to accelerate take operations
/// by providing indices-specialized kernels.
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

/// Trait for implementing take_from operations on specific indices types.
///
/// This trait allows indices array encodings to provide optimized implementations
/// of the take operation when they are the indices.
pub trait TakeFromKernel: VTable {
    /// Create a new array by taking values from `array` using these indices.
    ///
    /// This is called when the indices array matches this kernel's type,
    /// allowing for optimized implementations.
    ///
    /// Returns `None` if this kernel cannot handle the operation.
    fn take_from(&self, indices: &Self::Array, array: &dyn Array)
    -> VortexResult<Option<ArrayRef>>;
}

/// A reference to a take_from kernel implementation.
///
/// This type is used in the inventory collection system to register take_from kernels.
pub struct TakeFromKernelRef(pub ArcRef<dyn Kernel>);
inventory::collect!(TakeFromKernelRef);

/// Adapter to convert a VTable implementing TakeFromKernel into a Kernel.
#[derive(Debug)]
pub struct TakeFromKernelAdapter<V: VTable>(pub V);

impl<V: VTable + TakeFromKernel> TakeFromKernelAdapter<V> {
    /// Convert this adapter into a TakeFromKernelRef for registration.
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
