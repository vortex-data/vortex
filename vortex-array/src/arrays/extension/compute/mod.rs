mod compare;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::ExtensionVTable;
use crate::arrays::extension::ExtensionArray;
use crate::compute::{
    FilterKernel, FilterKernelAdapter, IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts,
    IsSortedKernel, IsSortedKernelAdapter, MinMaxKernel, MinMaxKernelAdapter, MinMaxResult,
    SumKernel, SumKernelAdapter, TakeKernel, TakeKernelAdapter, filter, is_constant_opts,
    is_sorted, is_strict_sorted, min_max, sum, take,
};
use crate::{Array, ArrayRef, IntoArray, register_kernel};

impl FilterKernel for ExtensionVTable {
    fn filter(&self, array: &ExtensionArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::new(array.ext_dtype().clone(), filter(array.storage(), mask)?)
                .into_array(),
        )
    }
}

register_kernel!(FilterKernelAdapter(ExtensionVTable).lift());

impl SumKernel for ExtensionVTable {
    fn sum(&self, array: &ExtensionArray) -> VortexResult<Scalar> {
        sum(array.storage())
    }
}

register_kernel!(SumKernelAdapter(ExtensionVTable).lift());

impl TakeKernel for ExtensionVTable {
    fn take(&self, array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(
            ExtensionArray::new(array.ext_dtype().clone(), take(array.storage(), indices)?)
                .into_array(),
        )
    }
}

register_kernel!(TakeKernelAdapter(ExtensionVTable).lift());

impl MinMaxKernel for ExtensionVTable {
    fn min_max(&self, array: &ExtensionArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(
            min_max(array.storage())?.map(|MinMaxResult { min, max }| MinMaxResult {
                min: Scalar::extension(array.ext_dtype().clone(), min),
                max: Scalar::extension(array.ext_dtype().clone(), max),
            }),
        )
    }
}

register_kernel!(MinMaxKernelAdapter(ExtensionVTable).lift());

impl IsConstantKernel for ExtensionVTable {
    fn is_constant(
        &self,
        array: &ExtensionArray,
        opts: &IsConstantOpts,
    ) -> VortexResult<Option<bool>> {
        is_constant_opts(array.storage(), opts)
    }
}

register_kernel!(IsConstantKernelAdapter(ExtensionVTable).lift());

impl IsSortedKernel for ExtensionVTable {
    fn is_sorted(&self, array: &ExtensionArray) -> VortexResult<bool> {
        is_sorted(array.storage())
    }

    fn is_strict_sorted(&self, array: &ExtensionArray) -> VortexResult<bool> {
        is_strict_sorted(array.storage())
    }
}

register_kernel!(IsSortedKernelAdapter(ExtensionVTable).lift());
