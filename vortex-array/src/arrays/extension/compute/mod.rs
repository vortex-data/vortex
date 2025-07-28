// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;

use std::sync::Arc;

use vortex_dtype::ExtDType;
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
        let taken_storage = take(array.storage(), indices)?;
        
        // If the storage dtype changed (e.g., became nullable due to nullable indices),
        // we need to update the extension dtype to match
        let ext_dtype = if taken_storage.dtype() != array.ext_dtype().storage_dtype() {
            Arc::new(ExtDType::new(
                array.ext_dtype().id().clone(),
                Arc::new(taken_storage.dtype().clone()),
                array.ext_dtype().metadata().cloned(),
            ))
        } else {
            array.ext_dtype().clone()
        };
        
        Ok(ExtensionArray::new(ext_dtype, taken_storage).into_array())
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

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_dtype::{DType, ExtDType, ExtID, Nullability, PType};

    use crate::IntoArray;
    use crate::arrays::{ExtensionArray, PrimitiveArray};
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::take::test_take_conformance;

    #[test]
    fn test_filter_extension_array() {
        // Create a simple extension type (e.g., UUID represented as u64)
        let ext_dtype = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            None,
        );

        // Create storage array
        let storage = PrimitiveArray::from_iter([1u64, 2, 3, 4, 5]).into_array();
        let array = ExtensionArray::new(Arc::new(ext_dtype), storage);
        test_filter_conformance(array.as_ref());

        // Test with nullable extension type
        let ext_dtype_nullable = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(DType::Primitive(PType::U64, Nullability::Nullable)),
            None,
        );
        let storage = PrimitiveArray::from_option_iter([Some(1u64), None, Some(3), Some(4), None])
            .into_array();
        let array = ExtensionArray::new(Arc::new(ext_dtype_nullable), storage);
        test_filter_conformance(array.as_ref());
    }

    #[test]
    fn test_take_extension_array() {
        // Create a simple extension type (e.g., UUID represented as u64)
        let storage = PrimitiveArray::from_iter([1u64, 2, 3, 4, 5]).into_array();
        let ext_dtype = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        let array = ExtensionArray::new(Arc::new(ext_dtype), storage);
        test_take_conformance(array.as_ref());

        // Test with nullable extension type
        let storage = PrimitiveArray::from_option_iter([Some(1u64), None, Some(3), Some(4), None])
            .into_array();
        let ext_dtype_nullable = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        let array = ExtensionArray::new(Arc::new(ext_dtype_nullable), storage);
        test_take_conformance(array.as_ref());

        // Test with single element
        let storage = PrimitiveArray::from_iter([42u64]).into_array();
        let ext_dtype_single = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        let array = ExtensionArray::new(Arc::new(ext_dtype_single), storage);
        test_take_conformance(array.as_ref());

        // Test with larger array for additional edge cases
        let storage = PrimitiveArray::from_iter(0u64..100).into_array();
        let ext_dtype_large = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        let array = ExtensionArray::new(Arc::new(ext_dtype_large), storage);
        test_take_conformance(array.as_ref());
    }
}
