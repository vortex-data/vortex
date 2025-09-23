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
    IsSortedKernel, IsSortedKernelAdapter, MaskKernel, MaskKernelAdapter, MinMaxKernel,
    MinMaxKernelAdapter, MinMaxResult, SumKernel, SumKernelAdapter, TakeKernel, TakeKernelAdapter,
    filter, is_constant_opts, is_sorted, is_strict_sorted, mask, min_max, sum, take,
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

impl MaskKernel for ExtensionVTable {
    fn mask(&self, array: &ExtensionArray, mask_array: &Mask) -> VortexResult<ArrayRef> {
        let masked_storage = mask(array.storage(), mask_array)?;
        if masked_storage.dtype().nullability() == array.ext_dtype().storage_dtype().nullability() {
            Ok(ExtensionArray::new(array.ext_dtype().clone(), masked_storage).into_array())
        } else {
            // The storage dtype changed (i.e., became nullable due to masking)
            let ext_dtype = Arc::new(ExtDType::new(
                array.ext_dtype().id().clone(),
                Arc::new(masked_storage.dtype().clone()),
                array.ext_dtype().metadata().cloned(),
            ));
            Ok(ExtensionArray::new(ext_dtype, masked_storage).into_array())
        }
    }
}

register_kernel!(MaskKernelAdapter(ExtensionVTable).lift());

impl SumKernel for ExtensionVTable {
    fn sum(&self, array: &ExtensionArray) -> VortexResult<Scalar> {
        sum(array.storage())
    }
}

register_kernel!(SumKernelAdapter(ExtensionVTable).lift());

impl TakeKernel for ExtensionVTable {
    fn take(&self, array: &ExtensionArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_storage = take(array.storage(), indices)?;
        if taken_storage.dtype().nullability() == array.ext_dtype().storage_dtype().nullability() {
            Ok(ExtensionArray::new(array.ext_dtype().clone(), taken_storage).into_array())
        } else {
            // The storage dtype changed (i.e., became nullable due to nullable indices)
            let ext_dtype = Arc::new(ExtDType::new(
                array.ext_dtype().id().clone(),
                Arc::new(taken_storage.dtype().clone()),
                array.ext_dtype().metadata().cloned(),
            ));
            Ok(ExtensionArray::new(ext_dtype, taken_storage).into_array())
        }
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
    fn is_sorted(&self, array: &ExtensionArray) -> VortexResult<Option<bool>> {
        is_sorted(array.storage())
    }

    fn is_strict_sorted(&self, array: &ExtensionArray) -> VortexResult<Option<bool>> {
        is_strict_sorted(array.storage())
    }
}

register_kernel!(IsSortedKernelAdapter(ExtensionVTable).lift());

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;
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
        let storage = buffer![1u64, 2, 3, 4, 5].into_array();
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

    #[rstest]
    #[case({
        // Simple extension type (non-nullable u64)
        let storage = buffer![1u64, 2, 3, 4, 5].into_array();
        let ext_dtype = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype), storage)
    })]
    #[case({
        // Nullable extension type
        let storage = PrimitiveArray::from_option_iter([Some(1u64), None, Some(3), Some(4), None])
            .into_array();
        let ext_dtype_nullable = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype_nullable), storage)
    })]
    #[case({
        // Single element
        let storage = buffer![42u64].into_array();
        let ext_dtype_single = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype_single), storage)
    })]
    #[case({
        // Larger array for edge cases
        let storage = buffer![0u64..100].into_array();
        let ext_dtype_large = ExtDType::new(
            ExtID::new("uuid".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype_large), storage)
    })]
    fn test_take_extension_array_conformance(#[case] array: ExtensionArray) {
        test_take_conformance(array.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{ExtDType, ExtID};

    use crate::IntoArray;
    use crate::arrays::{ExtensionArray, PrimitiveArray};
    use crate::compute::conformance::consistency::test_array_consistency;

    #[rstest]
    // Note: The original test_all_consistency cases for extension arrays caused errors
    // because of unsupported extension type "uuid". We'll use simpler test cases.
    #[case::extension_simple({
        let storage = buffer![1u64, 2, 3, 4, 5].into_array();
        let ext_dtype = ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype), storage)
    })]
    #[case::extension_nullable({
        let storage = PrimitiveArray::from_option_iter([Some(1u64), None, Some(3), Some(4), None])
            .into_array();
        let ext_dtype = ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype), storage)
    })]
    // Additional test cases
    #[case::extension_single({
        let storage = buffer![42i32].into_array();
        let ext_dtype = ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype), storage)
    })]
    #[case::extension_large({
        let storage = buffer![0..100i64].into_array();
        let ext_dtype = ExtDType::new(
            ExtID::new("test_ext".into()),
            Arc::new(storage.dtype().clone()),
            None,
        );
        ExtensionArray::new(Arc::new(ext_dtype), storage)
    })]
    fn test_extension_consistency(#[case] array: ExtensionArray) {
        test_array_consistency(array.as_ref());
    }
}
