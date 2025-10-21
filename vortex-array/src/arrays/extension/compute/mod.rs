// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod sum;
mod take;

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
