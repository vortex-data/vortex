// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<ExtensionVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ExtensionFilterPushDownRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(ExtensionVTable)),
]);

/// Push filter operations into the storage array of an extension array.
#[derive(Debug)]
struct ExtensionFilterPushDownRule;

impl ArrayParentReduceRule<ExtensionVTable> for ExtensionFilterPushDownRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        child: &ExtensionArray,
        parent: &FilterArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        debug_assert_eq!(child_idx, 0);
        let filtered_storage = child
            .storage()
            .clone()
            .filter(parent.filter_mask().clone())?;
        Ok(Some(
            ExtensionArray::new(child.ext_dtype().clone(), filtered_storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::ExtDType;
    use vortex_dtype::ExtDTypeRef;
    use vortex_dtype::ExtID;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::extension::EmptyMetadata;
    use vortex_dtype::extension::ExtDTypeVTable;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::Array;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::ExtensionVTable;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnArrayExt;
    use crate::expr::Binary;
    use crate::expr::Operator;
    use crate::optimizer::ArrayOptimizer;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExt;
    impl ExtDTypeVTable for TestExt {
        type Metadata = EmptyMetadata;

        fn id(&self) -> ExtID {
            ExtID::new_ref("test_ext")
        }

        fn validate_dtype(
            &self,
            _options: &Self::Metadata,
            _storage_dtype: &DType,
        ) -> VortexResult<()> {
            Ok(())
        }
    }

    fn test_ext_dtype() -> ExtDTypeRef {
        ExtDType::<TestExt>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap()
        .erased()
    }

    #[test]
    fn test_filter_pushdown() {
        let ext_dtype = test_ext_dtype();
        let storage = buffer![1i64, 2, 3, 4, 5].into_array();
        let ext_array = ExtensionArray::new(ext_dtype.clone(), storage).into_array();

        // Create a filter that selects elements at indices 0, 2, 4
        let mask = Mask::from_iter([true, false, true, false, true]);
        let filter_array = FilterArray::new(ext_array, mask).into_array();

        // Optimize should push the filter into the storage
        let optimized = filter_array.optimize().unwrap();

        // The result should be an ExtensionArray, not a FilterArray
        assert!(
            optimized.as_opt::<ExtensionVTable>().is_some(),
            "Expected ExtensionArray after optimization, got {}",
            optimized.encoding_id()
        );

        let ext_result = optimized.as_::<ExtensionVTable>();
        assert_eq!(ext_result.len(), 3);
        assert_eq!(ext_result.ext_dtype(), &ext_dtype);

        // Check the storage values
        let storage_result: &[i64] = &ext_result.storage().to_primitive().to_buffer::<i64>();
        assert_eq!(storage_result, &[1, 3, 5]);
    }

    #[test]
    fn test_filter_pushdown_nullable() {
        let ext_dtype = ExtDType::<TestExt>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .unwrap()
        .erased();
        let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3), Some(4), None])
            .into_array();
        let ext_array = ExtensionArray::new(ext_dtype, storage).into_array();

        let mask = Mask::from_iter([true, true, false, false, true]);
        let filter_array = FilterArray::new(ext_array, mask).into_array();

        let optimized = filter_array.optimize().unwrap();

        assert!(optimized.as_opt::<ExtensionVTable>().is_some());
        let ext_result = optimized.as_::<ExtensionVTable>();
        assert_eq!(ext_result.len(), 3);

        // Check values: should be [Some(1), None, None]
        let canonical = ext_result.storage().to_primitive();
        assert_eq!(canonical.len(), 3);
    }

    #[test]
    fn test_scalar_fn_no_pushdown_different_ext_types() {
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        struct TestExt2;
        impl ExtDTypeVTable for TestExt2 {
            type Metadata = EmptyMetadata;

            fn id(&self) -> ExtID {
                ExtID::new_ref("test_ext_2")
            }

            fn validate_dtype(
                &self,
                _options: &Self::Metadata,
                _storage_dtype: &DType,
            ) -> VortexResult<()> {
                Ok(())
            }
        }

        let ext_dtype1 = ExtDType::<TestExt>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )
        .unwrap()
        .erased();

        let storage = buffer![10i64, 20, 30].into_array();
        let ext_array = ExtensionArray::new(ext_dtype1, storage).into_array();

        // Create constant with different extension type
        let const_scalar = Scalar::extension::<TestExt2>(EmptyMetadata, Scalar::from(25i64));
        let const_array = ConstantArray::new(const_scalar, 3).into_array();

        let scalar_fn_array = Binary
            .try_new_array(3, Operator::Lt, [ext_array.clone(), const_array])
            .unwrap();

        let optimized = scalar_fn_array.optimize().unwrap();

        // The first child should still be an ExtensionArray (no pushdown happened)
        let scalar_fn = optimized.as_opt::<crate::arrays::ScalarFnVTable>().unwrap();
        assert!(
            scalar_fn.children()[0]
                .as_opt::<ExtensionVTable>()
                .is_some(),
            "Expected first child to remain ExtensionArray when ext types differ"
        );
    }

    #[test]
    fn test_scalar_fn_no_pushdown_non_constant_sibling() {
        let ext_dtype = test_ext_dtype();

        let storage1 = buffer![10i64, 20, 30].into_array();
        let ext_array1 = ExtensionArray::new(ext_dtype.clone(), storage1).into_array();

        let storage2 = buffer![15i64, 25, 35].into_array();
        let ext_array2 = ExtensionArray::new(ext_dtype, storage2).into_array();

        // Both children are extension arrays (not constants)
        let scalar_fn_array = Binary
            .try_new_array(3, Operator::Lt, [ext_array1.clone(), ext_array2])
            .unwrap();

        let optimized = scalar_fn_array.optimize().unwrap();

        // No pushdown should happen because sibling is not a constant
        let scalar_fn = optimized.as_opt::<crate::arrays::ScalarFnVTable>().unwrap();
        assert!(
            scalar_fn.children()[0]
                .as_opt::<ExtensionVTable>()
                .is_some(),
            "Expected first child to remain ExtensionArray when sibling is not constant"
        );
    }

    #[test]
    fn test_scalar_fn_no_pushdown_non_extension_constant() {
        let ext_dtype = test_ext_dtype();
        let storage = buffer![10i64, 20, 30].into_array();
        let ext_array = ExtensionArray::new(ext_dtype, storage).into_array();

        // Create a non-extension constant (plain primitive)
        let const_array = ConstantArray::new(Scalar::from(25i64), 3).into_array();

        let scalar_fn_array = Binary
            .try_new_array(3, Operator::Lt, [ext_array.clone(), const_array])
            .unwrap();

        let optimized = scalar_fn_array.optimize().unwrap();

        // No pushdown should happen because constant is not an extension scalar
        let scalar_fn = optimized.as_opt::<crate::arrays::ScalarFnVTable>().unwrap();
        assert!(
            scalar_fn.children()[0]
                .as_opt::<ExtensionVTable>()
                .is_some(),
            "Expected first child to remain ExtensionArray when constant is not extension"
        );
    }
}
