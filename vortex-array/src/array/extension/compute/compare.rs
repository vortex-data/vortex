use vortex_error::VortexResult;
use vortex_scalar::{ExtScalar, Scalar};

use crate::array::{ConstantArray, ExtensionArray, ExtensionEncoding};
use crate::compute::{compare, CompareFn, Operator};
use crate::encoding::EncodingVTable;
use crate::{ArrayDType, ArrayData, ArrayLen};

impl CompareFn for ExtensionArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> VortexResult<Option<ArrayData>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = other.as_constant() {
            let scalar_ext = ExtScalar::try_new(const_ext.dtype(), const_ext.value())?;
            let storage_scalar = ConstantArray::new(
                Scalar::new(self.storage().dtype().clone(), scalar_ext.value().clone()),
                self.len(),
            );

            return compare(self.storage(), storage_scalar, operator).map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if other.is_encoding(ExtensionEncoding.id()) {
            let rhs_ext = ExtensionArray::try_from(other.clone())?;
            return compare(self.storage(), rhs_ext.storage(), operator).map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}
