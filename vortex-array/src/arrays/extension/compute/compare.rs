use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ExtensionArray, ExtensionEncoding};
use crate::compute::{CompareFn, Operator, compare};
use crate::{Array, ArrayRef};

impl CompareFn<&ExtensionArray> for ExtensionEncoding {
    fn compare(
        &self,
        lhs: &ExtensionArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = rhs.as_constant() {
            let storage_scalar = const_ext.as_extension().storage();
            return compare(
                lhs.storage(),
                &ConstantArray::new(storage_scalar, lhs.len()),
                operator,
            )
            .map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if let Some(rhs_ext) = rhs.as_extension_typed() {
            return compare(lhs.storage(), &rhs_ext.storage_data(), operator).map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}
