use vortex_error::VortexResult;

use crate::array::{ConstantArray, ExtensionArray, ExtensionEncoding};
use crate::compute::{compare, CompareFn, Operator};
use crate::Array;

impl CompareFn<ExtensionArray> for ExtensionEncoding {
    fn compare(
        &self,
        lhs: &ExtensionArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = rhs.as_constant() {
            let storage_scalar = const_ext.as_extension().storage();
            return compare(
                lhs.storage(),
                ConstantArray::new(storage_scalar, lhs.len()),
                operator,
            )
            .map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if let Some(rhs_ext) = ExtensionArray::maybe_from(rhs) {
            return compare(lhs.storage(), rhs_ext.storage(), operator).map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}
