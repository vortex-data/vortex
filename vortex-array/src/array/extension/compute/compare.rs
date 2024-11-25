use vortex_error::VortexResult;
use vortex_scalar::ExtScalar;

use crate::array::{ConstantArray, ExtensionArray, ExtensionEncoding};
use crate::compute::{compare, CompareFn, Operator};
use crate::encoding::EncodingVTable;
use crate::{ArrayData, ArrayLen};

impl CompareFn<ExtensionArray> for ExtensionEncoding {
    fn compare(
        &self,
        lhs: &ExtensionArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = rhs.as_constant() {
            let storage_scalar = ExtScalar::try_from(&const_ext)?.storage();
            return compare(
                lhs.storage(),
                ConstantArray::new(storage_scalar, lhs.len()),
                operator,
            )
            .map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if rhs.is_encoding(ExtensionEncoding.id()) {
            let rhs_ext = ExtensionArray::try_from(rhs.clone())?;
            return compare(lhs.storage(), rhs_ext.storage(), operator).map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}
