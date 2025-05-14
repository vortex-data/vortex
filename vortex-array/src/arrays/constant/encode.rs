use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::{ConstantArray, ConstantEncoding, ConstantVTable};
use crate::vtable::EncodeVTable;

impl EncodeVTable<ConstantVTable> for ConstantVTable {
    fn encode(
        _encoding: &ConstantEncoding,
        canonical: &Canonical,
        _like: Option<&ConstantArray>,
    ) -> VortexResult<Option<ConstantArray>> {
        let canonical = canonical.as_ref();
        if canonical.is_constant() {
            let scalar = canonical.scalar_at(0)?;
            Ok(Some(ConstantArray::new(scalar, canonical.len())))
        } else {
            Ok(None)
        }
    }
}
