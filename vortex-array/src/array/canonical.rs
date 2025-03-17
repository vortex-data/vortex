use vortex_error::VortexResult;

use crate::Canonical;
use crate::builders::ArrayBuilder;

/// Implementation trait for canonicalization functions.
///
/// These functions should not be called directly, rather their equivalents on the base
/// [`crate::Array`] trait should be used.
pub trait ArrayCanonicalImpl {
    /// Returns the canonical representation of the array.
    ///
    /// ## Post-conditions
    /// - The length is equal to that of the input array.
    /// - The [`vortex_dtype::DType`] is equal to that of the input array.
    fn _to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// ## Post-conditions
    /// - The length of the builder is incremented by the length of the input array.
    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        let canonical = self._to_canonical()?;
        builder.extend_from_array(canonical.as_ref())
    }
}
