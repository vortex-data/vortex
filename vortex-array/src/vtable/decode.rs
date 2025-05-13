use vortex_error::VortexResult;

use crate::Canonical;
use crate::builders::ArrayBuilder;
use crate::vtable::VTable;

// TODO(ngates): rename to `DecodeVTable`.
pub trait CanonicalVTable<V: VTable> {
    /// Returns the canonical representation of the array.
    ///
    /// ## Post-conditions
    /// - The length is equal to that of the input array.
    /// - The [`vortex_dtype::DType`] is equal to that of the input array.
    // TODO(ngates): rename to `decode`
    fn canonicalize(array: &V::Array) -> VortexResult<Canonical>;

    /// Writes the array into a canonical builder.
    ///
    /// ## Post-conditions
    /// - The length of the builder is incremented by the length of the input array.
    fn append_to_builder(array: &V::Array, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        let canonical = Self::canonicalize(array)?;
        builder.extend_from_array(canonical.as_ref())
    }
}
