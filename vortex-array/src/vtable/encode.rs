use vortex_error::VortexResult;

use crate::vtable::{NotSupported, VTable};
use crate::{Array, Canonical};

pub trait EncodeVTable<V: VTable> {
    /// Encode a canonical array into this encoding using the given `like` array as a template.
    ///
    /// If this encoding does not support encoding the given array, then `None` is returned.
    // TODO(ngates): pass some abstract `dyn EncodeOptions`?
    fn encode(
        encoding: &V::Encoding,
        canonical: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<V::Array>>;
}

/// Default implementation for encodings that do not support encoding.
impl<V: VTable> EncodeVTable<V> for NotSupported {
    fn encode(
        _encoding: &V::Encoding,
        _canonical: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<V::Array>> {
        Ok(None)
    }
}
