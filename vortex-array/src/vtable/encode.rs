use vortex_error::VortexResult;

use crate::Canonical;
use crate::vtable::{NotSupported, VTable};

pub trait EncodeVTable<V: VTable> {
    /// Try to encode a canonical array into this encoding.
    ///
    /// The given `like` array is passed as a template, for example if the caller knows that
    /// this encoding was successfully used previously for a similar array.
    ///
    /// If the encoding does not support the given array (e.g. [`crate::arrays::ConstantEncoding`]
    /// was passed a non-constant array), then `None` is returned.
    fn encode(
        encoding: &V::Encoding,
        canonical: &Canonical,
        like: Option<&V::Array>,
    ) -> VortexResult<Option<V::Array>>;
}

/// Default implementation for encodings that do not support encoding.
impl<V: VTable> EncodeVTable<V> for NotSupported {
    fn encode(
        _encoding: &V::Encoding,
        _canonical: &Canonical,
        _like: Option<&V::Array>,
    ) -> VortexResult<Option<V::Array>> {
        Ok(None)
    }
}
