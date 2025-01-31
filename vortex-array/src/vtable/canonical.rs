use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::{Array, Canonical};

/// Encoding VTable for canonicalizing an array.
#[allow(clippy::wrong_self_convention)]
pub trait CanonicalVTable<Array> {
    fn into_canonical(&self, array: Array) -> VortexResult<Canonical>;
}

impl<E: Encoding> CanonicalVTable<Array> for E
where
    E: CanonicalVTable<E::Array>,
    E::Array: TryFrom<Array, Error = VortexError>,
{
    fn into_canonical(&self, data: Array) -> VortexResult<Canonical> {
        let encoding = data.vtable().clone();
        CanonicalVTable::into_canonical(
            encoding
                .as_any()
                .downcast_ref::<E>()
                .vortex_expect("Failed to downcast encoding"),
            E::Array::try_from(data)?,
        )
    }
}
