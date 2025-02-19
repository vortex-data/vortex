use log::debug;
use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::builders::ArrayBuilder;
use crate::encoding::Encoding;
use crate::{ArrayRef, Canonical, IntoArray};

/// Encoding VTable for canonicalizing an array.
#[allow(clippy::wrong_self_convention)]
pub trait CanonicalVTable<Array> {
    fn into_canonical(&self, array: Array) -> VortexResult<Canonical>;

    fn canonicalize_into(&self, array: Array, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        let canonical = self.into_canonical(array)?;
        debug!("default impl canonicalize_into {}", canonical.encoding());
        builder.extend_from_array(canonical.into_array())
    }
}

impl<E: Encoding> CanonicalVTable<ArrayRef> for E
where
    E: CanonicalVTable<E::Array>,
    E::Array: TryFrom<ArrayRef, Error = VortexError>,
{
    fn into_canonical(&self, data: ArrayRef) -> VortexResult<Canonical> {
        let encoding = data.vtable().clone();
        CanonicalVTable::into_canonical(
            encoding
                .as_any()
                .downcast_ref::<E>()
                .vortex_expect("Failed to downcast encoding"),
            E::Array::try_from(data)?,
        )
    }

    fn canonicalize_into(&self, array: ArrayRef, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        let encoding = array.vtable().clone();
        CanonicalVTable::canonicalize_into(
            encoding
                .as_any()
                .downcast_ref::<E>()
                .vortex_expect("Failed to downcast encoding"),
            E::Array::try_from(array)?,
            builder,
        )
    }
}
