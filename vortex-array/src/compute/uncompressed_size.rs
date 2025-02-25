use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};

use crate::{Array, Encoding};

pub trait UncompressedSizeFn<A> {
    /// Compute the approximated uncompressed size of the array, in bytes.
    fn uncompressed_size(&self, array: A) -> VortexResult<usize>;
}

impl<E: Encoding> UncompressedSizeFn<&dyn Array> for E
where
    E: for<'a> UncompressedSizeFn<&'a E::Array>,
{
    fn uncompressed_size(&self, array: &dyn Array) -> VortexResult<usize> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        UncompressedSizeFn::uncompressed_size(self, array_ref)
    }
}

pub fn uncompressed_size(array: &dyn Array) -> VortexResult<usize> {
    match array.vtable().uncompressed_size_fn() {
        Some(size_fn) => size_fn.uncompressed_size(array),
        None => {
            log::debug!(
                "No uncompressed_size implementation found for {}",
                array.encoding()
            );
            let array = array.to_canonical()?;
            let array_ref = array.as_ref();
            if let Some(size_fn) = array_ref.vtable().uncompressed_size_fn() {
                size_fn.uncompressed_size(array_ref)
            } else {
                vortex_bail!(
                    "No uncompressed_size function for canonical array: {}",
                    array.as_ref().encoding(),
                )
            }
        }
    }
}
