use log::info;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::{Array, ArrayDType as _, IntoCanonical as _};

pub trait TakeFn {
    fn take(&self, indices: &Array) -> VortexResult<Array>;
}

pub fn take(array: impl AsRef<Array>, indices: impl AsRef<Array>) -> VortexResult<Array> {
    let array = array.as_ref();
    let indices = indices.as_ref();

    if !indices.dtype().is_int() || indices.dtype().is_nullable() {
        vortex_bail!(
            "Take indices must be a non-nullable integer type, got {}",
            indices.dtype()
        );
    }

    array.with_dyn(|a| {
        if let Some(take) = a.take() {
            return take.take(indices);
        }

        // Otherwise, flatten and try again.
        info!("TakeFn not implemented for {}, flattening", array);
        Array::from(array.clone().into_canonical()?).with_dyn(|a| {
            a.take()
                .map(|t| t.take(indices))
                .unwrap_or_else(|| Err(vortex_err!(NotImplemented: "take", array.encoding().id())))
        })
    })
}
