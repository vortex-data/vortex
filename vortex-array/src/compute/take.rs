use log::info;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::stats::{ArrayStatistics, Stat};
use crate::{ArrayDType as _, ArrayData, IntoCanonical as _};

#[derive(Default, Debug, Clone, Copy)]
pub struct TakeOptions {
    pub skip_bounds_check: bool,
}

pub trait TakeFn {
    fn take(&self, indices: &ArrayData, options: TakeOptions) -> VortexResult<ArrayData>;
}

pub fn take(
    array: impl AsRef<ArrayData>,
    indices: impl AsRef<ArrayData>,
    mut options: TakeOptions,
) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    let indices = indices.as_ref();

    if !indices.dtype().is_int() || indices.dtype().is_nullable() {
        vortex_bail!(
            "Take indices must be a non-nullable integer type, got {}",
            indices.dtype()
        );
    }

    // If the indices are all within bounds, we can skip bounds checking.
    if indices
        .statistics()
        .get_as::<usize>(Stat::Max)
        .is_some_and(|max| max < array.len())
    {
        options.skip_bounds_check = true;
    }

    // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
    //  such that canonicalize does less work.

    array.with_dyn(|a| {
        if let Some(take) = a.take() {
            return take.take(indices, options);
        }

        // Otherwise, flatten and try again.
        info!("TakeFn not implemented for {}, flattening", array);
        ArrayData::from(array.clone().into_canonical()?).with_dyn(|a| {
            a.take()
                .map(|t| t.take(indices, options))
                .unwrap_or_else(|| Err(vortex_err!(NotImplemented: "take", array.encoding().id())))
        })
    })
}
