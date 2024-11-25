use log::info;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{ArrayStatistics, Stat};
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

#[derive(Default, Debug, Clone, Copy)]
pub struct TakeOptions {
    pub skip_bounds_check: bool,
}

pub trait TakeFn<Array> {
    fn take(
        &self,
        array: &Array,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData>;
}

impl<E: Encoding> TakeFn<ArrayData> for E
where
    E: TakeFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn take(
        &self,
        array: &ArrayData,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        TakeFn::take(encoding, array_ref, indices, options)
    }
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

    // TODO(ngates): if indices are sorted and unique (strict-sorted), then we should delegate to
    //  the filter function since they're typically optimised for this case.

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

    if let Some(take_fn) = array.encoding().take_fn() {
        return take_fn.take(array, indices, options);
    }

    // Otherwise, flatten and try again.
    info!("TakeFn not implemented for {}, flattening", array);
    let canonical = array.clone().into_canonical()?.into_array();
    canonical
        .encoding()
        .take_fn()
        .ok_or_else(|| vortex_err!(NotImplemented: "take", canonical.encoding().id()))?
        .take(&canonical, indices, options)
}
