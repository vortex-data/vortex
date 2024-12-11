use log::info;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::stats::{ArrayStatistics, Stat};
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

pub trait TakeFn<Array> {
    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Panics
    ///
    /// Using `indices` that are invalid for the given `array` will cause a panic.
    fn take(&self, array: &Array, indices: &ArrayData) -> VortexResult<ArrayData>;

    /// Create a new array by taking the values from the `array` at the
    /// given `indices`.
    ///
    /// # Safety
    ///
    /// This take variant will not perform bounds checking on indices, so it is the caller's
    /// responsibility to ensure that the `indices` are all valid for the provided `array`.
    /// Failure to do so could result in out of bounds memory access or UB.
    unsafe fn take_unchecked(&self, array: &Array, indices: &ArrayData) -> VortexResult<ArrayData> {
        self.take(array, indices)
    }
}

impl<E: Encoding> TakeFn<ArrayData> for E
where
    E: TakeFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn take(&self, array: &ArrayData, indices: &ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        TakeFn::take(encoding, array_ref, indices)
    }
}

pub fn take(
    array: impl AsRef<ArrayData>,
    indices: impl AsRef<ArrayData>,
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
    let checked_indices = indices
        .statistics()
        .get_as::<usize>(Stat::Max)
        .is_some_and(|max| max < array.len());

    // TODO(ngates): if indices min is quite high, we could slice self and offset the indices
    //  such that canonicalize does less work.

    // If TakeFn defined for the encoding, delegate to TakeFn.
    // If we know from stats that indices are all valid, we can avoid all bounds checks.
    if let Some(take_fn) = array.encoding().take_fn() {
        return if checked_indices {
            // SAFETY: indices are all inbounds per stats.
            // TODO(aduffy): this means stats must be trusted, can still trigger UB if stats are bad.
            unsafe { take_fn.take_unchecked(array, indices) }
        } else {
            take_fn.take(array, indices)
        };
    }

    // Otherwise, flatten and try again.
    info!("TakeFn not implemented for {}, flattening", array);
    let canonical = array.clone().into_canonical()?.into_array();
    let canonical_take_fn = canonical
        .encoding()
        .take_fn()
        .ok_or_else(|| vortex_err!(NotImplemented: "take", canonical.encoding().id()))?;

    if checked_indices {
        // SAFETY: indices are known to be in-bound from stats
        unsafe { canonical_take_fn.take_unchecked(&canonical, indices) }
    } else {
        canonical_take_fn.take(&canonical, indices)
    }
}
