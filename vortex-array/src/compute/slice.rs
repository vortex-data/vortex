use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::array::ConstantArray;
use crate::encoding::Encoding;
use crate::{ArrayData, IntoArrayData};

/// Limit array to start...stop range
pub trait SliceFn<Array> {
    /// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
    /// If start >= stop, returns an empty array of the same type as `self`.
    /// Assumes that start or stop are out of bounds, may panic otherwise.
    fn slice(&self, array: &Array, start: usize, stop: usize) -> VortexResult<ArrayData>;
}

impl<E: Encoding> SliceFn<ArrayData> for E
where
    E: SliceFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn slice(&self, array: &ArrayData, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        SliceFn::slice(encoding, array_ref, start, stop)
    }
}

/// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
///
/// # Errors
///
/// Slicing returns an error if you attempt to slice a range that exceeds the bounds of the
/// underlying array.
///
/// Slicing returns an error if the underlying codec's [slice](SliceFn::slice()) implementation
/// returns an error.
pub fn slice(array: impl AsRef<ArrayData>, start: usize, stop: usize) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    check_slice_bounds(array, start, stop)?;

    if let Some(const_scalar) = array.as_constant() {
        return Ok(ConstantArray::new(const_scalar, stop - start).into_array());
    }

    array
        .encoding()
        .slice_fn()
        .map(|f| f.slice(array, start, stop))
        .unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "slice",
                array.encoding().id()
            ))
        })
}

fn check_slice_bounds(array: &ArrayData, start: usize, stop: usize) -> VortexResult<()> {
    if start > array.len() {
        vortex_bail!(OutOfBounds: start, 0, array.len());
    }
    if stop > array.len() {
        vortex_bail!(OutOfBounds: stop, 0, array.len());
    }
    if start > stop {
        vortex_bail!("start ({start}) must be <= stop ({stop})");
    }
    Ok(())
}
