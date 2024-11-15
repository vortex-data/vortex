use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::ArrayData;

/// Limit array to start...stop range
pub trait SliceFn {
    /// Return a zero-copy slice of an array, between `start` (inclusive) and `end` (exclusive).
    /// If start >= stop, returns an empty array of the same type as `self`.
    /// Assumes that start or stop are out of bounds, may panic otherwise.
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData>;
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

    array.with_dyn(|c| {
        c.slice().map(|t| t.slice(start, stop)).unwrap_or_else(|| {
            Err(vortex_err!(
                NotImplemented: "slice",
                array.encoding().id()
            ))
        })
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
