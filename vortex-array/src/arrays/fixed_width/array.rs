// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::validity::Validity;

/// Type-specific access needed by shared fixed-width structural compute.
///
/// The shared `take` and `filter` kernels view an implementing array as `array.len()` records of
/// [`byte_width`] bytes each, which lets them move whole records without knowing the logical
/// type.
///
/// [`byte_width`]: FixedWidthArray::byte_width
pub(crate) trait FixedWidthArray: VTable {
    /// Returns the number of bytes each record occupies.
    fn byte_width(array: ArrayView<'_, Self>) -> usize;

    /// Returns the records of `array` as a single host-resident byte buffer.
    ///
    /// The returned buffer must contain exactly `array.len() * byte_width` bytes.
    fn values(array: ArrayView<'_, Self>) -> ByteBuffer;

    /// Rebuilds an array of this encoding from a records buffer, preserving the logical type of
    /// `array`.
    ///
    /// Callers must provide a `values` buffer of exactly `len * byte_width` bytes and a
    /// `validity` with logical length `len`. The shared kernels enforce the buffer length through
    /// the module-level `with_values` helper rather than in each implementation.
    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Array<Self>>;
}

/// Rebuilds a fixed-width array from `len` records in `values`, validating the buffer length
/// before dispatching to [`FixedWidthArray::with_values`].
pub(crate) fn with_values<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    values: ByteBuffer,
    len: usize,
    validity: Validity,
) -> VortexResult<Array<V>> {
    let expected_len = len
        .checked_mul(V::byte_width(array))
        .ok_or_else(|| vortex_err!("Fixed-width values buffer length overflows usize"))?;
    vortex_ensure_eq!(
        values.len(),
        expected_len,
        "Fixed-width values buffer length {} does not match expected length {expected_len}",
        values.len(),
    );
    V::with_values(array, values, len, validity)
}
