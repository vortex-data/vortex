use vortex_array::compute::{TakeFromFn, take};
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{RunEndArray, RunEndEncoding};

impl TakeFromFn<&RunEndArray> for RunEndEncoding {
    /// Takes values from the source array using run-end encoded indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Run-end encoded indices
    /// * `array` - Array to take values from
    ///
    /// # Returns
    ///
    /// * `Ok(Some(array))` - If successful
    /// * `Ok(None)` - If the source array has an unsupported dtype
    ///
    fn take_from(
        &self,
        indices: &RunEndArray,
        array: &dyn Array,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only `Primitive` and `Bool` are valid run-end value types. - TODO: Support additional DTypes
        if !matches!(array.dtype(), DType::Primitive(_, _) | DType::Bool(_)) {
            return Ok(None);
        }

        // Transform the run-end encoding from storing indices to storing values
        // by taking values from `array` at positions specified in `indices.values()`.
        let transformed = take(array, indices.values())?;

        // Create a new run-end array now containing the values instead of indices.
        let ree_array = RunEndArray::with_offset_and_length(
            indices.ends().clone(),
            transformed,
            indices.offset(),
            indices.len(),
        )?;

        Ok(Some(ree_array.into_array()))
    }
}
