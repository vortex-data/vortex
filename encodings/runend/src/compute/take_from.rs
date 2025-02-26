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
    /// * `indices` - The run-end encoded array containing the indices
    /// * `array` - The source array to take values from
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
        // Only `Primitive` and `Bool` are valid run-end value types.
        if !matches!(array.dtype(), DType::Primitive(_, _) | DType::Bool(_)) {
            return Ok(None);
        }

        // Order the values to prepare for runend decoding.
        let shuffled = take(array, indices.values())?;

        let ree_array = RunEndArray::with_offset_and_length(
            indices.ends().clone(),
            shuffled,
            indices.offset(),
            indices.len(),
        )?;

        Ok(Some(ree_array.into_array()))
    }
}
