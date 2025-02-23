use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;

/// Implementation trait for validity functions.
///
/// These functions should not be called directly, rather their equivalents on the base
/// [`Array`] trait should be used.
pub trait ArrayValidityImpl {
    /// Returns whether the `index` item is valid.
    ///
    /// ## Pre-conditions
    /// - `index` is less than the length of the array.
    fn _is_valid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether the array is all valid.
    fn _all_valid(&self) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    fn _all_invalid(&self) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn _valid_count(&self) -> VortexResult<usize> {
        Ok(self._validity_mask()?.true_count())
    }

    /// Returns the number of invalid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn _invalid_count(&self) -> VortexResult<usize> {
        Ok(self._validity_mask()?.false_count())
    }

    /// Returns the canonical validity mask for the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn _validity_mask(&self) -> VortexResult<Mask>;
}
