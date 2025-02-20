use arrow_array::Array;
use vortex_error::VortexResult;

use crate::ArrayRef;

/// Trait for converting a type into a Vortex [`ArrayRef`].
pub trait IntoArray {
    fn into_array(self) -> ArrayRef;
}

/// Trait for converting a type into a Vortex [`ArrayRef`], returning an error if the conversion fails.
pub trait TryIntoArray {
    fn try_into_array(self) -> VortexResult<ArrayRef>;
}

/// Trait for converting a type from a Vortex [`ArrayRef`], returning an error if the conversion fails.
pub trait TryFromArray: Sized {
    fn try_from_array(array: &dyn Array) -> VortexResult<Self>;
}

/// Trait for converting a type from a Vortex [`ArrayRef`], returning an error if the conversion fails.
pub trait TryFromArrayRef: Sized {
    fn try_from_array(array: ArrayRef) -> VortexResult<Self>;
}
