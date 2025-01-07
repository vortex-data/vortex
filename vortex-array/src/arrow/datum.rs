use arrow_array::{Array, ArrayRef, Datum as ArrowDatum};
use vortex_error::VortexError;

use crate::compute::slice;
use crate::{ArrayData, IntoCanonical};

/// A wrapper around a generic Arrow array that can be used as a Datum in Arrow compute.
#[derive(Debug)]
pub struct Datum {
    array: ArrayRef,
    is_scalar: bool,
}

impl TryFrom<ArrayData> for Datum {
    type Error = VortexError;

    fn try_from(array: ArrayData) -> Result<Self, Self::Error> {
        if array.is_constant() {
            Ok(Self {
                array: slice(array, 0, 1)?.into_arrow()?,
                is_scalar: true,
            })
        } else {
            Ok(Self {
                array: array.into_arrow()?,
                is_scalar: false,
            })
        }
    }
}

impl ArrowDatum for Datum {
    fn get(&self) -> (&dyn Array, bool) {
        (&self.array, self.is_scalar)
    }
}
