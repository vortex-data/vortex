use arrow_array::{Array, ArrayRef, Datum as ArrowDatum};
use vortex_error::VortexError;

use crate::compute::slice;
use crate::stats::{ArrayStatistics, Stat};
use crate::{ArrayData, IntoCanonical};

/// A wrapper around a generic Arrow array that can be used as a Datum in Arrow compute.
pub struct Datum {
    array: ArrayRef,
    is_scalar: bool,
}

impl TryFrom<ArrayData> for Datum {
    type Error = VortexError;

    fn try_from(array: ArrayData) -> Result<Self, Self::Error> {
        if array
            .statistics()
            .get_as::<bool>(Stat::IsConstant)
            .unwrap_or_default()
        {
            Ok(Self {
                array: slice(array, 0, 1)?.into_canonical()?.into_arrow()?,
                is_scalar: true,
            })
        } else {
            Ok(Self {
                array: array.into_canonical()?.into_arrow()?,
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
