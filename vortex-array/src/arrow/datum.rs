use arrow_array::{Array, ArrayRef, Datum as ArrowDatum};
use vortex_error::{vortex_panic, VortexResult};

use crate::array::ConstantArray;
use crate::arrow::FromArrowArray;
use crate::compute::{scalar_at, slice};
use crate::{ArrayData, IntoArrayData, IntoCanonical};

/// A wrapper around a generic Arrow array that can be used as a Datum in Arrow compute.
#[derive(Debug)]
pub struct Datum {
    array: ArrayRef,
    is_scalar: bool,
}

impl Datum {
    /// Create a new [`Datum`] from an [`ArrayData`], which can then be passed to Arrow compute.
    /// This is unsafe because it does not preserve the length of the array.
    ///
    /// # Safety
    /// The caller must ensure that the length of the array is preserved, and when processing
    /// the result of the Arrow compute, must check whether the result is a scalar (Arrow array of length 1),
    /// in which case it likely must be expanded to match the length of the original array.
    ///
    /// The utility function [`from_arrow_array_with_len`] can be used to ensure that the length of the
    /// result of the Arrow compute matches the length of the original array.
    pub unsafe fn try_new(array: ArrayData) -> VortexResult<Self> {
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

/// Convert an Arrow array to an ArrayData with a specific length.
/// This is useful for compute functions that delegate to Arrow using [Datum],
/// which will return a scalar (length 1 Arrow array) if the input array is constant.
///
/// Panics if the length of the array is not 1 and also not equal to the expected length.
pub fn from_arrow_array_with_len<A>(array: A, len: usize, nullable: bool) -> VortexResult<ArrayData>
where
    ArrayData: FromArrowArray<A>,
{
    let array = ArrayData::from_arrow(array, nullable);
    if array.len() == len {
        return Ok(array);
    }

    if array.len() != 1 {
        vortex_panic!(
            "Array length mismatch, expected {} got {} for encoding {}",
            len,
            array.len(),
            array.encoding().id()
        );
    }

    Ok(ConstantArray::new(scalar_at(&array, 0)?, len).into_array())
}
