use arrow_array::{Array, ArrayRef, Datum as ArrowDatum};
use vortex_error::{vortex_bail, VortexError, VortexResult};

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

/// Convert an Arrow array to an ArrayData with a specific length.
/// This is useful for compute functions that delegate to Arrow using [Datum],
/// which will return a scalar (length 1 Arrow array) if the input array is constant.
pub(crate) fn to_array_data_with_len<A>(
    array: A,
    len: usize,
    nullable: bool,
) -> VortexResult<ArrayData>
where
    ArrayData: FromArrowArray<A>,
{
    let array = ArrayData::from_arrow(array, nullable);
    if array.len() == len {
        return Ok(array);
    }

    if array.len() != 1 {
        vortex_bail!(
            "Array length mismatch, expected {} got {} for encoding {}",
            len,
            array.len(),
            array.encoding().id()
        );
    }

    Ok(ConstantArray::new(scalar_at(&array, 0)?, len).into_array())
}
