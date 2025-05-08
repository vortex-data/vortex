use arrow_array::{Array as ArrowArray, ArrayRef as ArrowArrayRef, Datum as ArrowDatum};
use arrow_schema::DataType;
use vortex_error::{VortexResult, vortex_panic};

use crate::arrays::ConstantArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::{Array, ArrayRef};

/// A wrapper around a generic Arrow array that can be used as a Datum in Arrow compute.
#[derive(Debug)]
pub struct Datum {
    array: ArrowArrayRef,
    is_scalar: bool,
}

impl Datum {
    /// Create a new [`Datum`] from an [`ArrayRef`], which can then be passed to Arrow compute.
    pub fn try_new(array: &dyn Array) -> VortexResult<Self> {
        if array.is_constant() {
            Ok(Self {
                array: array.slice(0, 1)?.into_arrow_preferred()?,
                is_scalar: true,
            })
        } else {
            Ok(Self {
                array: array.to_array().into_arrow_preferred()?,
                is_scalar: false,
            })
        }
    }

    pub fn with_target_datatype(
        array: &dyn Array,
        target_datatype: &DataType,
    ) -> VortexResult<Self> {
        if array.is_constant() {
            Ok(Self {
                array: array.slice(0, 1)?.into_arrow(target_datatype)?,
                is_scalar: true,
            })
        } else {
            Ok(Self {
                array: array.to_array().into_arrow(target_datatype)?,
                is_scalar: false,
            })
        }
    }
}

impl ArrowDatum for Datum {
    fn get(&self) -> (&dyn ArrowArray, bool) {
        (&self.array, self.is_scalar)
    }
}

/// Convert an Arrow array to an Array with a specific length.
/// This is useful for compute functions that delegate to Arrow using [Datum],
/// which will return a scalar (length 1 Arrow array) if the input array is constant.
///
/// Panics if the length of the array is not 1 and also not equal to the expected length.
pub fn from_arrow_array_with_len<A>(array: A, len: usize, nullable: bool) -> VortexResult<ArrayRef>
where
    ArrayRef: FromArrowArray<A>,
{
    let array = ArrayRef::from_arrow(array, nullable);
    if array.len() == len {
        return Ok(array);
    }

    if array.len() != 1 {
        vortex_panic!(
            "Array length mismatch, expected {} got {} for encoding {}",
            len,
            array.len(),
            array.encoding()
        );
    }

    Ok(ConstantArray::new(array.scalar_at(0)?, len).into_array())
}
