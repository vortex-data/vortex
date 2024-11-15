use arrow_array::cast::AsArray;
use vortex_dtype::{DType, Nullability};
use vortex_error::{vortex_bail, VortexResult};

use crate::arrow::FromArrowArray;
use crate::{ArrayDType, ArrayData, IntoCanonical};

pub trait FilterFn {
    /// Filter an array by the provided predicate.
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData>;
}

/// Return a new array by applying a boolean predicate to select items from a base Array.
///
/// # Performance
///
/// This function attempts to amortize the cost of copying
///
/// # Panics
///
/// The `predicate` must receive an Array with type non-nullable bool, and will panic if this is
/// not the case.
pub fn filter(
    array: impl AsRef<ArrayData>,
    predicate: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    let array = array.as_ref();
    let predicate = predicate.as_ref();

    if predicate.dtype() != &DType::Bool(Nullability::NonNullable) {
        vortex_bail!(
            "predicate must be non-nullable bool, has dtype {}",
            predicate.dtype(),
        );
    }
    if predicate.len() != array.len() {
        vortex_bail!(
            "predicate.len() is {}, does not equal array.len() of {}",
            predicate.len(),
            array.len()
        );
    }

    array.with_dyn(|a| {
        if let Some(filter_fn) = a.filter() {
            filter_fn.filter(predicate)
        } else {
            // Fallback: implement using Arrow kernels.
            let array_ref = array.clone().into_canonical()?.into_arrow()?;
            let predicate_ref = predicate.clone().into_canonical()?.into_arrow()?;
            let filtered =
                arrow_select::filter::filter(array_ref.as_ref(), predicate_ref.as_boolean())?;

            Ok(ArrayData::from_arrow(filtered, array.dtype().is_nullable()))
        }
    })
}

#[cfg(test)]
mod test {
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::compute::filter::filter;
    use crate::validity::Validity;
    use crate::{IntoArrayData, IntoCanonical};

    #[test]
    fn test_filter() {
        let items =
            PrimitiveArray::from_nullable_vec(vec![Some(0i32), None, Some(1i32), None, Some(2i32)])
                .into_array();
        let predicate =
            BoolArray::from_vec(vec![true, false, true, false, true], Validity::NonNullable)
                .into_array();

        let filtered = filter(&items, &predicate).unwrap();
        assert_eq!(
            filtered
                .into_canonical()
                .unwrap()
                .into_primitive()
                .unwrap()
                .into_maybe_null_slice::<i32>(),
            vec![0i32, 1i32, 2i32]
        );
    }
}
