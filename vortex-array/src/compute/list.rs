//! List-related compute operations.

use arrow_buffer::BooleanBuffer;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ListArray};
use crate::compute::{Operator, compare};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

/// Get a `Mask` representing the positions in `array` that contains the scalar `value`.
///
/// ## Example
///
/// ```rust
/// use vortex_array::{Array, IntoArray};
/// use vortex_array::arrays::{ListArray, VarBinArray};
/// use vortex_array::compute::list_contains;
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_dtype::DType;
/// let elements = VarBinArray::from_vec(
///         vec!["a", "a", "b", "a", "c"], DType::Utf8(false.into())).into_array();
/// let offsets = buffer![0u32, 1, 3, 5].into_array();
/// let list_array = ListArray::try_new(elements, offsets, Validity::NonNullable).unwrap();
///
/// let matches = list_contains(&list_array, "b".into()).unwrap();
/// let to_vec: Vec<bool> = matches.to_boolean_buffer().iter().collect();
/// assert_eq!(to_vec, vec![false, true, false]);
/// ```
pub fn list_contains(array: &dyn Array, value: Scalar) -> VortexResult<Mask> {
    // Ensure that the array must be of List type.
    let Some(list_array) = array.as_any().downcast_ref::<ListArray>() else {
        vortex_bail!("array must be of List type")
    };

    // Push the comparison down into the matching elements array.
    let elems = list_array.elements();
    let rhs = ConstantArray::new(value, elems.len());
    let matching_elements = compare(elems, &rhs, Operator::Eq)?;
    let ends = list_array.offsets().to_primitive()?;
    let matches = matching_elements.to_bool()?;

    match_each_integer_ptype!(ends.ptype(), |$T| {
        Ok(reduce_with_ends(ends.as_slice::<$T>(), &matches.boolean_buffer()))
    })
}

// Reduce each boolean values into a Mask that indicates which elements in the
// ListArray contain the matching value.
fn reduce_with_ends<T: NativePType + AsPrimitive<usize>>(
    ends: &[T],
    matches: &BooleanBuffer,
) -> Mask {
    if ends.len() == 1 {
        // The array is empty. Ignore the request.
        return Mask::new_false(0);
    }

    let match_count = matches.count_set_bits();

    // Fast paths: all match or none match.
    if match_count == 0 {
        return Mask::new_false(matches.len());
    }
    if match_count == matches.len() {
        return Mask::new_true(matches.len());
    }

    let mask: BooleanBuffer = ends
        .windows(2)
        .map(|window| {
            let len = window[1].as_() - window[0].as_();
            let segment = matches.slice(window[0].as_(), len);
            segment.count_set_bits() > 0
        })
        .collect();

    Mask::from_buffer(mask)
}

/// Returns a new array of `u64` representing the length of each list element.
///
/// ## Example
///
/// ```rust
/// use vortex_array::arrays::{ListArray, VarBinArray};
/// use vortex_array::{Array, IntoArray};
/// use vortex_array::compute::{list_elem_len, scalar_at};
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_dtype::DType;
///
/// let elements = VarBinArray::from_vec(
///         vec!["a", "a", "b", "a", "c"], DType::Utf8(false.into())).into_array();
/// let offsets = buffer![0u32, 1, 3, 5].into_array();
/// let list_array = ListArray::try_new(elements, offsets, Validity::NonNullable).unwrap();
///
/// let lens = list_elem_len(&list_array).unwrap();
/// assert_eq!(scalar_at(&lens, 0).unwrap(), 1u32.into());
/// assert_eq!(scalar_at(&lens, 1).unwrap(), 2u32.into());
/// assert_eq!(scalar_at(&lens, 2).unwrap(), 2u32.into());
/// ```
pub fn list_elem_len(array: &dyn Array) -> VortexResult<ArrayRef> {
    let Some(list_array) = array.as_any().downcast_ref::<ListArray>() else {
        vortex_bail!("array must be of List type")
    };

    let offsets = list_array.offsets().to_primitive()?;
    let lens_array = match_each_integer_ptype!(offsets.ptype(), |$T| {
        element_lens(offsets.as_slice::<$T>()).into_array()
    });

    Ok(lens_array)
}

fn element_lens<T: NativePType>(values: &[T]) -> Buffer<T> {
    values
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect()
}
