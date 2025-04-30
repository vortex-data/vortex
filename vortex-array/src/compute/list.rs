//! List-related compute operations.

use arrow_buffer::BooleanBuffer;
use arrow_buffer::bit_iterator::BitIndexIterator;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ListArray};
use crate::compute::{Operator, compare};
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ArrayStatistics, IntoArray, ToCanonical};

/// Compute a `Bool`-typed array the same length as `array` where elements are `true` if the list
/// item contains the `value`, or `false` otherwise.
///
/// ## Null handling
///
/// This function has the same NULL semantics as [`compare`], i.e. nulls will occupy a position
/// where there is
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
pub fn list_contains(array: &dyn Array, value: Scalar) -> VortexResult<ArrayRef> {
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

    // Fast path: all elements match or none match.
    if let Some(pred) = matches.as_constant() {
        return match pred.as_bool().value() {
            // TODO(aduffy): how do we handle null?
            None | Some(false) => Ok(ConstantArray::new(false.into(), matches.len()).into_array()),
            Some(true) => Ok(ConstantArray::new(true.into(), matches.len()).into_array()),
        };
    }

    match_each_integer_ptype!(ends.ptype(), |$T| {
        Ok(reduce_with_ends(ends.as_slice::<$T>(), &matches.boolean_buffer()))
    })
}

// Reduce each boolean values into a Mask that indicates which elements in the
// ListArray contain the matching value.
fn reduce_with_ends<T: NativePType + AsPrimitive<usize>>(
    ends: &[T],
    matches: &BooleanBuffer,
) -> ArrayRef {
    let mask: BooleanBuffer = ends
        .windows(2)
        .map(|window| {
            let len = window[1].as_() - window[0].as_();
            let mut set_bits = BitIndexIterator::new(matches.values(), window[0].as_(), len);
            set_bits.next().is_some()
        })
        .collect();

    Mask::from_buffer(mask).into_array()
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
