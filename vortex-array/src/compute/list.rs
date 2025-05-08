//! List-related compute operations.

use arrow_buffer::BooleanBuffer;
use arrow_buffer::bit_iterator::BitIndexIterator;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, ConstantArray, ListArray};
use crate::compute::{Operator, compare, invert};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef, ArrayStatistics, IntoArray, ToCanonical};

/// Compute a `Bool`-typed array the same length as `array` where elements are `true` if the list
/// item contains the `value`, or `false` otherwise.
///
/// If the ListArray is nullable, then the result will contain nulls matching the null mask
/// of the original array.
///
/// ## Null scalar handling
///
/// When the search scalar is `NULL`, then the resulting array will be a `BoolArray` containing
/// `true` if the list contains any nulls, and `false` if the list does not contain any nulls,
/// or `NULL` for null lists.
///
/// ## Example
///
/// ```rust
/// use vortex_array::{Array, IntoArray, ToCanonical};
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
/// let to_vec: Vec<bool> = matches.to_bool().unwrap().boolean_buffer().iter().collect();
/// assert_eq!(to_vec, vec![false, true, false]);
/// ```
pub fn list_contains(array: &dyn Array, value: Scalar) -> VortexResult<ArrayRef> {
    let DType::List(elem_dtype, _nullability) = array.dtype() else {
        vortex_bail!("Array must be of List type");
    };
    if &**elem_dtype != value.dtype() {
        vortex_bail!("Element type of ListArray does not match search value");
    }

    // If the list array is constant, we perform a single comparison.
    if array.is_constant() && array.len() > 1 {
        let contains = list_contains(&array.slice(0, 1)?, value)?;
        return Ok(ConstantArray::new(contains.scalar_at(0)?, array.len()).into_array());
    }

    // Canonicalize to a list array.
    // NOTE(ngates): we may wish to add elements and offsets accessors to the ListArrayTrait.
    let list_array = array.to_list()?;

    if value.is_null() {
        return list_contains_null(&list_array);
    }

    let elems = list_array.elements();
    let ends = list_array.offsets().to_primitive()?;

    let rhs = ConstantArray::new(value, elems.len());
    let matching_elements = compare(elems, &rhs, Operator::Eq)?;
    let matches = matching_elements.to_bool()?;

    // Fast path: no elements match.
    if let Some(pred) = matches.as_constant() {
        if matches!(pred.as_bool().value(), None | Some(false)) {
            // TODO(aduffy): how do we handle null?
            return Ok(ConstantArray::new::<bool>(false, list_array.len()).into_array());
        }
    }

    match_each_integer_ptype!(ends.ptype(), |$T| {
        Ok(reduce_with_ends(ends.as_slice::<$T>(), &matches.boolean_buffer(), list_array.validity().clone()))
    })
}

/// Returns a `Bool` array with `true` for lists which contains NULL and `false` if not, or
/// NULL if the list itself is null.
fn list_contains_null(list_array: &ListArray) -> VortexResult<ArrayRef> {
    let elems = list_array.elements();

    // Check element validity. We need to intersect
    match elems.validity_mask()? {
        // No NULL elements
        Mask::AllTrue(_) => match list_array.validity() {
            Validity::NonNullable => {
                Ok(ConstantArray::new::<bool>(false, list_array.len()).into_array())
            }
            Validity::AllValid => Ok(ConstantArray::new(
                Scalar::bool(true, Nullability::Nullable),
                list_array.len(),
            )
            .into_array()),
            Validity::AllInvalid => Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                list_array.len(),
            )
            .into_array()),
            Validity::Array(list_mask) => {
                // Create a new bool array with false, and the provided nulls
                let buffer = BooleanBuffer::new_unset(list_array.len());
                Ok(BoolArray::new(buffer, Validity::Array(list_mask.clone())).into_array())
            }
        },
        // All null elements
        Mask::AllFalse(_) => Ok(ConstantArray::new::<bool>(true, list_array.len()).into_array()),
        Mask::Values(mask) => {
            let nulls = invert(&mask.into_array())?.to_bool()?;
            let ends = list_array.offsets().to_primitive()?;
            match_each_integer_ptype!(ends.ptype(), |$T| {
                Ok(reduce_with_ends(
                    list_array.offsets().to_primitive()?.as_slice::<$T>(),
                    &nulls.boolean_buffer(),
                    list_array.validity().clone(),
                ))
            })
        }
    }
}

// Reduce each boolean values into a Mask that indicates which elements in the
// ListArray contain the matching value.
fn reduce_with_ends<T: NativePType + AsPrimitive<usize>>(
    ends: &[T],
    matches: &BooleanBuffer,
    validity: Validity,
) -> ArrayRef {
    let mask: BooleanBuffer = ends
        .windows(2)
        .map(|window| {
            let len = window[1].as_() - window[0].as_();
            let mut set_bits = BitIndexIterator::new(matches.values(), window[0].as_(), len);
            set_bits.next().is_some()
        })
        .collect();

    BoolArray::new(mask, validity).into_array()
}

/// Returns a new array of `u64` representing the length of each list element.
///
/// ## Example
///
/// ```rust
/// use vortex_array::arrays::{ListArray, VarBinArray};
/// use vortex_array::{Array, IntoArray};
/// use vortex_array::compute::{list_elem_len};
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
/// assert_eq!(lens.scalar_at(0).unwrap(), 1u32.into());
/// assert_eq!(lens.scalar_at(1).unwrap(), 2u32.into());
/// assert_eq!(lens.scalar_at(2).unwrap(), 2u32.into());
/// ```
pub fn list_elem_len(array: &dyn Array) -> VortexResult<ArrayRef> {
    if !matches!(array.dtype(), DType::List(..)) {
        vortex_bail!("Array must be of list type");
    }

    // Short-circuit for constant list arrays.
    if array.is_constant() && array.len() > 1 {
        let elem_lens = list_elem_len(&array.slice(0, 1)?)?;
        return Ok(ConstantArray::new(elem_lens.scalar_at(0)?, array.len()).into_array());
    }

    let list_array = array.to_list()?;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::IntoArray;
    use crate::arrays::{BoolArray, ConstantArray, ListArray, VarBinArray};
    use crate::canonical::ToCanonical;
    use crate::compute::list_contains;
    use crate::validity::Validity;
    use crate::{Array, ArrayExt, ArrayRef};

    fn nonnull_strings(values: Vec<Vec<&str>>) -> ArrayRef {
        ListArray::from_iter_slow::<u64, _>(values, Arc::new(DType::Utf8(Nullability::NonNullable)))
            .unwrap()
            .into_array()
    }

    fn null_strings(values: Vec<Vec<Option<&str>>>) -> ArrayRef {
        let elements = values.iter().flatten().cloned().collect_vec();
        let mut offsets = values
            .iter()
            .scan(0u64, |st, v| {
                *st += v.len() as u64;
                Some(*st)
            })
            .collect_vec();
        offsets.insert(0, 0u64);
        let offsets = Buffer::from_iter(offsets).into_array();

        let elements =
            VarBinArray::from_iter(elements, DType::Utf8(Nullability::Nullable)).into_array();

        ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .into_array()
    }

    fn bool_array(values: Vec<bool>, validity: Option<Vec<bool>>) -> BoolArray {
        let validity = match validity {
            None => Validity::NonNullable,
            Some(v) => Validity::from_iter(v),
        };

        BoolArray::new(values.into_iter().collect(), validity)
    }

    #[rstest]
    // Case 1: list(utf8)
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a", "b"]]),
        Some("a"),
        bool_array(vec![false, true, true], None)
    )]
    // Case 2: list(utf8?) with NULL search scalar
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("a"), None, Some("b")]]),
        None,
        bool_array(vec![false, true, true], None)
    )]
    // Case 3: list(utf8) with all elements matching, but some empty lists
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a"]]),
        Some("a"),
        bool_array(vec![false, true, true], None)
    )]
    // Case 4: list(utf8) all lists empty.
    #[case(
        nonnull_strings(vec![vec![], vec![], vec![]]),
        Some("a"),
        bool_array(vec![false, false, false], None)
    )]
    // Case 5: list(utf8) no elements matching.
    #[case(
        nonnull_strings(vec![vec!["b"], vec![], vec!["b"]]),
        Some("a"),
        bool_array(vec![false, false, false], None)
    )]
    fn test_contains_nullable(
        #[case] list_array: ArrayRef,
        #[case] value: Option<&str>,
        #[case] expected: BoolArray,
    ) {
        let element_nullability = list_array.dtype().as_list_element().unwrap().nullability();
        let scalar = match value {
            None => Scalar::null(DType::Utf8(Nullability::Nullable)),
            Some(v) => Scalar::utf8(v, element_nullability),
        };
        let result = list_contains(&list_array, scalar).expect("list_contains failed");
        let bool_result = result.to_bool().expect("to_bool failed");
        assert_eq!(
            bool_result.boolean_buffer().iter().collect_vec(),
            expected.boolean_buffer().iter().collect_vec()
        );
        assert_eq!(bool_result.validity(), expected.validity());
    }

    #[test]
    fn test_constant_list() {
        let list_array = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                vec![1i32.into(), 2i32.into(), 3i32.into()],
                Nullability::NonNullable,
            ),
            2,
        )
        .into_array();

        let contains = list_contains(&list_array, 2i32.into()).unwrap();
        assert!(contains.is::<ConstantArray>(), "Expected constant result");
        assert_eq!(
            contains
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect_vec(),
            vec![true, true],
        );
    }
}
