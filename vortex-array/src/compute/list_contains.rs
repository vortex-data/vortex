// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! List-related compute operations.

use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::bit_iterator::BitIndexIterator;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::{ListScalar, Scalar};

use crate::arrays::{BoolArray, ConstantArray, ListArray};
use crate::compute::{
    BinaryArgs, ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Operator, Output, compare,
    fill_null, or,
};
use crate::validity::Validity;
use crate::vtable::{VTable, ValidityHelper};
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

static LIST_CONTAINS_FN: LazyLock<ComputeFn> = LazyLock::new(|| {
    let compute = ComputeFn::new("list_contains".into(), ArcRef::new_ref(&ListContains));
    for kernel in inventory::iter::<ListContainsKernelRef> {
        compute.register_kernel(kernel.0.clone());
    }
    compute
});

/// Compute a `Bool`-typed array the same length as `array` where elements is `true` if the list
/// item contains the `value`, `false` otherwise.
///
/// ## Null scalar handling
///
/// If the `value` or `array` is `null` at any index the result at that index is `null`.
///
/// ## Format semantics
/// ```txt
/// list_contains(list, elem)
///   ==> (!is_null(list) or NULL) and (!is_null(elem) or NULL) and any({elem = elem_i | elem_i in list}),
/// ```
///
/// ## Example
///
/// ```rust
/// use vortex_array::{Array, IntoArray, ToCanonical};
/// use vortex_array::arrays::{ConstantArray, ListArray, VarBinArray};
/// use vortex_array::compute::list_contains;
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_dtype::DType;
/// use vortex_scalar::Scalar;
/// let elements = VarBinArray::from_vec(
///         vec!["a", "a", "b", "a", "c"], DType::Utf8(false.into())).into_array();
/// let offsets = buffer![0u32, 1, 3, 5].into_array();
/// let list_array = ListArray::try_new(elements, offsets, Validity::NonNullable).unwrap();
///
/// let matches = list_contains(list_array.as_ref(), ConstantArray::new(Scalar::from("b"), list_array.len()).as_ref()).unwrap();
/// let to_vec: Vec<bool> = matches.to_bool().boolean_buffer().iter().collect();
/// assert_eq!(to_vec, vec![false, true, false]);
/// ```
pub fn list_contains(array: &dyn Array, value: &dyn Array) -> VortexResult<ArrayRef> {
    LIST_CONTAINS_FN
        .invoke(&InvocationArgs {
            inputs: &[array.into(), value.into()],
            options: &(),
        })?
        .unwrap_array()
}

pub struct ListContains;

impl ComputeFnVTable for ListContains {
    fn invoke(
        &self,
        args: &InvocationArgs,
        kernels: &[ArcRef<dyn Kernel>],
    ) -> VortexResult<Output> {
        let BinaryArgs {
            lhs: array,
            rhs: value,
            ..
        } = BinaryArgs::<()>::try_from(args)?;

        let DType::List(elem_dtype, _) = array.dtype() else {
            vortex_bail!("Array must be of List type");
        };
        if !elem_dtype.as_ref().eq_ignore_nullability(value.dtype()) {
            vortex_bail!(
                "Element type {} of ListArray does not match search value {}",
                elem_dtype,
                value.dtype(),
            );
        };

        if value.all_invalid() || array.all_invalid() {
            return Ok(Output::Array(
                ConstantArray::new(
                    Scalar::null(DType::Bool(Nullability::Nullable)),
                    array.len(),
                )
                .to_array(),
            ));
        }

        for kernel in kernels {
            if let Some(output) = kernel.invoke(args)? {
                return Ok(output);
            }
        }
        if let Some(output) = array.invoke(&LIST_CONTAINS_FN, args)? {
            return Ok(output);
        }

        let nullability = array.dtype().nullability() | value.dtype().nullability();

        let result = if let Some(value_scalar) = value.as_constant() {
            list_contains_scalar(array, &value_scalar, nullability)
        } else if let Some(list_scalar) = array.as_constant() {
            constant_list_scalar_contains(&list_scalar.as_list(), value, nullability)
        } else {
            todo!("unsupported list contains with list and element as arrays")
        };

        result.map(Output::Array)
    }

    fn return_dtype(&self, args: &InvocationArgs) -> VortexResult<DType> {
        let input = BinaryArgs::<()>::try_from(args)?;
        Ok(DType::Bool(
            input.lhs.dtype().nullability() | input.rhs.dtype().nullability(),
        ))
    }

    fn return_len(&self, args: &InvocationArgs) -> VortexResult<usize> {
        Ok(BinaryArgs::<()>::try_from(args)?.lhs.len())
    }

    fn is_elementwise(&self) -> bool {
        true
    }
}

pub trait ListContainsKernel: VTable {
    fn list_contains(
        &self,
        list: &dyn Array,
        element: &Self::Array,
    ) -> VortexResult<Option<ArrayRef>>;
}

pub struct ListContainsKernelRef(ArcRef<dyn Kernel>);
inventory::collect!(ListContainsKernelRef);

#[derive(Debug)]
pub struct ListContainsKernelAdapter<V: VTable>(pub V);

impl<V: VTable + ListContainsKernel> ListContainsKernelAdapter<V> {
    pub const fn lift(&'static self) -> ListContainsKernelRef {
        ListContainsKernelRef(ArcRef::new_ref(self))
    }
}

impl<V: VTable + ListContainsKernel> Kernel for ListContainsKernelAdapter<V> {
    fn invoke(&self, args: &InvocationArgs) -> VortexResult<Option<Output>> {
        let BinaryArgs {
            lhs: array,
            rhs: value,
            ..
        } = BinaryArgs::<()>::try_from(args)?;
        let Some(value) = value.as_opt::<V>() else {
            return Ok(None);
        };
        self.0
            .list_contains(array, value)
            .map(|c| c.map(Output::Array))
    }
}

// Then there is a constant list scalar (haystack) being compared to an array of needles.
fn constant_list_scalar_contains(
    list_scalar: &ListScalar,
    values: &dyn Array,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let elements = list_scalar.elements().vortex_expect("non null");

    let len = values.len();
    let mut result: Option<ArrayRef> = None;
    let false_scalar = Scalar::bool(false, nullability);
    for element in elements {
        let res = fill_null(
            &compare(
                ConstantArray::new(element, len).as_ref(),
                values,
                Operator::Eq,
            )?,
            &false_scalar,
        )?;
        if let Some(acc) = result {
            result = Some(or(&acc, &res)?)
        } else {
            result = Some(res);
        }
    }
    Ok(result.unwrap_or_else(|| ConstantArray::new(false_scalar, len).to_array()))
}

fn list_contains_scalar(
    array: &dyn Array,
    value: &Scalar,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    // If the list array is constant, we perform a single comparison.
    if array.len() > 1 && array.is_constant() {
        let contains = list_contains_scalar(&array.slice(0..1), value, nullability)?;
        return Ok(ConstantArray::new(contains.scalar_at(0), array.len()).into_array());
    }

    // Canonicalize to a list array.
    // NOTE(ngates): we may wish to add elements and offsets accessors to the ListArrayTrait.
    let list_array = array.to_list();

    let elems = list_array.elements();
    if elems.is_empty() {
        // Must return false when a list is empty (but valid), or null when the list itself is null.
        return list_false_or_null(&list_array, nullability);
    }

    let rhs = ConstantArray::new(value.clone(), elems.len());
    let matching_elements = compare(elems, rhs.as_ref(), Operator::Eq)?;
    let matches = matching_elements.to_bool();

    // Fast path: no elements match.
    if let Some(pred) = matches.as_constant() {
        return match pred.as_bool().value() {
            // All comparisons are invalid (result in `null`), and search is not null because
            // we already checked for null above.
            None => {
                assert!(
                    !rhs.scalar().is_null(),
                    "Search value must not be null here"
                );
                // False, unless the list itself is null in which case we return null.
                list_false_or_null(&list_array, nullability)
            }
            // No elements match, and all comparisons are valid (result in `false`).
            Some(false) => {
                // False, but match the nullability to the input list array.
                Ok(
                    ConstantArray::new(Scalar::bool(false, nullability), list_array.len())
                        .into_array(),
                )
            }
            // All elements match, and all comparisons are valid (result in `true`).
            Some(true) => {
                // True, unless the list itself is empty or NULL.
                list_is_not_empty(&list_array, nullability)
            }
        };
    }

    let ends = list_array.offsets().to_primitive();
    match_each_integer_ptype!(ends.ptype(), |T| {
        Ok(reduce_with_ends(
            ends.as_slice::<T>(),
            matches.boolean_buffer(),
            list_array.validity().clone().union_nullability(nullability),
        ))
    })
}

/// Returns a `Bool` array with `false` for lists that are valid,
/// or `NULL` if the list itself is null.
fn list_false_or_null(list_array: &ListArray, nullability: Nullability) -> VortexResult<ArrayRef> {
    match list_array.validity() {
        Validity::NonNullable => {
            // All false.
            Ok(ConstantArray::new(Scalar::bool(false, nullability), list_array.len()).into_array())
        }
        Validity::AllValid => {
            // All false, but nullable.
            Ok(
                ConstantArray::new(Scalar::bool(false, Nullability::Nullable), list_array.len())
                    .into_array(),
            )
        }
        Validity::AllInvalid => {
            // All nulls, must be nullable result.
            Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                list_array.len(),
            )
            .into_array())
        }
        Validity::Array(validity_array) => {
            // Create a new bool array with false, and the provided nulls
            let buffer = BooleanBuffer::new_unset(list_array.len());
            Ok(
                BoolArray::from_bool_buffer(buffer, Validity::Array(validity_array.clone()))
                    .into_array(),
            )
        }
    }
}

/// Returns a `Bool` array with `true` for lists which are NOT empty, or `false` if they are empty,
/// or `NULL` if the list itself is null.
fn list_is_not_empty(list_array: &ListArray, nullability: Nullability) -> VortexResult<ArrayRef> {
    // Short-circuit for all invalid.
    if matches!(list_array.validity(), Validity::AllInvalid) {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            list_array.len(),
        )
        .into_array());
    }

    let offsets = list_array.offsets().to_primitive();
    let buffer = match_each_integer_ptype!(offsets.ptype(), |T| {
        element_is_not_empty(offsets.as_slice::<T>())
    });

    // Copy over the validity mask from the input.
    Ok(BoolArray::from_bool_buffer(
        buffer,
        list_array.validity().clone().union_nullability(nullability),
    )
    .into_array())
}

/// Reduces each boolean values into a Mask that indicates which elements in the
/// ListArray contain the matching value.
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

    BoolArray::from_bool_buffer(mask, validity).into_array()
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
/// let lens = list_elem_len(list_array.as_ref()).unwrap();
/// assert_eq!(lens.scalar_at(0), 1u32.into());
/// assert_eq!(lens.scalar_at(1), 2u32.into());
/// assert_eq!(lens.scalar_at(2), 2u32.into());
/// ```
pub fn list_elem_len(array: &dyn Array) -> VortexResult<ArrayRef> {
    if !matches!(array.dtype(), DType::List(..)) {
        vortex_bail!("Array must be of list type");
    }

    // Short-circuit for constant list arrays.
    if array.is_constant() && array.len() > 1 {
        let elem_lens = list_elem_len(&array.slice(0..1))?;
        return Ok(ConstantArray::new(elem_lens.scalar_at(0), array.len()).into_array());
    }

    let list_array = array.to_list();
    let offsets = list_array.offsets().to_primitive();
    let lens_array = match_each_integer_ptype!(offsets.ptype(), |T| {
        element_lens(offsets.as_slice::<T>()).into_array()
    });

    Ok(lens_array)
}

fn element_lens<T: NativePType>(values: &[T]) -> Buffer<T> {
    values
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect()
}

fn element_is_not_empty<T: NativePType>(values: &[T]) -> BooleanBuffer {
    BooleanBuffer::from_iter(values.windows(2).map(|window| window[1] != window[0]))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::arrays::{
        BoolArray, ConstantArray, ConstantVTable, ListArray, PrimitiveArray, VarBinArray,
    };
    use crate::canonical::ToCanonical;
    use crate::compute::list_contains;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{Array, ArrayRef, IntoArray};

    fn nonnull_strings(values: Vec<Vec<&str>>) -> ArrayRef {
        ListArray::from_iter_slow::<u64, _>(values, Arc::new(DType::Utf8(Nullability::NonNullable)))
            .unwrap()
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

    fn bool_array(values: Vec<bool>, validity: Validity) -> BoolArray {
        BoolArray::from_bool_buffer(values.into_iter().collect(), validity)
    }

    #[rstest]
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a", "b"]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::NonNullable)
    )]
    // Cast 2: valid scalar search over nullable list, with all nulls matched
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("a"), None, Some("b")]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::AllValid)
    )]
    // Cast 3: valid scalar search over nullable list, with some nulls not matched (return no nulls)
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("b"), None, None]]),
        Some("a"),
        bool_array(vec![false, true, false], Validity::AllValid)
    )]
    // Case 4: list(utf8) with all elements matching, but some empty lists
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a"]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::NonNullable)
    )]
    // Case 5: list(utf8) all lists empty.
    #[case(
        nonnull_strings(vec![vec![], vec![], vec![]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::NonNullable)
    )]
    // Case 6: list(utf8) no elements matching.
    #[case(
        nonnull_strings(vec![vec!["b"], vec![], vec!["b"]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::NonNullable)
    )]
    // Case 7: list(utf8?) with empty + NULL elements and NULL search
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        None,
        bool_array(vec![false, true, true], Validity::AllInvalid)
    )]
    // Case 8: list(utf8?) with empty + NULL elements and search scalar
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::AllValid)
    )]
    fn test_contains_nullable(
        #[case] list_array: ArrayRef,
        #[case] value: Option<&str>,
        #[case] expected: BoolArray,
    ) {
        let element_nullability = list_array
            .dtype()
            .as_list_element_opt()
            .unwrap()
            .nullability();
        let scalar = match value {
            None => Scalar::null(DType::Utf8(Nullability::Nullable)),
            Some(v) => Scalar::utf8(v, element_nullability),
        };
        let elem = ConstantArray::new(scalar, list_array.len());
        let result = list_contains(&list_array, elem.as_ref()).expect("list_contains failed");
        let bool_result = result.to_bool();
        assert_eq!(
            bool_result.opt_bool_vec().unwrap(),
            expected.opt_bool_vec().unwrap()
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

        let contains = list_contains(
            &list_array,
            ConstantArray::new(Scalar::from(2i32), list_array.len()).as_ref(),
        )
        .unwrap();
        assert!(contains.is::<ConstantVTable>(), "Expected constant result");
        assert_eq!(
            contains.to_bool().boolean_buffer().iter().collect_vec(),
            vec![true, true],
        );
    }

    #[test]
    fn test_all_nulls() {
        let list_array = ConstantArray::new(
            Scalar::null(DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Nullability::Nullable,
            )),
            5,
        )
        .into_array();

        let contains = list_contains(
            &list_array,
            ConstantArray::new(Scalar::from(2i32), list_array.len()).as_ref(),
        )
        .unwrap();
        assert!(contains.is::<ConstantVTable>(), "Expected constant result");

        assert_eq!(contains.len(), 5);
        assert_eq!(contains.to_bool().validity(), &Validity::AllInvalid);
    }

    #[test]
    fn test_list_array_element() {
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1.into(), 3.into(), 6.into()],
            Nullability::NonNullable,
        );

        let contains = list_contains(
            ConstantArray::new(list_scalar, 7).as_ref(),
            (0..7).collect::<PrimitiveArray>().as_ref(),
        )
        .unwrap();

        assert_eq!(contains.len(), 7);
        assert_eq!(
            contains.to_bool().opt_bool_vec().unwrap(),
            vec![
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(true)
            ]
        );
    }
}
