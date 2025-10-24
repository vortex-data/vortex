// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! List-related compute operations.

// TODO(connor)[ListView]: Should this compute function be moved up the `arrays/listview`?
// TODO(connor)[ListView]: Clean up this file.

use std::sync::LazyLock;

use arcref::ArcRef;
use arrow_buffer::bit_iterator::BitIndexIterator;
use num_traits::Zero;
use vortex_buffer::BitBuffer;
use vortex_dtype::{DType, IntegerPType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::{ListScalar, Scalar};

use crate::arrays::{BoolArray, ConstantArray, ListViewArray, PrimitiveArray};
use crate::compute::{
    self, BinaryArgs, ComputeFn, ComputeFnVTable, InvocationArgs, Kernel, Operator, Output,
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

pub(crate) fn warm_up_vtable() -> usize {
    LIST_CONTAINS_FN.kernels().len()
}

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
/// # use vortex_array::{Array, IntoArray, ToCanonical};
/// # use vortex_array::arrays::{ConstantArray, ListViewArray, VarBinArray};
/// # use vortex_array::compute;
/// # use vortex_array::validity::Validity;
/// # use vortex_buffer::{buffer, bitbuffer};
/// # use vortex_dtype::DType;
/// # use vortex_scalar::Scalar;
/// #
/// let elements = VarBinArray::from_vec(
///         vec!["a", "a", "b", "a", "c"], DType::Utf8(false.into())).into_array();
/// let offsets = buffer![0u32, 1, 3].into_array();
/// let sizes = buffer![1u32, 2, 2].into_array();
/// let list_array =
///     ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable).unwrap();
///
/// let matches = compute::list_contains(
///     list_array.as_ref(),
///     ConstantArray::new(Scalar::from("b"),
///     list_array.len()).as_ref()
/// ).unwrap();
///
/// assert_eq!(matches.to_bool().bit_buffer(), &bitbuffer![false, true, false]);
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
                "Element type {} of `ListViewArray` does not match search value {}",
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
        let res = compute::fill_null(
            &compute::compare(
                ConstantArray::new(element, len).as_ref(),
                values,
                Operator::Eq,
            )?,
            &false_scalar,
        )?;
        if let Some(acc) = result {
            result = Some(compute::or(&acc, &res)?)
        } else {
            result = Some(res);
        }
    }
    Ok(result.unwrap_or_else(|| ConstantArray::new(false_scalar, len).to_array()))
}

/// Returns a [`BoolArray`] where each bit represents if a list contains the scalar.
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

    let list_array = array.to_listview();

    let elems = list_array.elements();
    if elems.is_empty() {
        // Must return false when a list is empty (but valid), or null when the list itself is null.
        return list_false_or_null(&list_array, nullability);
    }

    let rhs = ConstantArray::new(value.clone(), elems.len());
    let matching_elements = compute::compare(elems, rhs.as_ref(), Operator::Eq)?;
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

    // Get the offsets and sizes as primitive arrays.
    let offsets = list_array.offsets().to_primitive();
    let sizes = list_array.sizes().to_primitive();

    // Process based on the offset and size types.
    let list_matches = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            process_matches::<O, S>(matches, list_array.len(), offsets, sizes)
        })
    });

    Ok(BoolArray::from_bit_buffer(
        list_matches,
        list_array.validity().clone().union_nullability(nullability),
    )
    .into_array())
}

/// Returns a [`BooleanBuffer`] where each bit represents if a list contains the scalar, derived
/// from a [`BoolArray`] of matches on the child elements array.
fn process_matches<O, S>(
    matches: BoolArray,
    list_array_len: usize,
    offsets: PrimitiveArray,
    sizes: PrimitiveArray,
) -> BitBuffer
where
    O: IntegerPType,
    S: IntegerPType,
{
    let offsets_slice = offsets.as_slice::<O>();
    let sizes_slice = sizes.as_slice::<S>();

    (0..list_array_len)
        .map(|i| {
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits =
                BitIndexIterator::new(matches.bit_buffer().inner().as_ref(), offset, size);
            set_bits.next().is_some()
        })
        .collect::<BitBuffer>()
}

/// Returns a `Bool` array with `false` for lists that are valid,
/// or `NULL` if the list itself is null.
fn list_false_or_null(
    list_array: &ListViewArray,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
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
            let buffer = BitBuffer::new_unset(list_array.len());
            Ok(
                BoolArray::from_bit_buffer(buffer, Validity::Array(validity_array.clone()))
                    .into_array(),
            )
        }
    }
}

/// Returns a `Bool` array with `true` for lists which are NOT empty, or `false` if they are empty,
/// or `NULL` if the list itself is null.
fn list_is_not_empty(
    list_array: &ListViewArray,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    // Short-circuit for all invalid.
    if matches!(list_array.validity(), Validity::AllInvalid) {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            list_array.len(),
        )
        .into_array());
    }

    let sizes = list_array.sizes().to_primitive();
    let buffer = match_each_integer_ptype!(sizes.ptype(), |S| {
        BitBuffer::from_iter(sizes.as_slice::<S>().iter().map(|&size| size != S::zero()))
    });

    // Copy over the validity mask from the input.
    Ok(BoolArray::from_bit_buffer(
        buffer,
        list_array.validity().clone().union_nullability(nullability),
    )
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_buffer::{Buffer, bitbuffer};
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::arrays::{
        BoolArray, ConstantArray, ConstantVTable, ListArray, ListVTable, ListViewArray,
        PrimitiveArray, VarBinArray, list_view_from_list,
    };
    use crate::canonical::ToCanonical;
    use crate::compute::list_contains;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;
    use crate::{Array, ArrayRef, IntoArray};

    fn nonnull_strings(values: Vec<Vec<&str>>) -> ArrayRef {
        list_view_from_list(
            ListArray::from_iter_slow::<u64, _>(
                values,
                Arc::new(DType::Utf8(Nullability::NonNullable)),
            )
            .unwrap()
            .as_::<ListVTable>()
            .clone(),
        )
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

        list_view_from_list(ListArray::try_new(elements, offsets, Validity::NonNullable).unwrap())
            .into_array()
    }

    fn bool_array(values: Vec<bool>, validity: Validity) -> BoolArray {
        BoolArray::from_bit_buffer(values.into_iter().collect(), validity)
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
        assert_eq!(bool_result.opt_bool_vec(), expected.opt_bool_vec());
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
        assert_eq!(contains.to_bool().bit_buffer(), &bitbuffer![true, true],);
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
            contains.to_bool().opt_bool_vec(),
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

    #[test]
    fn test_list_contains_empty_listview() {
        // Create a completely empty ListView with no elements
        let empty_elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
        let offsets = Buffer::from_iter([0u32, 0, 0, 0]).into_array();
        let sizes = Buffer::from_iter([0u32, 0, 0, 0]).into_array();

        let list_array = ListViewArray::try_new(
            empty_elements.into_array(),
            offsets,
            sizes,
            Validity::NonNullable,
        )
        .unwrap();

        // Test with a non-null search value
        let search = ConstantArray::new(Scalar::from(42i32), list_array.len());
        let result = list_contains(list_array.as_ref(), search.as_ref()).unwrap();

        // All lists are empty, so all should return false
        assert_eq!(result.len(), 4);
        assert_eq!(
            result.to_bool().bool_vec(),
            vec![false, false, false, false]
        );
    }

    #[test]
    fn test_list_contains_all_null_elements() {
        // Create lists containing only null elements
        let elements = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None, None]);
        let offsets = Buffer::from_iter([0u32, 2, 4]).into_array();
        let sizes = Buffer::from_iter([2u32, 2, 1]).into_array();

        let list_array =
            ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable)
                .unwrap();

        // Test searching for a null value
        let null_search = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            list_array.len(),
        );
        let result = list_contains(list_array.as_ref(), null_search.as_ref()).unwrap();

        // Searching for null in lists with null elements should return null
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_bool().validity(), &Validity::AllInvalid);

        // Test searching for a non-null value
        let non_null_search = ConstantArray::new(Scalar::from(42i32), list_array.len());
        let result2 = list_contains(list_array.as_ref(), non_null_search.as_ref()).unwrap();

        // All comparisons result in null, but search is not null, so should return false
        assert_eq!(result2.len(), 3);
        assert_eq!(result2.to_bool().bool_vec(), vec![false, false, false]);
    }

    #[test]
    fn test_list_contains_large_offsets() {
        // Test with large offset values that are still valid
        // ListView allows non-contiguous views into the elements array
        let elements = Buffer::from_iter([1i32, 2, 3, 4, 5]).into_array();

        // Create lists with various offsets, testing the flexibility of ListView
        // List 0: element at offset 0 (value 1)
        // List 1: elements at offset 1-2 (values 2, 3)
        // List 2: element at offset 4 (value 5)
        // List 3: empty list
        let offsets = Buffer::from_iter([0u32, 1, 4, 0]).into_array();
        let sizes = Buffer::from_iter([1u32, 2, 1, 0]).into_array();

        let list_array =
            ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable)
                .unwrap();

        // Test searching for value 2, which appears only in list 1
        let search = ConstantArray::new(Scalar::from(2i32), list_array.len());
        let result = list_contains(list_array.as_ref(), search.as_ref()).unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(
            result.to_bool().bool_vec(),
            vec![false, true, false, false] // Value 2 is only in list 1
        );

        // Test searching for value 5, which appears only in list 2
        let search5 = ConstantArray::new(Scalar::from(5i32), list_array.len());
        let result5 = list_contains(list_array.as_ref(), search5.as_ref()).unwrap();

        assert_eq!(
            result5.to_bool().bool_vec(),
            vec![false, false, true, false] // Value 5 is only in list 2
        );
    }

    #[test]
    fn test_list_contains_offset_size_boundary() {
        // Test edge case where offset + size approaches type boundaries
        // We create lists where the last valid index (offset + size - 1) is at various boundaries

        // For u8 boundary
        let elements = Buffer::from_iter(0..256).into_array();
        let offsets = Buffer::from_iter([0u8, 100, 200, 254]).into_array();
        let sizes = Buffer::from_iter([50u8, 50, 54, 2]).into_array(); // Last list goes to index 255

        let list_array =
            ListViewArray::try_new(elements.into_array(), offsets, sizes, Validity::NonNullable)
                .unwrap();

        // Search for value 255 which should only be in the last list
        let search = ConstantArray::new(Scalar::from(255i32), list_array.len());
        let result = list_contains(list_array.as_ref(), search.as_ref()).unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result.to_bool().bool_vec(), vec![false, false, false, true]);

        // Search for value 0 which should only be in the first list
        let search_zero = ConstantArray::new(Scalar::from(0i32), list_array.len());
        let result_zero = list_contains(list_array.as_ref(), search_zero.as_ref()).unwrap();

        assert_eq!(
            result_zero.to_bool().bool_vec(),
            vec![true, false, false, false]
        );
    }
}
