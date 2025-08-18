// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compute;
mod serde;

use std::sync::Arc;

#[cfg(feature = "test-harness")]
use itertools::Itertools;
use num_traits::{AsPrimitive, NumCast, PrimInt};
use vortex_dtype::{DType, NativePType, match_each_integer_ptype, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_ensure};
use vortex_scalar::Scalar;

use crate::arrays::PrimitiveVTable;
#[cfg(feature = "test-harness")]
use crate::builders::{ArrayBuilder, ListBuilder};
use crate::compute::{min_max, sub_scalar};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use crate::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};

vtable!(List);

impl VTable for ListVTable {
    type Array = ListArray;
    type Encoding = ListEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.list")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ListEncoding.as_ref())
    }
}

/// A list array that stores variable-length lists of elements, similar to `Vec<Vec<T>>`.
///
/// This mirrors the Apache Arrow List array encoding and provides efficient storage
/// for nested data where each row contains a list of elements of the same type.
///
/// ## Data Layout
///
/// The list array uses an offset-based encoding:
/// - **Elements array**: A flat array containing all list elements concatenated together
/// - **Offsets array**: Integer array where `offsets[i]` is an (inclusive) start index into
///   the **elements** and `offsets[i+1]` is the (exclusive) stop index for the `i`th list.
/// - **Validity**: Optional mask indicating which lists are null
///
/// This allows for excellent cascading compression of the elements and offsets, as similar values
/// are clustered together and the offsets have a predictable pattern and small deltas between
/// consecutive elements.
///
/// ## Offset Semantics
///
/// - Offsets must be non-nullable integers (i32, i64, etc.)
/// - Offsets array has length `n+1` where `n` is the number of lists
/// - List `i` contains elements from `elements[offsets[i]..offsets[i+1]]`  
/// - Offsets must be monotonically increasing
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::{ListArray, PrimitiveArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use std::sync::Arc;
///
/// // Create a list array representing [[1, 2], [3, 4, 5], []]
/// let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
/// let offsets = PrimitiveArray::from_iter([0u32, 2, 5, 5]); // 3 lists
///
/// let list_array = ListArray::try_new(
///     elements.into_array(),
///     offsets.into_array(),
///     Validity::NonNullable,
/// ).unwrap();
///
/// assert_eq!(list_array.len(), 3);
///
/// // Access individual lists
/// let first_list = list_array.elements_at(0);
/// assert_eq!(first_list.len(), 2); // [1, 2]
///
/// let third_list = list_array.elements_at(2);
/// assert!(third_list.is_empty()); // []
/// ```
#[derive(Clone, Debug)]
pub struct ListArray {
    dtype: DType,
    elements: ArrayRef,
    offsets: ArrayRef,
    validity: Validity,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ListEncoding;

pub trait OffsetPType: NativePType + PrimInt + AsPrimitive<usize> + Into<Scalar> {}

impl<T> OffsetPType for T where T: NativePType + PrimInt + AsPrimitive<usize> + Into<Scalar> {}

// A list is valid if the:
// - offsets start at a value in elements
// - offsets are sorted
// - the final offset points to an element in the elements list, pointing to zero
//   if elements are empty.
// - final_offset >= start_offset
// - The size of the validity is the size-1 of the offset array

impl ListArray {
    fn validate(
        elements: &dyn Array,
        offsets: &dyn Array,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Offsets must be of integer type, and cannot go lower than 0.
        vortex_ensure!(
            offsets.dtype().is_int() && !offsets.dtype().is_nullable(),
            "offsets have invalid type {}",
            offsets.dtype()
        );

        // We can safely unwrap the DType as primitive now
        let offsets_ptype = offsets.dtype().as_ptype();

        // Offsets must be sorted (but not strictly sorted, zero-length lists are allowed)
        if let Some(is_sorted) = offsets.statistics().compute_is_sorted() {
            vortex_ensure!(is_sorted, "offsets must be sorted");
        } else {
            vortex_bail!("offsets must report is_sorted statistic");
        }

        // Validate that offsets min is non-negative, and max does not exceed the length of
        // the elements array.
        if let Some(min_max) = min_max(offsets)? {
            match_each_integer_ptype!(offsets_ptype, |P| {
                let max_offset = <P as NumCast>::from(elements.len()).unwrap_or(P::MAX);

                #[allow(clippy::absurd_extreme_comparisons, unused_comparisons)]
                {
                    if let Some(min) = min_max.min.as_primitive().as_::<P>() {
                        vortex_ensure!(
                            min >= 0 && min <= max_offset,
                            "offsets minimum {min} outside valid range [0, {max_offset}]"
                        );
                    }

                    if let Some(max) = min_max.max.as_primitive().as_::<P>() {
                        vortex_ensure!(
                            max >= 0 && max <= max_offset,
                            "offsets maximum {max} outside valid range [0, {max_offset}]"
                        )
                    }
                }
            })
        } else {
            // TODO(aduffy): fallback to slower validation pathway?
            vortex_bail!(
                "offsets array with encoding {} must support min_max compute function",
                offsets.encoding_id()
            );
        };

        // If a validity array is present, it must be the same length as the ListArray
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == offsets.len() - 1,
                "validity with size {validity_len} does not match array size {}",
                offsets.len() - 1
            );
        }

        Ok(())
    }
}

impl ListArray {
    pub fn new(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> Self {
        Self::try_new(elements, offsets, validity).vortex_expect("ListArray new")
    }

    pub fn try_new(
        elements: ArrayRef,
        offsets: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        let nullability = validity.nullability();

        if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
            vortex_bail!(
                "Expected offsets to be an non-nullable integer type, got {:?}",
                offsets.dtype()
            );
        }

        if offsets.is_empty() {
            vortex_bail!("Offsets must have at least one element, [0] for an empty list");
        }

        Self::validate(&elements, &offsets, &validity)?;

        Ok(Self {
            dtype: DType::List(Arc::new(elements.dtype().clone()), nullability),
            elements,
            offsets,
            validity,
            stats_set: Default::default(),
        })
    }

    /// Returns the offset at the given index from the list array.
    ///
    /// Panics if the index is out of bounds.
    pub fn offset_at(&self, index: usize) -> usize {
        assert!(
            index <= self.len(),
            "Index {index} out of bounds 0..={}",
            self.len()
        );

        self.offsets()
            .as_opt::<PrimitiveVTable>()
            .map(|p| match_each_native_ptype!(p.ptype(), |P| { p.as_slice::<P>()[index].as_() }))
            .unwrap_or_else(|| {
                self.offsets()
                    .scalar_at(index)
                    .as_primitive()
                    .as_::<usize>()
                    .vortex_expect("index must fit in usize")
            })
    }

    /// Returns the elements at the given index from the list array.
    pub fn elements_at(&self, index: usize) -> ArrayRef {
        let start = self.offset_at(index);
        let end = self.offset_at(index + 1);
        self.elements().slice(start, end)
    }

    /// Returns elements of the list array referenced by the offsets array
    pub fn sliced_elements(&self) -> ArrayRef {
        let start = self.offset_at(0);
        let end = self.offset_at(self.len());
        self.elements().slice(start, end)
    }

    /// Returns the offsets array.
    pub fn offsets(&self) -> &ArrayRef {
        &self.offsets
    }

    /// Returns the elements array.
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
    }

    /// Create a copy of this array by adjusting offsets to start at 0 and removing elements not referenced by the offsets
    pub fn reset_offsets(&self) -> VortexResult<Self> {
        let elements = self.sliced_elements();
        let offsets = self.offsets();
        let first_offset = offsets.scalar_at(0);
        let adjusted_offsets = sub_scalar(offsets, first_offset)?;

        Self::try_new(elements, adjusted_offsets, self.validity.clone())
    }
}

impl ArrayVTable<ListVTable> for ListVTable {
    fn len(array: &ListArray) -> usize {
        array.offsets.len().saturating_sub(1)
    }

    fn dtype(array: &ListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl OperationsVTable<ListVTable> for ListVTable {
    fn slice(array: &ListArray, start: usize, stop: usize) -> ArrayRef {
        ListArray::new(
            array.elements().clone(),
            array.offsets().slice(start, stop + 1),
            array.validity().slice(start, stop),
        )
        .into_array()
    }

    fn scalar_at(array: &ListArray, index: usize) -> Scalar {
        let elem = array.elements_at(index);
        let scalars: Vec<Scalar> = (0..elem.len()).map(|i| elem.scalar_at(i)).collect();

        Scalar::list(
            Arc::new(elem.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        )
    }
}

impl CanonicalVTable<ListVTable> for ListVTable {
    fn canonicalize(array: &ListArray) -> VortexResult<Canonical> {
        Ok(Canonical::List(array.clone()))
    }
}

impl ValidityHelper for ListArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

#[cfg(feature = "test-harness")]
impl ListArray {
    /// This is a convenience method to create a list array from an iterator of iterators.
    /// This method is slow however since each element is first converted to a scalar and then
    /// appended to the array.
    pub fn from_iter_slow<O: OffsetPType, I: IntoIterator>(
        iter: I,
        dtype: Arc<DType>,
    ) -> VortexResult<ArrayRef>
    where
        I::Item: IntoIterator,
        <I::Item as IntoIterator>::Item: Into<Scalar>,
    {
        let iter = iter.into_iter();
        let mut builder = ListBuilder::<O>::with_capacity(
            dtype.clone(),
            vortex_dtype::Nullability::NonNullable,
            iter.size_hint().0,
        );

        for v in iter {
            let elem = Scalar::list(
                dtype.clone(),
                v.into_iter().map(|x| x.into()).collect_vec(),
                dtype.nullability(),
            );
            builder.append_value(elem.as_list())?
        }
        Ok(builder.finish())
    }

    pub fn from_iter_opt_slow<O: OffsetPType, I: IntoIterator<Item = Option<T>>, T>(
        iter: I,
        dtype: Arc<DType>,
    ) -> VortexResult<ArrayRef>
    where
        T: IntoIterator,
        T::Item: Into<Scalar>,
    {
        let iter = iter.into_iter();
        let mut builder = ListBuilder::<O>::with_capacity(
            dtype.clone(),
            vortex_dtype::Nullability::Nullable,
            iter.size_hint().0,
        );

        for v in iter {
            if let Some(v) = v {
                let elem = Scalar::list(
                    dtype.clone(),
                    v.into_iter().map(|x| x.into()).collect_vec(),
                    dtype.nullability(),
                );
                builder.append_value(elem.as_list())?
            } else {
                builder.append_null()
            }
        }
        Ok(builder.finish())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType::I32;
    use vortex_error::VortexUnwrap;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::arrays::list::ListArray;
    use crate::arrays::{ListVTable, PrimitiveArray};
    use crate::builders::{ArrayBuilder, ListBuilder};
    use crate::compute::filter;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_empty_list_array() {
        let elements = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
        let offsets = PrimitiveArray::from_iter([0]);
        let validity = Validity::AllValid;

        let list =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(0, list.len());
    }

    #[test]
    fn test_simple_list_array() {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let offsets = PrimitiveArray::from_iter([0, 2, 4, 5]);
        let validity = Validity::AllValid;

        let list =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(
            Scalar::list(
                Arc::new(I32.into()),
                vec![1.into(), 2.into()],
                Nullability::Nullable
            ),
            list.scalar_at(0)
        );
        assert_eq!(
            Scalar::list(
                Arc::new(I32.into()),
                vec![3.into(), 4.into()],
                Nullability::Nullable
            ),
            list.scalar_at(1)
        );
        assert_eq!(
            Scalar::list(Arc::new(I32.into()), vec![5.into()], Nullability::Nullable),
            list.scalar_at(2)
        );
    }

    #[test]
    fn test_simple_list_array_from_iter() {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3]);
        let offsets = PrimitiveArray::from_iter([0, 2, 3]);
        let validity = Validity::NonNullable;

        let list =
            ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        let list_from_iter =
            ListArray::from_iter_slow::<u32, _>(vec![vec![1i32, 2], vec![3]], Arc::new(I32.into()))
                .unwrap();

        assert_eq!(list.len(), list_from_iter.len());
        assert_eq!(list.scalar_at(0), list_from_iter.scalar_at(0));
        assert_eq!(list.scalar_at(1), list_from_iter.scalar_at(1));
    }

    #[test]
    fn test_simple_list_filter() {
        let elements = PrimitiveArray::from_option_iter([None, Some(2), Some(3), Some(4), Some(5)]);
        let offsets = PrimitiveArray::from_iter([0, 2, 4, 5]);
        let validity = Validity::AllValid;

        let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
            .unwrap()
            .into_array();

        let filtered = filter(
            &list,
            &Mask::from(BooleanBuffer::from(vec![false, true, true])),
        );

        assert!(filtered.is_ok())
    }

    #[test]
    fn test_offset_to_0() {
        let mut builder =
            ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), Nullability::NonNullable, 5);
        builder
            .append_value(
                Scalar::list(
                    Arc::new(I32.into()),
                    vec![1.into(), 2.into(), 3.into()],
                    Nullability::NonNullable,
                )
                .as_list(),
            )
            .vortex_unwrap();
        builder
            .append_value(
                Scalar::list(
                    Arc::new(I32.into()),
                    vec![4.into(), 5.into(), 6.into()],
                    Nullability::NonNullable,
                )
                .as_list(),
            )
            .vortex_unwrap();
        builder
            .append_value(
                Scalar::list(
                    Arc::new(I32.into()),
                    vec![7.into(), 8.into(), 9.into()],
                    Nullability::NonNullable,
                )
                .as_list(),
            )
            .vortex_unwrap();
        builder
            .append_value(
                Scalar::list(
                    Arc::new(I32.into()),
                    vec![10.into(), 11.into(), 12.into()],
                    Nullability::NonNullable,
                )
                .as_list(),
            )
            .vortex_unwrap();
        builder
            .append_value(
                Scalar::list(
                    Arc::new(I32.into()),
                    vec![13.into(), 14.into(), 15.into()],
                    Nullability::NonNullable,
                )
                .as_list(),
            )
            .vortex_unwrap();
        let list = builder.finish().slice(2, 4);
        let list = list.as_::<ListVTable>().reset_offsets().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list.offsets().len(), 3);
        assert_eq!(list.elements().len(), 6);
        assert_eq!(list.offsets().scalar_at(0), 0u32.into());
    }
}
