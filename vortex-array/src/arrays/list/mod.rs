// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compute;
mod serde;

use std::ops::Range;
use std::sync::Arc;

#[cfg(feature = "test-harness")]
use itertools::Itertools;
use num_traits::{AsPrimitive, PrimInt};
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
    type PipelineVTable = NotSupported;
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

impl ListArray {
    /// Creates a new [`ListArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`ListArray::new_unchecked`].
    pub fn new(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> Self {
        Self::try_new(elements, offsets, validity).vortex_expect("ListArray new")
    }

    /// Constructs a new `ListArray`.
    ///
    /// See [`ListArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`ListArray::new_unchecked`].
    pub fn try_new(
        elements: ArrayRef,
        offsets: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&elements, &offsets, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(elements, offsets, validity) })
    }

    /// Creates a new [`ListArray`] without validation from these components:
    ///
    /// * `elements` is a flat array containing all list elements concatenated.
    /// * `offsets` is an integer array where `offsets[i]` is the start index for list `i`.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - Offsets must be a non-nullable integer array.
    /// - Offsets must have at least one element (even for empty lists, it should contain \[0\]).
    /// - Offsets must be sorted (monotonically increasing).
    /// - All offset values must be non-negative.
    /// - The maximum offset must not exceed `elements.len()`.
    /// - If validity is an array, its length must equal `offsets.len() - 1`.
    pub unsafe fn new_unchecked(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> Self {
        Self {
            dtype: DType::List(Arc::new(elements.dtype().clone()), validity.nullability()),
            elements,
            offsets,
            validity,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`ListArray`].
    ///
    /// This function checks all the invariants required by [`ListArray::new_unchecked`].
    pub(crate) fn validate(
        elements: &dyn Array,
        offsets: &dyn Array,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Offsets must have at least one element
        vortex_ensure!(
            !offsets.is_empty(),
            "Offsets must have at least one element, [0] for an empty list"
        );

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
                let max_offset = P::try_from(offsets.scalar_at(offsets.len() - 1))
                    .vortex_expect("Offsets type must fit offsets values");

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

                vortex_ensure!(
                    max_offset
                        <= P::try_from(elements.len())
                            .vortex_expect("Offsets type must be able to fit elements length"),
                    "Max offset {max_offset} is beyond the length of the elements array {}",
                    elements.len()
                );
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
        self.elements().slice(start..end)
    }

    /// Returns elements of the list array referenced by the offsets array
    pub fn sliced_elements(&self) -> ArrayRef {
        let start = self.offset_at(0);
        let end = self.offset_at(self.len());
        self.elements().slice(start..end)
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
    fn slice(array: &ListArray, range: Range<usize>) -> ArrayRef {
        ListArray::new(
            array.elements().clone(),
            array.offsets().slice(range.start..range.end + 1),
            array.validity().slice(range),
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
    fn canonicalize(array: &ListArray) -> Canonical {
        Canonical::List(array.clone())
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
mod tests;
