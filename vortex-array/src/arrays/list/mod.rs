// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compute;
mod serde;

use std::sync::Arc;

use itertools::Itertools;
use num_traits::{AsPrimitive, PrimInt};
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};
use vortex_scalar::Scalar;

use crate::arrays::PrimitiveVTable;
#[cfg(feature = "test-harness")]
use crate::builders::{ArrayBuilder, ListBuilder};
use crate::compute::{is_sorted, min_max};
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

/// The canonical array for List type data. This is modeled similarly to a ListArray.
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
    /// Creates a new ListArray from the provided elements, offsets, and validity.
    ///
    /// A ListArray is valid if all of the following are true:
    /// - Offsets are sorted, non-empty, and non-negative.
    /// - The first offset is less than or equal to the length of the elements array.
    /// - The final offset points to an element in the elements list, pointing to zero if the elements list is empty.
    /// - The size of the validity is the size-1 of the offset array.
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

        // Check that the lengths of the offsets and validity arrays are consistent.
        if offsets.is_empty() {
            vortex_bail!("Offsets must have at least one element, [0] for an empty ListArray");
        }
        if let Some(validity_len) = validity.maybe_len() {
            if validity_len != offsets.len().saturating_sub(1) {
                vortex_bail!(
                    "Validity length must be one less than the offset length, got {} and {}",
                    validity_len,
                    offsets.len()
                );
            }
        }

        // Check that the offsets are sorted, and that the min & max offsets are within bounds.
        if !is_sorted(&offsets)? {
            vortex_bail!("Offsets must be sorted");
        }

        if let Some(min_max_result) = min_max(&offsets)? {
            let min_offset: usize = min_max_result
                .min
                .as_primitive()
                .as_()?
                .ok_or_else(|| vortex_err!("Failed to get min offset as usize"))?;
            if min_offset > elements.len() {
                vortex_bail!(
                    "The first offset must be less than or equal to the length of the elements array, got {} and {}",
                    min_offset,
                    elements.len()
                );
            }

            let max_offset: usize = min_max_result
                .max
                .as_primitive()
                .as_()?
                .ok_or_else(|| vortex_err!("Failed to get max offset as usize"))?;
            if max_offset > elements.len() {
                vortex_bail!(
                    "The max offset must be less than or equal to the length of the elements array, got {} and {}",
                    max_offset,
                    elements.len()
                );
            }
        }

        Ok(Self {
            dtype: DType::List(Arc::new(elements.dtype().clone()), nullability),
            elements,
            offsets,
            validity,
            stats_set: Default::default(),
        })
    }

    // TODO: merge logic with varbin
    // TODO(ngates): should return a result if it requires canonicalizing offsets
    pub fn offset_at(&self, index: usize) -> usize {
        self.offsets
            .as_opt::<PrimitiveVTable>()
            .map(|p| match_each_native_ptype!(p.ptype(), |P| { p.as_slice::<P>()[index].as_() }))
            .unwrap_or_else(|| {
                self.offsets
                    .scalar_at(index)
                    .unwrap_or_else(|err| {
                        vortex_panic!(err, "Failed to get offset at index: {}", index)
                    })
                    .as_ref()
                    .try_into()
                    .vortex_expect("Failed to convert offset to usize")
            })
    }

    // TODO: fetches the elements at index
    pub fn elements_at(&self, index: usize) -> VortexResult<ArrayRef> {
        let start = self.offset_at(index);
        let end = self.offset_at(index + 1);
        self.elements().slice(start, end)
    }

    // TODO: fetches the offsets of the array ignoring validity
    pub fn offsets(&self) -> &ArrayRef {
        &self.offsets
    }

    // TODO: fetches the elements of the array ignoring validity
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
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
    fn slice(array: &ListArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ListArray::try_new(
            array.elements().clone(),
            array.offsets().slice(start, stop + 1)?,
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }

    fn scalar_at(array: &ListArray, index: usize) -> VortexResult<Scalar> {
        let elem = array.elements_at(index)?;
        let scalars: Vec<Scalar> = (0..elem.len()).map(|i| elem.scalar_at(i)).try_collect()?;

        Ok(Scalar::list(
            Arc::new(elem.dtype().clone()),
            scalars,
            array.dtype().nullability(),
        ))
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
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::arrays::PrimitiveArray;
    use crate::arrays::list::ListArray;
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
            list.scalar_at(0).unwrap()
        );
        assert_eq!(
            Scalar::list(
                Arc::new(I32.into()),
                vec![3.into(), 4.into()],
                Nullability::Nullable
            ),
            list.scalar_at(1).unwrap()
        );
        assert_eq!(
            Scalar::list(Arc::new(I32.into()), vec![5.into()], Nullability::Nullable),
            list.scalar_at(2).unwrap()
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
        assert_eq!(
            list.scalar_at(0).unwrap(),
            list_from_iter.scalar_at(0).unwrap()
        );
        assert_eq!(
            list.scalar_at(1).unwrap(),
            list_from_iter.scalar_at(1).unwrap()
        );
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
}
