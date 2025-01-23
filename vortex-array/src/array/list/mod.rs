mod compute;

use std::fmt::Display;
use std::sync::Arc;

#[cfg(feature = "test-harness")]
use itertools::Itertools;
use num_traits::AsPrimitive;
use rkyv::{access, to_bytes};
use serde::{Deserialize, Serialize};
#[cfg(feature = "test-harness")]
use vortex_dtype::Nullability;
use vortex_dtype::{match_each_native_ptype, DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexExpect, VortexResult};
#[cfg(feature = "test-harness")]
use vortex_scalar::Scalar;

use crate::array::PrimitiveArray;
#[cfg(feature = "test-harness")]
use crate::builders::{ArrayBuilder, ListBuilder};
use crate::compute::{scalar_at, slice};
use crate::encoding::ids;
use crate::stats::{StatisticsVTable, StatsSet};
use crate::validate::ValidateVTable;
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{ListArrayTrait, PrimitiveArrayTrait, VariantsVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, Canonical, DeserializeMetadata, IntoCanonical,
    RkyvMetadata,
};

impl_encoding!("vortex.list", ids::LIST, List, RkyvMetadata<ListMetadata>);

#[derive(
    Debug, Clone, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct ListMetadata {
    pub(crate) validity: ValidityMetadata,
    pub(crate) elements_len: usize,
    pub(crate) offset_ptype: PType,
}

impl Display for ListMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListMetadata")
    }
}

// A list is valid if the:
// - offsets start at a value in elements
// - offsets are sorted
// - the final offset points to an element in the elements list, pointing to zero
//   if elements are empty.
// - final_offset >= start_offset
// - The size of the validity is the size-1 of the offset array

impl ListArray {
    pub fn try_new(
        elements: ArrayData,
        offsets: ArrayData,
        validity: Validity,
    ) -> VortexResult<Self> {
        let nullability = validity.nullability();
        let list_len = offsets.len() - 1;
        let element_len = elements.len();

        let validity_metadata = validity.to_metadata(list_len)?;

        if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
            vortex_bail!(
                "Expected offsets to be an non-nullable integer type, got {:?}",
                offsets.dtype()
            );
        }

        if offsets.is_empty() {
            vortex_bail!("Offsets must have at least one element, [0] for an empty list");
        }

        let offset_ptype = PType::try_from(offsets.dtype())?;

        let list_dtype = DType::List(Arc::new(elements.dtype().clone()), nullability);

        let mut children = vec![elements, offsets];
        if let Some(val) = validity.into_array() {
            children.push(val);
        }

        Self::try_from_parts(
            list_dtype,
            list_len,
            RkyvMetadata(ListMetadata {
                validity: validity_metadata,
                elements_len: element_len,
                offset_ptype,
            }),
            None,
            Some(children.into()),
            StatsSet::default(),
        )
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(2, &Validity::DTYPE, self.len())
                .vortex_expect("ListArray: validity child")
        })
    }

    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    // TODO: merge logic with varbin
    pub fn offset_at(&self, index: usize) -> usize {
        PrimitiveArray::try_from(self.offsets())
            .ok()
            .map(|p| {
                match_each_native_ptype!(p.ptype(), |$P| {
                    p.as_slice::<$P>()[index].as_()
                })
            })
            .unwrap_or_else(|| {
                scalar_at(self.offsets(), index)
                    .unwrap_or_else(|err| {
                        vortex_panic!(err, "Failed to get offset at index: {}", index)
                    })
                    .as_ref()
                    .try_into()
                    .vortex_expect("Failed to convert offset to usize")
            })
    }

    // TODO: fetches the elements at index
    pub fn elements_at(&self, index: usize) -> VortexResult<ArrayData> {
        let start = self.offset_at(index);
        let end = self.offset_at(index + 1);
        slice(self.elements(), start, end)
    }

    // TODO: fetches the offsets of the array ignoring validity
    pub fn offsets(&self) -> ArrayData {
        // TODO: find cheap transform
        self.as_ref()
            .child(1, &self.metadata().offset_ptype.into(), self.len() + 1)
            .vortex_expect("array contains offsets")
    }

    // TODO: fetches the elements of the array ignoring validity
    pub fn elements(&self) -> ArrayData {
        let dtype = self
            .dtype()
            .as_list_element()
            .vortex_expect("must be list dtype");
        self.as_ref()
            .child(0, dtype, self.metadata().elements_len)
            .vortex_expect("array contains elements")
    }
}

impl VariantsVTable<ListArray> for ListEncoding {
    fn as_list_array<'a>(&self, array: &'a ListArray) -> Option<&'a dyn ListArrayTrait> {
        Some(array)
    }
}

impl ValidateVTable<ListArray> for ListEncoding {}

impl VisitorVTable<ListArray> for ListEncoding {
    fn accept(&self, array: &ListArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("offsets", &array.offsets())?;
        visitor.visit_child("elements", &array.elements())?;
        visitor.visit_validity(&array.validity())
    }
}

impl IntoCanonical for ListArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::List(self))
    }
}

impl StatisticsVTable<ListArray> for ListEncoding {}

impl ListArrayTrait for ListArray {}

impl ValidityVTable<ListArray> for ListEncoding {
    fn is_valid(&self, array: &ListArray, index: usize) -> bool {
        array.is_valid(index)
    }

    fn logical_validity(&self, array: &ListArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

#[cfg(feature = "test-harness")]
impl ListArray {
    /// This is a convenience method to create a list array from an iterator of iterators.
    /// This method is slow however since each element is first converted to a scalar and then
    /// appended to the array.
    pub fn from_iter_slow<I: IntoIterator>(iter: I, dtype: Arc<DType>) -> VortexResult<ArrayData>
    where
        I::Item: IntoIterator,
        <I::Item as IntoIterator>::Item: Into<Scalar>,
    {
        let iter = iter.into_iter();
        let mut builder = ListBuilder::<u32>::with_capacity(
            dtype.clone(),
            Nullability::NonNullable,
            iter.size_hint().0,
        );

        for v in iter {
            let elem = Scalar::list(
                dtype.clone(),
                v.into_iter().map(|x| x.into()).collect_vec(),
                Nullability::Nullable,
            );
            builder.append_value(elem.as_list())?
        }
        builder.finish()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use vortex_dtype::Nullability;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::array::list::ListArray;
    use crate::array::PrimitiveArray;
    use crate::compute::{filter, scalar_at};
    use crate::validity::Validity;
    use crate::{ArrayLen, IntoArrayData};

    #[test]
    fn test_empty_list_array() {
        let elements = PrimitiveArray::empty::<u32>(NonNullable);
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
            scalar_at(&list, 0).unwrap()
        );
        assert_eq!(
            Scalar::list(
                Arc::new(I32.into()),
                vec![3.into(), 4.into()],
                Nullability::Nullable
            ),
            scalar_at(&list, 1).unwrap()
        );
        assert_eq!(
            Scalar::list(Arc::new(I32.into()), vec![5.into()], Nullability::Nullable),
            scalar_at(&list, 2).unwrap()
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
            ListArray::from_iter_slow(vec![vec![1i32, 2], vec![3]], Arc::new(I32.into())).unwrap();

        assert_eq!(list.len(), list_from_iter.len());
        assert_eq!(
            scalar_at(&list, 0).unwrap(),
            scalar_at(&list_from_iter, 0).unwrap()
        );
        assert_eq!(
            scalar_at(&list, 1).unwrap(),
            scalar_at(&list_from_iter, 1).unwrap()
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
