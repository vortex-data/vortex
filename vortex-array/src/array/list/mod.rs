use std::fmt::Display;
use std::sync::Arc;
use crate::encoding::{ids};
use crate::{impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayVariant, IntoCanonical};
use serde::{Deserialize, Serialize};
use vortex_dtype::{DType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use crate::array::PrimitiveArray;
use crate::compute::{slice, ComputeVTable};
use crate::stats::{ArrayStatistics, Stat, StatisticsVTable, StatsSet};
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::ArrayVariants;
use crate::visitor::{ArrayVisitor, VisitorVTable};

impl_encoding!("vortex.list", ids::LIST, List);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListMetadata {
    validity: ValidityMetadata,
    element_len: usize,
    offset_dtype: DType,
}

impl Display for ListMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListMetadata")
    }
}

impl ListArray {
    pub fn try_new(elements: ArrayData, offsets: ArrayData, validity: Validity) -> VortexResult<Self> {
        let nullability = validity.nullability();
        let list_len = offsets.len() - 1;
        let element_len = elements.len();
        let offset_dtype = offsets.dtype().clone();

        let validity_metadata = validity.to_metadata(list_len)?;

        if !offsets.dtype().is_int() || offsets.dtype().is_nullable() {
            vortex_bail!("Expected offsets to be an non-nullable integer type, got {:?}", offsets.dtype());
        }

        // A list is valid if the:
        // - offsets start at 0
        // - offsets are sorted
        // - the final offset points to the final element in the elements list, pointing to zero
        //   if elements are empty

        if offsets.is_empty() {
            vortex_bail!("Offsets must have at least one element, [0] for an empty list");
        }

        let Some(min_value) = offsets.statistics().compute_as_cast::<i64>(Stat::Min) else {
            unreachable!("Array must have min value");
        };
        if min_value != 0 {
            vortex_bail!("Expected smallest value in offsets array to be 0, however it was {}", min_value);
        }


        if offsets.dtype().is_signed_int() {
            let final_idx = slice(&offsets, list_len, list_len + 1).vortex_expect("slice exists").into_primitive().vortex_expect("prim").get_as_cast::<i64>(0);
            if final_idx != element_len as i64 {
                vortex_bail!("Expected final to point to final element of elements list, however final idx=({}) and final element idx=({})", final_idx, element_len);
            }
        } else if offsets.dtype().is_unsigned_int() {
            let final_idx = slice(&offsets, list_len, list_len + 1).vortex_expect("slice exists").into_primitive().vortex_expect("prim").get_as_cast::<u64>(0);
            if final_idx != element_len as u64 {
                vortex_bail!("Expected final to point to final element of elements list, however final idx=({}) and element len=({})", final_idx, elements.len());
            }
        } else {
            unreachable!("Offsets are integers");
        }


        let is_sorted = offsets.statistics().compute_is_sorted();
        if !is_sorted.unwrap_or(false) {
            vortex_bail!("Expected offsets to be sorted, got {:?}",is_sorted);
        }

        let list_dtype = DType::List(Arc::new(elements.dtype().clone()), nullability);

        let mut children = vec![elements, offsets];
        if let Some(val) = validity.into_array() {
            children.push(val);
        }


        Self::try_from_parts(
            list_dtype,
            list_len,
            ListMetadata {
                validity: validity_metadata,
                element_len,
                offset_dtype,
            },
            children.into(),
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

    fn index(&self, index: usize) -> Option<ArrayData> {
        if index >= self.len() {
            return None;
        }
        if !self.is_valid(index) {
            return None;
        }
        let offsets = self.offsets();
        let start = offsets.get_as_cast::<i64>(index);
        let end = offsets.get_as_cast::<i64>(index + 1);

        slice(self.elements(), start as usize, end as usize).ok()
    }

    fn offsets(&self) -> PrimitiveArray {
        // TODO: find cheep transform
        self.as_ref().child(1, &self.metadata().offset_dtype, self.len() + 1).vortex_expect("array contains offsets").into_primitive().vortex_expect("offsets are primitive")
    }

    fn elements(&self) -> ArrayData {
        self.as_ref().child(0, self.dtype().as_list().vortex_expect("must be list dtype"), self.metadata().element_len).vortex_expect("array contains elements")
    }
}

impl ArrayVariants for ListArray {}

impl ArrayTrait for ListArray {}

impl ComputeVTable for ListEncoding {}

impl VisitorVTable<ListArray> for ListEncoding {
    fn accept(&self, _array: &ListArray, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        todo!()
    }
}

impl IntoCanonical for ListArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        todo!()
    }
}

impl StatisticsVTable<ListArray> for ListEncoding {
    fn compute_statistics(&self, _array: &ListArray, _stat: Stat) -> VortexResult<StatsSet> {
        todo!()
    }
}

impl ValidityVTable<ListArray> for ListEncoding {
    fn is_valid(&self, array: &ListArray, index: usize) -> bool {
        array.is_valid(index)
    }

    fn logical_validity(&self, array: &ListArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::ArrowNativeType;
    use itertools::Itertools;
    use vortex_dtype::NativePType;
    use crate::array::list::ListArray;
    use crate::array::{BoolArray, PrimitiveArray};
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};
    use crate::accessor::ArrayAccessor;
    use crate::validity::Validity;

    fn idx_into_slice<T: NativePType + ArrowNativeType>(list: &ListArray, idx: usize) -> Vec<T> {
        let binding = list.index(idx).unwrap().into_primitive().unwrap();
        binding.into_maybe_null_slice::<T>()
    }

    fn idx_into_opt_slice_opt<T: NativePType + ArrowNativeType>(list: &ListArray, idx: usize) -> Option<Vec<Option<T>>> {
        let binding = list.index(idx)?.into_primitive().unwrap();
        Some(binding.with_iterator(|iter| iter.map(|i| i.cloned()).collect_vec()).unwrap())
    }

    #[test]
    fn test_empty_list_array() {
        let elements = PrimitiveArray::from(vec![] as Vec<u32>);
        let offsets = PrimitiveArray::from(vec![0]);
        let validity = Validity::AllValid;

        let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(0, list.len());
    }

    #[test]
    fn test_simple_list_array() {
        let elements = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5]);
        let offsets = PrimitiveArray::from(vec![0, 2, 4, 5]);
        let validity = Validity::AllValid;

        let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(vec![1, 2], idx_into_slice::<i32>(&list, 0));
        assert_eq!(vec![3, 4], idx_into_slice::<i32>(&list, 1));
        assert_eq!(vec![5], idx_into_slice::<i32>(&list, 2));
    }

    #[test]
    fn test_list_empty_elem_array() {
        let elements = PrimitiveArray::from(vec![1]);
        let offsets = PrimitiveArray::from(vec![0, 0, 1, 1]);

        let list = ListArray::try_new(elements.into_array(), offsets.into_array(), Validity::AllValid).unwrap();

        assert_eq!(Some(vec![]), idx_into_opt_slice_opt::<i32>(&list, 0));
        assert_eq!(Some(vec![Some(1)]), idx_into_opt_slice_opt::<i32>(&list, 1));
        assert_eq!(Some(vec![]), idx_into_opt_slice_opt::<i32>(&list, 2));
    }

    #[test]
    fn test_list_validation_array() {
        let elements = PrimitiveArray::from_nullable_vec(vec![None, Some(2), Some(3)]);
        let offsets = PrimitiveArray::from(vec![0, 0, 2, 3]);
        let validity = Validity::Array(BoolArray::from_iter(vec![false, true, true]).into_array());

        let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

        assert_eq!(None, idx_into_opt_slice_opt::<i32>(&list, 0));
        assert_eq!(Some(vec![None, Some(2)]), idx_into_opt_slice_opt::<i32>(&list, 1));
        assert_eq!(Some(vec![Some(3)]), idx_into_opt_slice_opt::<i32>(&list, 2));
    }
}