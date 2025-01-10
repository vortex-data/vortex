use std::cmp::Ordering;

use arrow_buffer::BooleanBufferBuilder;
use arrow_ord::ord::make_comparator;
use arrow_schema::SortOptions;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::struct_::StructArray;
use crate::array::{BoolArray, StructEncoding};
use crate::compute::{
    filter, scalar_at, slice, take, CompareFn, ComputeVTable, FilterFn, FilterMask, Operator,
    ScalarAtFn, SliceFn, TakeFn,
};
use crate::variants::StructArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical};

impl ComputeVTable for StructEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<StructArray> for StructEncoding {
    fn scalar_at(&self, array: &StructArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            array.dtype().clone(),
            array
                .children()
                .map(|field| scalar_at(&field, index))
                .try_collect()?,
        ))
    }
}

impl TakeFn<StructArray> for StructEncoding {
    fn take(&self, array: &StructArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        StructArray::try_new(
            array.names().clone(),
            array
                .children()
                .map(|field| take(&field, indices))
                .try_collect()?,
            indices.len(),
            array.validity().take(indices)?,
        )
        .map(|a| a.into_array())
    }
}

impl SliceFn<StructArray> for StructEncoding {
    fn slice(&self, array: &StructArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let fields = array
            .children()
            .map(|field| slice(&field, start, stop))
            .try_collect()?;
        StructArray::try_new(
            array.names().clone(),
            fields,
            stop - start,
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}

impl FilterFn<StructArray> for StructEncoding {
    fn filter(&self, array: &StructArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = array.validity().filter(&mask)?;

        let fields: Vec<ArrayData> = array
            .children()
            .map(|field| filter(&field, mask.clone()))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .unwrap_or_else(|| mask.true_count());

        StructArray::try_new(array.names().clone(), fields, length, validity)
            .map(|a| a.into_array())
    }
}

impl CompareFn<StructArray> for StructEncoding {
    fn compare(
        &self,
        lhs: &StructArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        match StructArray::try_from(rhs.clone()) {
            Err(_) => Ok(None),
            Ok(rhs) => {
                let arrow_lhs = lhs.clone().into_arrow()?;
                let arrow_rhs = rhs.into_arrow()?;

                let cmp_fn = make_comparator(&arrow_lhs, &arrow_rhs, SortOptions::default())?;
                let ordering_fn = ordering_to_bool_fn(operator);

                let mut bool_builder = BooleanBufferBuilder::new(arrow_lhs.len());

                for idx in 0..arrow_lhs.len() {
                    let o = cmp_fn(idx, idx);

                    bool_builder.append(ordering_fn(o));
                }

                Ok(Some(
                    BoolArray::new(bool_builder.finish(), lhs.dtype().nullability()).into_array(),
                ))
            }
        }
    }
}

fn ordering_to_bool_fn(op: Operator) -> impl Fn(Ordering) -> bool {
    match op {
        Operator::Eq => |o: Ordering| o.is_eq(),
        Operator::NotEq => |o: Ordering| o.is_ne(),
        Operator::Gt => |o: Ordering| o.is_gt(),
        Operator::Gte => |o: Ordering| o.is_eq() | o.is_gt(),
        Operator::Lt => |o: Ordering| o.is_lt(),
        Operator::Lte => |o: Ordering| o.is_lt() | o.is_eq(),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::array::StructArray;
    use crate::compute::{compare, filter, FilterMask};
    use crate::validity::Validity;
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = vec![
            false, true, false, true, false, true, false, true, false, true,
        ];
        let filtered = filter(struct_arr.as_ref(), FilterMask::from_iter(mask)).unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = filter(struct_arr.as_ref(), FilterMask::from_iter::<[bool; 0]>([])).unwrap();
        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn basic_compare_test() {
        let n1 = buffer![1u32, 2, 3, 4].into_array();
        let n2 = buffer![1i32, 2, 3, 4].into_array();

        let st1 = StructArray::from_fields(&[("n1", n1.clone()), ("n2", n2)]).unwrap();

        let r = compare(&st1, &st1, Operator::Eq).unwrap();
        let true_count = r.into_bool().unwrap().boolean_buffer().count_set_bits();
        assert_eq!(true_count, st1.len());
    }
}
