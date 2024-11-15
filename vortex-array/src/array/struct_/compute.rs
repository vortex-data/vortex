use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::struct_::StructArray;
use crate::compute::unary::{scalar_at, scalar_at_unchecked, ScalarAtFn};
use crate::compute::{filter, slice, take, ArrayCompute, FilterFn, SliceFn, TakeFn};
use crate::stats::ArrayStatistics;
use crate::variants::StructArrayTrait;
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ArrayCompute for StructArray {
    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl ScalarAtFn for StructArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::r#struct(
            self.dtype().clone(),
            self.children()
                .map(|field| scalar_at(&field, index).map(|s| s.into_value()))
                .try_collect()?,
        ))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        Scalar::r#struct(
            self.dtype().clone(),
            self.children()
                .map(|field| scalar_at_unchecked(&field, index).into_value())
                .collect(),
        )
    }
}

impl TakeFn for StructArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        Self::try_new(
            self.names().clone(),
            self.children()
                .map(|field| take(&field, indices))
                .try_collect()?,
            indices.len(),
            self.validity().take(indices)?,
        )
        .map(|a| a.into_array())
    }
}

impl SliceFn for StructArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let fields = self
            .children()
            .map(|field| slice(&field, start, stop))
            .try_collect()?;
        Self::try_new(
            self.names().clone(),
            fields,
            stop - start,
            self.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }
}

impl FilterFn for StructArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        let fields: Vec<ArrayData> = self
            .children()
            .map(|field| filter(&field, predicate))
            .try_collect()?;
        let length = fields
            .first()
            .map(|a| a.len())
            .or_else(|| predicate.statistics().compute_true_count())
            .unwrap_or_default();

        Self::try_new(
            self.names().clone(),
            fields,
            length,
            self.validity().filter(predicate)?,
        )
        .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{BoolArray, StructArray};
    use crate::compute::filter;
    use crate::validity::Validity;

    #[test]
    fn filter_empty_struct() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 10, Validity::NonNullable).unwrap();
        let mask = vec![
            false, true, false, true, false, true, false, true, false, true,
        ];
        let filtered = filter(struct_arr.as_ref(), BoolArray::from(mask)).unwrap();
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn filter_empty_struct_with_empty_filter() {
        let struct_arr =
            StructArray::try_new(vec![].into(), vec![], 0, Validity::NonNullable).unwrap();
        let filtered = filter(struct_arr.as_ref(), BoolArray::from(vec![])).unwrap();
        assert_eq!(filtered.len(), 0);
    }
}
