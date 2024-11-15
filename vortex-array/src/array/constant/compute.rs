use std::cmp::Ordering;

use vortex_dtype::Nullability;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::array::constant::ConstantArray;
use crate::compute::unary::{scalar_at, ScalarAtFn};
use crate::compute::{
    scalar_cmp, AndFn, ArrayCompute, FilterFn, MaybeCompareFn, Operator, OrFn, SearchResult,
    SearchSortedFn, SearchSortedSide, SliceFn, TakeFn,
};
use crate::stats::{ArrayStatistics, Stat};
use crate::{ArrayDType, ArrayData, IntoArrayData};

impl ArrayCompute for ConstantArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }

    fn and(&self) -> Option<&dyn AndFn> {
        Some(self)
    }

    fn or(&self) -> Option<&dyn OrFn> {
        Some(self)
    }
}

impl ScalarAtFn for ConstantArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(<Self as ScalarAtFn>::scalar_at_unchecked(self, index))
    }

    fn scalar_at_unchecked(&self, _index: usize) -> Scalar {
        self.owned_scalar()
    }
}

impl TakeFn for ConstantArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::new(self.owned_scalar(), indices.len()).into_array())
    }
}

impl SliceFn for ConstantArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(Self::new(self.owned_scalar(), stop - start).into_array())
    }
}

impl FilterFn for ConstantArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::new(
            self.owned_scalar(),
            predicate.with_dyn(|p| {
                p.as_bool_array()
                    .ok_or(vortex_err!(
                        NotImplemented: "as_bool_array",
                        predicate.encoding().id()
                    ))
                    .map(|x| x.true_count())
            })?,
        )
        .into_array())
    }
}

impl SearchSortedFn for ConstantArray {
    fn search_sorted(&self, value: &Scalar, side: SearchSortedSide) -> VortexResult<SearchResult> {
        match self
            .scalar_value()
            .partial_cmp(value.value())
            .unwrap_or(Ordering::Less)
        {
            Ordering::Greater => Ok(SearchResult::NotFound(0)),
            Ordering::Less => Ok(SearchResult::NotFound(self.len())),
            Ordering::Equal => match side {
                SearchSortedSide::Left => Ok(SearchResult::Found(0)),
                SearchSortedSide::Right => Ok(SearchResult::Found(self.len())),
            },
        }
    }
}

impl MaybeCompareFn for ConstantArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        (ConstantArray::try_from(other).is_ok()
            || other
                .statistics()
                .get_as::<bool>(Stat::IsConstant)
                .unwrap_or_default())
        .then(|| {
            let lhs = self.owned_scalar();
            let rhs = scalar_at(other, 0).vortex_expect("Expected scalar");
            let scalar = scalar_cmp(&lhs, &rhs, operator);
            Ok(ConstantArray::new(scalar, self.len()).into_array())
        })
    }
}

fn and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    left.zip(right).map(|(l, r)| l & r)
}

fn kleene_and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) => Some(false),
        (_, Some(false)) => Some(false),
        (None, _) => None,
        (_, None) => None,
        (Some(l), Some(r)) => Some(l & r),
    }
}

fn or(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    left.zip(right).map(|(l, r)| l | r)
}

fn kleene_or(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(true), _) => Some(true),
        (_, Some(true)) => Some(true),
        (None, _) => None,
        (_, None) => None,
        (Some(l), Some(r)) => Some(l | r),
    }
}

impl AndFn for ConstantArray {
    fn and(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        constant_array_bool_impl(self, array, and, |other, this| {
            other.with_dyn(|other| other.and().map(|other| other.and(this)))
        })
    }

    fn and_kleene(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        constant_array_bool_impl(self, array, kleene_and, |other, this| {
            other.with_dyn(|other| other.and_kleene().map(|other| other.and_kleene(this)))
        })
    }
}

impl OrFn for ConstantArray {
    fn or(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        constant_array_bool_impl(self, array, or, |other, this| {
            other.with_dyn(|other| other.or().map(|other| other.or(this)))
        })
    }

    fn or_kleene(&self, array: &ArrayData) -> VortexResult<ArrayData> {
        constant_array_bool_impl(self, array, kleene_or, |other, this| {
            other.with_dyn(|other| other.or_kleene().map(|other| other.or_kleene(this)))
        })
    }
}

fn constant_array_bool_impl(
    constant_array: &ConstantArray,
    other: &ArrayData,
    bool_op: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
    fallback_fn: impl Fn(&ArrayData, &ArrayData) -> Option<VortexResult<ArrayData>>,
) -> VortexResult<ArrayData> {
    // If the right side is constant
    if other.statistics().get_as::<bool>(Stat::IsConstant) == Some(true) {
        let lhs = constant_array.scalar_value().as_bool()?;
        let rhs = scalar_at(other, 0)?.value().as_bool()?;

        let scalar = match bool_op(lhs, rhs) {
            Some(b) => Scalar::bool(b, Nullability::Nullable),
            None => Scalar::null(constant_array.dtype().as_nullable()),
        };

        Ok(ConstantArray::new(scalar, constant_array.len()).into_array())
    } else {
        // try and use a the rhs specialized implementation if it exists
        match fallback_fn(other, constant_array.as_ref()) {
            Some(r) => r,
            None => vortex_bail!("Operation is not supported"),
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::array::constant::ConstantArray;
    use crate::array::BoolArray;
    use crate::compute::unary::scalar_at;
    use crate::compute::{and, or, search_sorted, SearchResult, SearchSortedSide};
    use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

    #[test]
    pub fn search() {
        let cst = ConstantArray::new(42, 5000).into_array();
        assert_eq!(
            search_sorted(&cst, 33, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&cst, 55, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(5000)
        );
    }

    #[test]
    pub fn search_equals() {
        let cst = ConstantArray::new(42, 5000).into_array();
        assert_eq!(
            search_sorted(&cst, 42, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            search_sorted(&cst, 42, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(5000)
        );
    }

    #[rstest]
    #[case(ConstantArray::new(true, 4).into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(), ConstantArray::new(true, 4).into_array())]
    fn test_or(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = or(&lhs, &rhs).unwrap().into_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().value().as_bool().unwrap();
        let v1 = scalar_at(&r, 1).unwrap().value().as_bool().unwrap();
        let v2 = scalar_at(&r, 2).unwrap().value().as_bool().unwrap();
        let v3 = scalar_at(&r, 3).unwrap().value().as_bool().unwrap();

        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(v3.unwrap());
    }

    #[rstest]
    #[case(ConstantArray::new(true, 4).into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        ConstantArray::new(true, 4).into_array())]
    fn test_and(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = and(&lhs, &rhs).unwrap().into_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().value().as_bool().unwrap();
        let v1 = scalar_at(&r, 1).unwrap().value().as_bool().unwrap();
        let v2 = scalar_at(&r, 2).unwrap().value().as_bool().unwrap();
        let v3 = scalar_at(&r, 3).unwrap().value().as_bool().unwrap();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }
}
