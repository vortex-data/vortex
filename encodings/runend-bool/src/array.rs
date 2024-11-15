use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use vortex_array::compute::unary::scalar_at;
use vortex_array::compute::{search_sorted, SearchSortedSide};
use vortex_array::encoding::ids;
use vortex_array::stats::{ArrayStatistics, ArrayStatisticsCompute, Stat, StatsSet};
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata};
use vortex_array::variants::{ArrayVariants, BoolArrayTrait, PrimitiveArrayTrait};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayTrait, Canonical, IntoArrayData, IntoArrayVariant,
    IntoCanonical,
};
use vortex_dtype::{match_each_unsigned_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::compress::runend_bool_decode;

impl_encoding!("vortex.runendbool", ids::RUN_END_BOOL, RunEndBool);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEndBoolMetadata {
    start: bool,
    validity: ValidityMetadata,
    ends_ptype: PType,
    num_runs: usize,
    offset: usize,
}

impl Display for RunEndBoolMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl RunEndBoolArray {
    pub fn try_new(ends: ArrayData, start: bool, validity: Validity) -> VortexResult<Self> {
        let length: usize = scalar_at(&ends, ends.len() - 1)?.as_ref().try_into()?;
        Self::with_offset_and_size(ends, start, validity, length, 0)
    }

    pub(crate) fn with_offset_and_size(
        ends: ArrayData,
        start: bool,
        validity: Validity,
        length: usize,
        offset: usize,
    ) -> VortexResult<Self> {
        if !ends.statistics().compute_is_strict_sorted().unwrap_or(true) {
            vortex_bail!("Ends array must be strictly sorted",);
        }
        if !ends.dtype().is_unsigned_int() || ends.dtype().is_nullable() {
            vortex_bail!(
                "Ends array must be an unsigned integer type, got {}",
                ends.dtype()
            );
        }
        if ends.is_empty() && length != 0 {
            vortex_bail!(
                "Ends array cannot be empty when length ({}) is not zero",
                length
            );
        }

        if offset != 0 {
            let first_run_end: usize = scalar_at(&ends, 0)?.as_ref().try_into()?;
            if first_run_end <= offset {
                vortex_bail!("First run end {first_run_end} must be bigger than offset {offset}");
            }
        }

        let dtype = DType::Bool(validity.nullability());
        let ends_ptype = ends.dtype().try_into()?;
        let metadata = RunEndBoolMetadata {
            start,
            validity: validity.to_metadata(length)?,
            ends_ptype,
            num_runs: ends.len(),
            offset,
        };

        let stats = if matches!(validity, Validity::AllValid | Validity::NonNullable) {
            let ends_len = ends.len();
            let is_constant = ends_len <= 1;
            let is_sorted = is_constant || (!start && ends_len == 2);
            let is_strict_sorted =
                (is_constant && length <= 1) || (!is_constant && is_sorted && length == 2);
            let run_count = ends_len;
            let min = start && is_constant; // i.e., true iff all are true
            let max = start || ends_len > 1; // i.e., true iff any are true
            StatsSet::from(HashMap::from([
                (Stat::IsConstant, is_constant.into()),
                (Stat::IsSorted, is_sorted.into()),
                (Stat::IsStrictSorted, is_strict_sorted.into()),
                (Stat::RunCount, run_count.into()),
                (Stat::Min, min.into()),
                (Stat::Max, max.into()),
            ]))
        } else {
            StatsSet::new()
        };

        let mut children = Vec::with_capacity(2);
        children.push(ends);
        if let Some(a) = validity.into_array() {
            children.push(a)
        }

        Self::try_from_parts(dtype, length, metadata, children.into(), stats)
    }

    pub(crate) fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        search_sorted(&self.ends(), index + self.offset(), SearchSortedSide::Right)
            .map(|s| s.to_ends_index(self.ends().len()))
    }

    #[inline]
    pub(crate) fn offset(&self) -> usize {
        self.metadata().offset
    }

    #[inline]
    pub(crate) fn start(&self) -> bool {
        self.metadata().start
    }

    #[inline]
    pub(crate) fn ends(&self) -> ArrayData {
        self.as_ref()
            .child(
                0,
                &self.metadata().ends_ptype.into(),
                self.metadata().num_runs,
            )
            .vortex_expect("RunEndBoolArray is missing its run ends")
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(1, &Validity::DTYPE, self.len())
                .vortex_expect("RunEndBoolArray: validity child")
        })
    }
}

impl BoolArrayTrait for RunEndBoolArray {
    fn invert(&self) -> VortexResult<ArrayData> {
        RunEndBoolArray::try_new(self.ends(), !self.start(), self.validity())
            .map(|a| a.into_array())
    }

    fn maybe_null_indices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a> {
        todo!()
    }

    fn maybe_null_slices_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (usize, usize)> + 'a> {
        todo!()
    }
}

impl ArrayVariants for RunEndBoolArray {
    fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }
}

impl ArrayTrait for RunEndBoolArray {}

impl ArrayValidity for RunEndBoolArray {
    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.validity().to_logical(self.len())
    }
}

impl IntoCanonical for RunEndBoolArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let pends = self.ends().into_primitive()?;
        runend_bool_decode(
            &pends,
            self.start(),
            self.validity(),
            self.offset(),
            self.len(),
        )
        .map(Canonical::Bool)
    }
}

impl AcceptArrayVisitor for RunEndBoolArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("ends", &self.ends())?;
        visitor.visit_validity(&self.validity())
    }
}

impl ArrayStatisticsCompute for RunEndBoolArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        let maybe_scalar: Option<Scalar> = match stat {
            Stat::NullCount => Some(self.validity().null_count(self.len())?.into()),
            Stat::TrueCount => {
                let pends = self.ends().into_primitive()?;
                let mut true_count: u64 = 0;
                let mut prev_end: u64 = 0;
                let mut include = self.start();
                match_each_unsigned_integer_ptype!(pends.ptype(), |$P| {
                    for end in pends.maybe_null_slice::<$P>() {
                        if include {
                            true_count += (*end as u64 - prev_end);
                        }
                        include = !include;
                        prev_end = *end as u64;
                    }
                });
                Some(true_count.into())
            }
            _ => None,
        };
        if let Some(scalar) = maybe_scalar {
            Ok(StatsSet::from(HashMap::from([(stat, scalar)])))
        } else {
            Ok(StatsSet::new())
        }
    }
}

#[cfg(test)]
mod test {
    use core::iter;

    use itertools::Itertools as _;
    use rstest::rstest;
    use vortex_array::array::BoolArray;
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::compute::{slice, take};
    use vortex_array::stats::{ArrayStatistics as _, ArrayStatisticsCompute};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoCanonical, ToArrayData};
    use vortex_dtype::{DType, Nullability};

    use crate::RunEndBoolArray;

    #[test]
    fn new() {
        // [false, false, true, true, false]
        let arr =
            RunEndBoolArray::try_new(vec![2u32, 4, 5].into_array(), false, Validity::NonNullable)
                .unwrap();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.dtype(), &DType::Bool(Nullability::NonNullable));

        assert_eq!(scalar_at(arr.as_ref(), 0).unwrap(), false.into());
        assert_eq!(scalar_at(arr.as_ref(), 2).unwrap(), true.into());
        assert_eq!(scalar_at(arr.as_ref(), 4).unwrap(), false.into());
    }

    #[test]
    fn slice_array() {
        let arr = slice(
            // [t, t, f, f, f, t, f, t, t, t]
            RunEndBoolArray::try_new(
                vec![2u32, 5, 6, 7, 10].into_array(),
                true,
                Validity::NonNullable,
            )
            .unwrap()
            .as_ref(),
            2,
            8,
        )
        .unwrap();
        assert_eq!(arr.dtype(), &DType::Bool(Nullability::NonNullable));

        assert_eq!(
            to_bool_vec(&arr),
            vec![false, false, false, true, false, true],
        );
    }

    #[test]
    fn slice_slice_array() {
        let raw = BoolArray::from(vec![
            true, true, false, false, false, true, false, true, true, true,
        ])
        .to_array();
        let arr = slice(&raw, 2, 8).unwrap();
        assert_eq!(arr.dtype(), &DType::Bool(Nullability::NonNullable));

        assert_eq!(
            to_bool_vec(&arr),
            vec![false, false, false, true, false, true],
        );

        let arr2 = slice(&arr, 3, 6).unwrap();
        assert_eq!(to_bool_vec(&arr2), vec![true, false, true],);

        let arr3 = slice(&arr2, 1, 3).unwrap();
        assert_eq!(to_bool_vec(&arr3), vec![false, true],);
    }

    #[test]
    fn flatten() {
        let arr =
            RunEndBoolArray::try_new(vec![2u32, 4, 5].into_array(), true, Validity::NonNullable)
                .unwrap();

        assert_eq!(
            to_bool_vec(&arr.to_array()),
            vec![true, true, false, false, true]
        );
    }

    #[test]
    fn take_bool() {
        let arr = take(
            RunEndBoolArray::try_new(
                vec![2u32, 4, 5, 10].into_array(),
                true,
                Validity::NonNullable,
            )
            .unwrap(),
            vec![0, 0, 6, 4].into_array(),
        )
        .unwrap();

        assert_eq!(to_bool_vec(&arr), vec![true, true, false, true]);
    }

    fn to_bool_vec(arr: &ArrayData) -> Vec<bool> {
        arr.clone()
            .into_canonical()
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .iter()
            .collect::<Vec<_>>()
    }

    #[rstest]
    #[case(true, 1, 1)]
    #[case(true, 1, 2)]
    #[case(true, 2, 2)]
    #[case(false, 1, 1)]
    #[case(false, 1, 2)]
    #[case(false, 2, 2)]
    #[case(false, 3, 32)]
    #[case(true, 3, 32)]
    fn stats(#[case] start: bool, #[case] ends_len: usize, #[case] len: usize) {
        use vortex_array::stats::Stat;

        let ends = (1u32..(ends_len as u32))
            .chain(iter::once(len as u32))
            .collect_vec();
        assert_eq!(ends.len(), ends_len);
        assert_eq!(*ends.last().unwrap(), len as u32);

        let arr =
            RunEndBoolArray::try_new(ends.into_array(), start, Validity::NonNullable).unwrap();
        let bools = arr.clone().into_canonical().unwrap().into_bool().unwrap();
        for stat in [
            Stat::IsConstant,
            Stat::NullCount,
            Stat::TrueCount,
            Stat::Min,
            Stat::Max,
            Stat::IsSorted,
            Stat::IsStrictSorted,
        ] {
            // call compute_statistics directly to avoid caching
            let bools_stats = bools.compute_statistics(stat).unwrap();
            let expected = bools_stats.get(stat).unwrap();
            let actual = arr.statistics().compute(stat).unwrap();
            assert_eq!(expected, &actual);
        }

        assert_eq!(arr.statistics().compute_run_count(), Some(ends_len));
    }
}
