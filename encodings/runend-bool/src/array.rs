use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_array::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use vortex_array::compute::unary::scalar_at;
use vortex_array::compute::{search_sorted, SearchSortedSide};
use vortex_array::encoding::ids;
use vortex_array::stats::{ArrayStatistics, ArrayStatisticsCompute, StatsSet};
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata};
use vortex_array::variants::{ArrayVariants, BoolArrayTrait};
use vortex_array::{
    impl_encoding, Array, ArrayDType, ArrayTrait, Canonical, IntoArray, IntoArrayVariant,
    IntoCanonical,
};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

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
    pub fn try_new(ends: Array, start: bool, validity: Validity) -> VortexResult<Self> {
        let length: usize = scalar_at(&ends, ends.len() - 1)?.as_ref().try_into()?;
        Self::with_offset_and_size(ends, start, validity, length, 0)
    }

    pub(crate) fn with_offset_and_size(
        ends: Array,
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
        if ends.is_empty() {
            vortex_bail!("Ends array must have at least one element");
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

        let mut children = Vec::with_capacity(2);
        children.push(ends);
        if let Some(a) = validity.into_array() {
            children.push(a)
        }

        Self::try_from_parts(dtype, length, metadata, children.into(), StatsSet::new())
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
    pub(crate) fn ends(&self) -> Array {
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
    fn invert(&self) -> VortexResult<Array> {
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

impl ArrayStatisticsCompute for RunEndBoolArray {}

#[cfg(test)]
mod test {
    use vortex_array::array::BoolArray;
    use vortex_array::compute::unary::scalar_at;
    use vortex_array::compute::{slice, take};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayDType, IntoArray, IntoCanonical, ToArray};
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

    fn to_bool_vec(arr: &Array) -> Vec<bool> {
        arr.clone()
            .into_canonical()
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .iter()
            .collect::<Vec<_>>()
    }
}
