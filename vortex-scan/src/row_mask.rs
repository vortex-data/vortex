use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::ops::{BitAnd, RangeBounds};

use vortex_array::array::{BooleanBuffer, PrimitiveArray, SparseArray};
use vortex_array::compute::{and, filter, slice, try_cast, FilterMask};
use vortex_array::validity::{ArrayValidity, LogicalValidity, Validity};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};

/// A RowMask captures a set of selected rows within a range.
///
/// The range itself can be [`u64`], but the length of the range must fit into a [`usize`].
#[derive(Debug, Clone)]
pub struct RowMask {
    mask: FilterMask,
    begin: u64,
    end: u64,
}

// We don't want to implement full logical equality, this naive equality is sufficient for tests.
#[cfg(test)]
impl PartialEq for RowMask {
    fn eq(&self, other: &Self) -> bool {
        self.begin == other.begin && self.end == other.end && self.mask == other.mask
    }
}

impl Display for RowMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowSelector [{}..{}]", self.begin, self.end)
    }
}

impl RowMask {
    pub fn new(mask: FilterMask, begin: u64) -> Self {
        let end = begin + (mask.len() as u64);
        Self { mask, begin, end }
    }

    /// Construct a RowMask which is valid in the given range.
    ///
    /// ## Panics
    ///
    /// If the size of the range is too large to fit into a usize.
    pub fn new_valid_between(begin: u64, end: u64) -> Self {
        let length =
            usize::try_from(end - begin).vortex_expect("Range length does not fit into a usize");
        RowMask::new(FilterMask::from(BooleanBuffer::new_set(length)), begin)
    }

    /// Construct a RowMask which is invalid everywhere in the given range.
    pub fn new_invalid_between(begin: u64, end: u64) -> Self {
        let length =
            usize::try_from(end - begin).vortex_expect("Range length does not fit into a usize");
        RowMask::new(FilterMask::from(BooleanBuffer::new_unset(length)), begin)
    }

    /// Creates a RowMask from an array, only supported boolean and integer types.
    pub fn from_array(array: &ArrayData, begin: u64, end: u64) -> VortexResult<Self> {
        if array.dtype().is_int() {
            Self::from_index_array(array, begin, end)
        } else if array.dtype().is_boolean() {
            Self::from_mask_array(array, begin)
        } else {
            vortex_bail!(
                "RowMask can only be created from integer or boolean arrays, got {} instead.",
                array.dtype()
            );
        }
    }

    /// Construct a RowMask from a Boolean typed array.
    ///
    /// True-valued positions are kept by the returned mask.
    fn from_mask_array(array: &ArrayData, begin: u64) -> VortexResult<Self> {
        match array.logical_validity() {
            LogicalValidity::AllValid(_) => {
                Ok(Self::new(FilterMask::try_from(array.clone())?, begin))
            }
            LogicalValidity::AllInvalid(_) => {
                Ok(Self::new_invalid_between(begin, begin + array.len() as u64))
            }
            LogicalValidity::Array(validity) => {
                let bitmask = and(array.clone(), validity)?;
                Ok(Self::new(FilterMask::try_from(bitmask)?, begin))
            }
        }
    }

    /// Construct a RowMask from an integral array.
    ///
    /// The array values are interpreted as indices and those indices are kept by the returned mask.
    #[allow(clippy::cast_possible_truncation)]
    fn from_index_array(array: &ArrayData, begin: u64, end: u64) -> VortexResult<Self> {
        let length = usize::try_from(end - begin)
            .map_err(|_| vortex_err!("Range length does not fit into a usize"))?;

        let indices =
            try_cast(array, &DType::Primitive(PType::U64, NonNullable))?.into_primitive()?;

        let mask = FilterMask::from_indices(
            length,
            indices
                .as_slice::<u64>()
                .iter()
                .map(|i| *i as usize)
                .collect(),
        );

        Ok(RowMask::new(mask, begin))
    }

    /// Whether the mask is disjoint with the given range.
    ///
    /// This function may return false negatives, but never false positives.
    ///
    /// TODO(ngates): improve this function to take into account the [`FilterMask`].
    pub fn is_disjoint(&self, range: impl RangeBounds<u64>) -> bool {
        use std::ops::Bound;

        // Get the start bound of the input range
        let start = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        // Get the end bound of the input range
        let end = match range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => u64::MAX,
        };

        // Two ranges are disjoint if one ends before the other begins
        self.end <= start || end <= self.begin
    }

    /// Combine the RowMask with bitmask values resulting in new RowMask containing only values true in the bitmask
    pub fn and_bitmask(&self, bitmask: ArrayData) -> VortexResult<Self> {
        // If we are a dense all true bitmap just take the bitmask array
        if self.mask.true_count() == self.len() {
            if bitmask.len() != self.len() {
                vortex_bail!(
                    "Bitmask length {} does not match our length {}",
                    bitmask.len(),
                    self.mask.len()
                );
            }
            Self::from_mask_array(&bitmask, self.begin)
        } else {
            // TODO(robert): Avoid densifying sparse values just to get true indices
            let sparse_mask =
                SparseArray::try_new(self.to_indices_array()?, bitmask, self.len(), false.into())?
                    .into_array()
                    .into_bool()?;
            Self::from_mask_array(sparse_mask.as_ref(), self.begin())
        }
    }

    pub fn and_rowmask(self, other: RowMask) -> VortexResult<Self> {
        if other.true_count() == other.len() {
            return Ok(self);
        }

        // If both masks align perfectly
        if self.begin == other.begin && self.end == other.end {
            return Ok(RowMask::new(self.mask.bitand(&other.mask), self.begin));
        }

        // Disjoint row ranges
        if self.end <= other.begin || self.begin >= other.end {
            return Ok(RowMask::new_invalid_between(
                min(self.begin, other.begin),
                max(self.end, other.end),
            ));
        }

        let output_begin = min(self.begin, other.begin);
        let output_end = max(self.end, other.end);
        let output_len = usize::try_from(output_end - output_begin)
            .map_err(|_| vortex_err!("Range length does not fit into a usize"))?;

        let output_mask = FilterMask::from_intersection_indices(
            output_len,
            self.mask
                .indices()
                .iter()
                .copied()
                .map(|v| v as u64 + self.begin - output_begin)
                .map(|v| usize::try_from(v).vortex_expect("mask index must fit into usize")),
            other
                .mask
                .indices()
                .iter()
                .copied()
                .map(|v| v as u64 + other.begin - output_begin)
                .map(|v| usize::try_from(v).vortex_expect("mask index must fit into usize")),
        );

        Ok(Self::new(output_mask, output_begin))
    }

    pub fn is_all_false(&self) -> bool {
        self.mask.true_count() == 0
    }

    pub fn begin(&self) -> u64 {
        self.begin
    }

    pub fn end(&self) -> u64 {
        self.end
    }

    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Returns the [`FilterMask`] whose true values are relative to the range of this `RowMask`.
    pub fn filter_mask(&self) -> &FilterMask {
        &self.mask
    }

    /// Limit mask to `[begin..end)` range
    pub fn slice(&self, begin: u64, end: u64) -> VortexResult<Self> {
        let range_begin = max(self.begin, begin);
        let range_end = min(self.end, end);
        Ok(RowMask::new(
            if range_begin == self.begin && range_end == self.end {
                self.mask.clone()
            } else {
                self.mask.slice(
                    usize::try_from(range_begin - self.begin)
                        .vortex_expect("we know this must fit into usize"),
                    usize::try_from(range_end - range_begin)
                        .vortex_expect("we know this must fit into usize"),
                )
            },
            range_begin,
        ))
    }

    /// Filter array with this `RowMask`.
    ///
    /// This function assumes that Array is no longer than the mask length and that the mask starts on same offset as the array,
    /// i.e. the beginning of the array corresponds to the beginning of the mask with begin = 0
    pub fn filter_array(&self, array: impl AsRef<ArrayData>) -> VortexResult<Option<ArrayData>> {
        let true_count = self.mask.true_count();
        if true_count == 0 {
            return Ok(None);
        }

        let array = array.as_ref();

        let sliced = if self.len() == array.len() {
            array
        } else {
            // TODO(ngates): I thought the point was the array only covers the valid row range of
            //  the mask?
            // FIXME(ngates): this is made more obvious by the unsafe u64 cast.
            &slice(
                array,
                usize::try_from(self.begin).vortex_expect("TODO(ngates): fix this bad cast"),
                usize::try_from(self.end).vortex_expect("TODO(ngates): fix this bad cast"),
            )?
        };

        if true_count == sliced.len() {
            return Ok(Some(sliced.clone()));
        }

        filter(sliced, &self.mask).map(Some)
    }

    fn to_indices_array(&self) -> VortexResult<ArrayData> {
        Ok(PrimitiveArray::new(
            self.mask
                .indices()
                .iter()
                .map(|i| *i as u64)
                .collect::<Buffer<u64>>(),
            Validity::NonNullable,
        )
        .into_array())
    }

    /// Shift the [`RowMask`] down by the given offset.
    pub fn shift(self, offset: u64) -> VortexResult<RowMask> {
        let valid_shift = self.begin >= offset;
        if !valid_shift {
            vortex_bail!(
                "Can shift RowMask by at most {}, tried to shift by {offset}",
                self.begin
            )
        }
        Ok(RowMask::new(self.mask, self.begin - offset))
    }

    // Get the true count of the underlying mask.
    pub fn true_count(&self) -> usize {
        self.mask.true_count()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::FilterMask;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::{buffer, Buffer};
    use vortex_error::VortexUnwrap;

    use super::*;

    #[rstest]
    #[case(
        RowMask::new(FilterMask::from_iter([true, true, true, false, false, false, false, false, true, true]), 0), (0, 1),
        RowMask::new(FilterMask::from_iter([true]), 0))]
    #[case(
        RowMask::new(FilterMask::from_iter([false, false, false, false, false, true, true, true, true, true]), 0), (2, 5),
        RowMask::new(FilterMask::from_iter([false, false, false]), 2)
    )]
    #[case(
        RowMask::new(FilterMask::from_iter([true, true, true, true, false, false, false, false, false, false]), 0), (2, 5),
        RowMask::new(FilterMask::from_iter([true, true, false]), 2)
    )]
    #[case(
        RowMask::new(FilterMask::from_iter([true, true, true, false, false, true, true, false, false, false]), 0), (2, 6),
        RowMask::new(FilterMask::from_iter([true, false, false, true]), 2))]
    #[case(
        RowMask::new(FilterMask::from_iter([false, false, false, false, false, true, true, true, true, true]), 0), (7, 11),
        RowMask::new(FilterMask::from_iter([true, true, true]), 7))]
    #[case(
        RowMask::new(FilterMask::from_iter([false, true, true, true, true, true]), 3), (0, 5),
        RowMask::new(FilterMask::from_iter([false, true]), 3))]
    #[cfg_attr(miri, ignore)]
    fn slice(#[case] first: RowMask, #[case] range: (u64, u64), #[case] expected: RowMask) {
        assert_eq!(first.slice(range.0, range.1).vortex_unwrap(), expected);
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn shift_invalid() {
        RowMask::new(FilterMask::from_iter([true, true, true, true, true]), 5)
            .shift(7)
            .unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shift() {
        assert_eq!(
            RowMask::new(FilterMask::from_iter([true, true, true, true, true]), 5)
                .shift(5)
                .unwrap(),
            RowMask::new(FilterMask::from_iter([true, true, true, true, true]), 0)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter_array() {
        let mask = RowMask::new(
            FilterMask::from_iter([
                false, false, false, false, false, true, true, true, true, true,
            ]),
            0,
        );
        let array = Buffer::from_iter(0..20).into_array();
        let filtered = mask.filter_array(array).unwrap().unwrap();
        assert_eq!(
            filtered.into_primitive().unwrap().as_slice::<i32>(),
            (5..10).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic]
    fn test_row_mask_type_validation() {
        let array = PrimitiveArray::new(buffer![1.0, 2.0], Validity::AllInvalid).into_array();
        RowMask::from_array(&array, 0, 2).unwrap();
    }

    #[test]
    fn test_and_rowmap_disjoint() {
        let a = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();
        let b = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            15,
            20,
        )
        .unwrap();

        let output = a.and_rowmask(b).unwrap();

        assert_eq!(output.begin, 0);
        assert_eq!(output.end, 20);
        assert!(output.is_all_false());
    }

    #[test]
    fn test_and_rowmap_aligned() {
        let a = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();
        let b = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 7], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();

        let output = a.and_rowmask(b).unwrap();

        assert_eq!(output.begin, 0);
        assert_eq!(output.end, 10);
        assert_eq!(output.true_count(), 2);
    }

    #[test]
    fn test_and_rowmap_intersect() {
        let a = RowMask::from_array(
            PrimitiveArray::new(buffer![1, 2, 3], Validity::AllValid).as_ref(),
            0,
            10,
        )
        .unwrap();
        let b = RowMask::from_array(
            PrimitiveArray::new(buffer!(1, 2, 7), Validity::AllValid).as_ref(),
            5,
            15,
        )
        .unwrap();

        let output = a.and_rowmask(b).unwrap();

        assert_eq!(output.begin, 0);
        assert_eq!(output.end, 15);
        assert_eq!(output.true_count(), 0);
    }
}
