use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::ops::{Range, RangeBounds};

use vortex_array::compute::{filter, slice, try_cast};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;

/// A RowMask captures a set of selected rows within a range.
///
/// The range itself can be [`u64`], but the length of the range must fit into a [`usize`], this
/// allows us to use a `usize` filter mask within a much larger file.
#[derive(Debug, Clone)]
pub struct RowMask {
    mask: Mask,
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
    /// Define a new [`RowMask`] with the given mask and offset into the file.
    pub fn new(mask: Mask, begin: u64) -> Self {
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
        RowMask::new(Mask::new_true(length), begin)
    }

    /// Construct a RowMask which is invalid everywhere in the given range.
    pub fn new_invalid_between(begin: u64, end: u64) -> Self {
        let length =
            usize::try_from(end - begin).vortex_expect("Range length does not fit into a usize");
        RowMask::new(Mask::new_false(length), begin)
    }

    /// Creates a RowMask from an array, only supported boolean and integer types.
    pub fn from_array(array: &dyn Array, begin: u64, end: u64) -> VortexResult<Self> {
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
    fn from_mask_array(array: &dyn Array, begin: u64) -> VortexResult<Self> {
        Ok(Self::new(array.validity_mask()?, begin))
    }

    /// Construct a RowMask from an integral array.
    ///
    /// The array values are interpreted as indices and those indices are kept by the returned mask.
    #[allow(clippy::cast_possible_truncation)]
    fn from_index_array(array: &dyn Array, begin: u64, end: u64) -> VortexResult<Self> {
        let length = usize::try_from(end - begin)
            .map_err(|_| vortex_err!("Range length does not fit into a usize"))?;

        let indices =
            try_cast(array, &DType::Primitive(PType::U64, NonNullable))?.to_primitive()?;

        let mask = Mask::from_indices(
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
    /// TODO(ngates): improve this function to take into account the [`Mask`].
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

    /// The beginning of the masked range.
    #[inline]
    pub fn begin(&self) -> u64 {
        self.begin
    }

    /// The end of the masked range.
    #[inline]
    pub fn end(&self) -> u64 {
        self.end
    }

    /// The length of the mask is the number of possible rows between the `begin` and `end`,
    /// regardless of how many appear in the mask. For the number of masked rows, see `true_count`.
    #[inline]
    // There is good definition of is_empty, does it mean len == 0 or true_count == 0?
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Returns the [`Mask`] whose true values are relative to the range of this `RowMask`.
    pub fn filter_mask(&self) -> &Mask {
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
    pub fn filter_array(&self, array: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        let true_count = self.mask.true_count();
        if true_count == 0 {
            return Ok(None);
        }

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
            return Ok(Some(sliced.to_array()));
        }

        filter(sliced, &self.mask).map(Some)
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

    /// The number of masked rows within the range.
    pub fn true_count(&self) -> usize {
        self.mask.true_count()
    }
}

pub fn range_intersection(range: &Range<u64>, row_indices: &Buffer<u64>) -> Option<Range<usize>> {
    if row_indices.first().is_some_and(|&first| first >= range.end)
        || row_indices.last().is_some_and(|&last| range.start >= last)
    {
        return None;
    }

    // For the given row range, find the indices that are within the row_indices.
    let start_idx = row_indices
        .binary_search(&range.start)
        .unwrap_or_else(|x| x);
    let end_idx = row_indices.binary_search(&range.end).unwrap_or_else(|x| x);
    (start_idx != end_idx).then_some(start_idx..end_idx)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::{Buffer, buffer};
    use vortex_error::VortexUnwrap;
    use vortex_mask::Mask;

    use super::*;

    #[rstest]
    #[case(
        RowMask::new(Mask::from_iter([true, true, true, false, false, false, false, false, true, true]), 0), (0, 1),
        RowMask::new(Mask::from_iter([true]), 0))]
    #[case(
        RowMask::new(Mask::from_iter([false, false, false, false, false, true, true, true, true, true]), 0), (2, 5),
        RowMask::new(Mask::from_iter([false, false, false]), 2)
    )]
    #[case(
        RowMask::new(Mask::from_iter([true, true, true, true, false, false, false, false, false, false]), 0), (2, 5),
        RowMask::new(Mask::from_iter([true, true, false]), 2)
    )]
    #[case(
        RowMask::new(Mask::from_iter([true, true, true, false, false, true, true, false, false, false]), 0), (2, 6),
        RowMask::new(Mask::from_iter([true, false, false, true]), 2))]
    #[case(
        RowMask::new(Mask::from_iter([false, false, false, false, false, true, true, true, true, true]), 0), (7, 11),
        RowMask::new(Mask::from_iter([true, true, true]), 7))]
    #[case(
        RowMask::new(Mask::from_iter([false, true, true, true, true, true]), 3), (0, 5),
        RowMask::new(Mask::from_iter([false, true]), 3))]
    #[cfg_attr(miri, ignore)]
    fn slice(#[case] first: RowMask, #[case] range: (u64, u64), #[case] expected: RowMask) {
        assert_eq!(first.slice(range.0, range.1).vortex_unwrap(), expected);
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn shift_invalid() {
        RowMask::new(Mask::from_iter([true, true, true, true, true]), 5)
            .shift(7)
            .unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shift() {
        assert_eq!(
            RowMask::new(Mask::from_iter([true, true, true, true, true]), 5)
                .shift(5)
                .unwrap(),
            RowMask::new(Mask::from_iter([true, true, true, true, true]), 0)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter_array() {
        let mask = RowMask::new(
            Mask::from_iter([
                false, false, false, false, false, true, true, true, true, true,
            ]),
            0,
        );
        let array = Buffer::from_iter(0..20).into_array();
        let filtered = mask.filter_array(&array).unwrap().unwrap();
        assert_eq!(
            filtered.to_primitive().unwrap().as_slice::<i32>(),
            (5..10).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic]
    fn test_row_mask_type_validation() {
        let array = PrimitiveArray::new(buffer![1.0, 2.0], Validity::AllInvalid).into_array();
        RowMask::from_array(&array, 0, 2).unwrap();
    }
}
