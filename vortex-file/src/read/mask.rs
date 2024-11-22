use std::cmp::{max, min};
use std::fmt::{Display, Formatter};

use arrow_buffer::BooleanBuffer;
use vortex_array::array::{BoolArray, PrimitiveArray, SparseArray};
use vortex_array::compute::unary::try_cast;
use vortex_array::compute::{and, filter, slice, take, FilterMask, TakeOptions};
use vortex_array::stats::ArrayStatistics;
use vortex_array::validity::LogicalValidity;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

const PREFER_TAKE_TO_FILTER_DENSITY: f64 = 1.0 / 1024.0;

/// Bitmap of selected rows within given [begin, end) row range
#[derive(Debug, Clone)]
pub struct RowMask {
    bitmask: ArrayData,
    begin: usize,
    end: usize,
}

#[cfg(test)]
impl PartialEq for RowMask {
    fn eq(&self, other: &Self) -> bool {
        use vortex_error::VortexUnwrap;
        self.begin == other.begin
            && self.end == other.end
            && self
                .bitmask
                .clone()
                .into_bool()
                .vortex_unwrap()
                .boolean_buffer()
                == other
                    .bitmask
                    .clone()
                    .into_bool()
                    .vortex_unwrap()
                    .boolean_buffer()
    }
}

impl Display for RowMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowSelector [{}..{}]", self.begin, self.end)
    }
}

impl RowMask {
    pub fn try_new(bitmask: ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        if bitmask.dtype() != &DType::Bool(NonNullable) {
            vortex_bail!(
                "bitmask must be a nonnullable bool array {}",
                bitmask.dtype()
            )
        }
        if bitmask.len() != (end - begin) {
            vortex_bail!(
                "Bitmask must be the same length {} as the given range {}..{}",
                bitmask.len(),
                begin,
                end
            );
        }
        Ok(Self {
            bitmask,
            begin,
            end,
        })
    }

    /// Construct a RowMask which is valid in the given range.
    pub fn new_valid_between(begin: usize, end: usize) -> Self {
        unsafe {
            RowMask::new_unchecked(
                BoolArray::from(BooleanBuffer::new_set(end - begin)).into_array(),
                begin,
                end,
            )
        }
    }

    /// Construct a RowMask which is invalid everywhere in the given range.
    pub fn new_invalid_between(begin: usize, end: usize) -> Self {
        unsafe {
            RowMask::new_unchecked(
                BoolArray::from(BooleanBuffer::new_unset(end - begin)).into_array(),
                begin,
                end,
            )
        }
    }

    /// Construct a RowMask from given bitmask, begin and end.
    ///
    /// # Safety
    ///
    /// The bitmask must be of a nonnullable bool array and length of end - begin
    pub unsafe fn new_unchecked(bitmask: ArrayData, begin: usize, end: usize) -> Self {
        Self {
            bitmask,
            begin,
            end,
        }
    }

    /// Construct a RowMask from a Boolean typed array.
    ///
    /// True-valued positions are kept by the returned mask.
    pub fn from_mask_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        match array.with_dyn(|a| a.logical_validity()) {
            LogicalValidity::AllValid(_) => Self::try_new(array.clone(), begin, end),
            LogicalValidity::AllInvalid(_) => Ok(Self::new_invalid_between(begin, end)),
            LogicalValidity::Array(validity) => {
                let bitmask = and(array.clone(), validity)?;
                Self::try_new(bitmask, begin, end)
            }
        }
    }

    /// Construct a RowMask from an integral array.
    ///
    /// The array values are interpreted as indices and those indices are kept by the returned mask.
    pub fn from_index_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        let indices =
            try_cast(array, &DType::Primitive(PType::U64, NonNullable))?.into_primitive()?;
        let bools = BoolArray::from_indices(
            end - begin,
            indices
                .maybe_null_slice::<u64>()
                .iter()
                .map(|&i| i as usize),
        );
        RowMask::try_new(bools.into_array(), begin, end)
    }

    /// Combine the RowMask with bitmask values resulting in new RowMask containing only values true in the bitmask
    pub fn and_bitmask(self, bitmask: ArrayData) -> VortexResult<Self> {
        // If we are a dense all true bitmap just take the bitmask array
        if self.len()
            == self
                .bitmask
                .statistics()
                .compute_true_count()
                .vortex_expect("Must have a true count")
        {
            if bitmask.len() != self.len() {
                vortex_bail!(
                    "Bitmask length {} does not match our length {}",
                    bitmask.len(),
                    self.bitmask.len()
                );
            }
            Self::from_mask_array(&bitmask, self.begin, self.end)
        } else {
            // TODO(robert): Avoid densifying sparse values just to get true indices
            let sparse_mask =
                SparseArray::try_new(self.to_indices_array()?, bitmask, self.len(), false.into())?
                    .into_array()
                    .into_bool()?;
            Self::from_mask_array(sparse_mask.as_ref(), self.begin(), self.end())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bitmask
            .statistics()
            .compute_true_count()
            .vortex_expect("Must have true count")
            == 0
    }

    pub fn begin(&self) -> usize {
        self.begin
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn len(&self) -> usize {
        self.bitmask.len()
    }

    /// Limit mask to [begin..end) range
    pub fn slice(&self, begin: usize, end: usize) -> Self {
        let range_begin = max(self.begin, begin);
        let range_end = min(self.end, end);
        unsafe {
            RowMask::new_unchecked(
                if range_begin == self.begin && range_end == self.end {
                    self.bitmask.clone()
                } else {
                    slice(
                        &self.bitmask,
                        range_begin - self.begin,
                        range_end - self.begin,
                    )
                    .vortex_expect("Must be a valid slice")
                },
                range_begin,
                range_end,
            )
        }
    }

    /// Filter array with this `RowMask`.
    ///
    /// This function assumes that Array is no longer than the mask length and that the mask starts on same offset as the array,
    /// i.e. the beginning of the array corresponds to the beginning of the mask with begin = 0
    pub fn filter_array(&self, array: impl AsRef<ArrayData>) -> VortexResult<Option<ArrayData>> {
        let true_count = self
            .bitmask
            .statistics()
            .compute_true_count()
            .vortex_expect("Must have a true count");
        if true_count == 0 {
            return Ok(None);
        }

        let array = array.as_ref();

        let sliced = if self.len() == array.len() {
            array
        } else {
            &slice(array, self.begin, self.end)?
        };

        if true_count == sliced.len() {
            return Ok(Some(sliced.clone()));
        }

        if (true_count as f64 / sliced.len() as f64) < PREFER_TAKE_TO_FILTER_DENSITY {
            let indices = self.to_indices_array()?;
            take(sliced, indices, TakeOptions::default()).map(Some)
        } else {
            let mask = FilterMask::try_from(self.bitmask.clone())?;
            filter(sliced, mask).map(Some)
        }
    }

    pub fn to_indices_array(&self) -> VortexResult<ArrayData> {
        Ok(PrimitiveArray::from(
            self.bitmask
                .clone()
                .into_bool()?
                .boolean_buffer()
                .set_indices()
                .map(|i| i as u64)
                .collect::<Vec<_>>(),
        )
        .into_array())
    }

    pub fn shift(self, offset: usize) -> VortexResult<RowMask> {
        let valid_shift = self.begin >= offset;
        if !valid_shift {
            vortex_bail!(
                "Can shift RowMask by at most {}, tried to shift by {offset}",
                self.begin
            )
        }
        Ok(unsafe { RowMask::new_unchecked(self.bitmask, self.begin - offset, self.end - offset) })
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rstest::rstest;
    use vortex_array::array::{BoolArray, PrimitiveArray};
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_dtype::Nullability;

    use crate::read::mask::RowMask;

    #[rstest]
    #[case(
        RowMask::try_new(BoolArray::from_iter([true, true, true, false, false, false, false, false, true, true]).into_array(), 0, 10).unwrap(), (0, 1),
        RowMask::try_new(BoolArray::from_iter([true]).into_array(), 0, 1).unwrap())]
    #[case(
        RowMask::try_new(BoolArray::from_iter([false, false, false, false, false, true, true, true, true, true]).into_array(), 0, 10).unwrap(), (2, 5),
        RowMask::try_new(BoolArray::from_iter([false, false, false]).into_array(), 2, 5).unwrap()
    )]
    #[case(
        RowMask::try_new(BoolArray::from_iter([true, true, true, true, false, false, false, false, false, false]).into_array(), 0, 10).unwrap(), (2, 5),
        RowMask::try_new(BoolArray::from_iter([true, true, false]).into_array(), 2, 5).unwrap()
    )]
    #[case(
        RowMask::try_new(BoolArray::from_iter([true, true, true, false, false, true, true, false, false, false]).into_array(), 0, 10).unwrap(), (2, 6),
        RowMask::try_new(BoolArray::from_iter([true, false, false, true]).into_array(), 2, 6).unwrap())]
    #[case(
        RowMask::try_new(BoolArray::from_iter([false, false, false, false, false, true, true, true, true, true]).into_array(), 0, 10).unwrap(), (7, 11),
        RowMask::try_new(BoolArray::from_iter([true, true, true]).into_array(), 7, 10).unwrap())]
    #[case(
        RowMask::try_new(BoolArray::from_iter([false, true, true, true, true, true]).into_array(), 3, 9).unwrap(), (0, 5),
        RowMask::try_new(BoolArray::from_iter([false, true]).into_array(), 3, 5).unwrap())]
    #[cfg_attr(miri, ignore)]
    fn slice(#[case] first: RowMask, #[case] range: (usize, usize), #[case] expected: RowMask) {
        assert_eq!(first.slice(range.0, range.1), expected);
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn test_new() {
        RowMask::try_new(
            BoolArray::new(BooleanBuffer::new_unset(10), Nullability::NonNullable).into_array(),
            5,
            10,
        )
        .unwrap();
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn shift_invalid() {
        RowMask::try_new(
            BoolArray::from_iter([true, true, true, true, true]).into_array(),
            5,
            10,
        )
        .unwrap()
        .shift(7)
        .unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shift() {
        assert_eq!(
            RowMask::try_new(
                BoolArray::from_iter([true, true, true, true, true]).into_array(),
                5,
                10
            )
            .unwrap()
            .shift(5)
            .unwrap(),
            RowMask::try_new(
                BoolArray::from_iter([true, true, true, true, true]).into_array(),
                0,
                5
            )
            .unwrap()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter_array() {
        let mask = RowMask::try_new(
            BoolArray::from_iter([
                false, false, false, false, false, true, true, true, true, true,
            ])
            .into_array(),
            0,
            10,
        )
        .unwrap();
        let array = PrimitiveArray::from((0..20).collect::<Vec<_>>()).into_array();
        let filtered = mask.filter_array(array).unwrap().unwrap();
        assert_eq!(
            filtered.into_primitive().unwrap().maybe_null_slice::<i32>(),
            (5..10).collect::<Vec<_>>()
        );
    }
}
