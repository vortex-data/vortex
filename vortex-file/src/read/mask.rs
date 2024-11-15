use std::cmp::{max, min};
use std::fmt::{Display, Formatter};

use arrow_buffer::{BooleanBuffer, MutableBuffer};
use croaring::Bitmap;
use vortex_array::array::{BoolArray, PrimitiveArray, SparseArray};
use vortex_array::compute::{filter, slice, take};
use vortex_array::validity::{LogicalValidity, Validity};
use vortex_array::{iterate_integer_array, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::PType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

const PREFER_TAKE_TO_FILTER_DENSITY: f64 = 1.0 / 1024.0;

/// Bitmap of selected rows within given [begin, end) row range
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowMask {
    values: Bitmap,
    begin: usize,
    end: usize,
}

impl Display for RowMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowSelector [{}..{}]", self.begin, self.end)
    }
}

impl RowMask {
    pub fn try_new(values: Bitmap, begin: usize, end: usize) -> VortexResult<Self> {
        if values
            .maximum()
            .map(|m| m > (end - begin) as u32)
            .unwrap_or(false)
        {
            vortex_bail!("Values bitmap must be in 0..(end-begin) range")
        }
        Ok(Self { values, begin, end })
    }

    /// Construct a RowMask which is valid in the given range.
    pub fn new_valid_between(begin: usize, end: usize) -> Self {
        unsafe { RowMask::new_unchecked(Bitmap::from_range(0..(end - begin) as u32), begin, end) }
    }

    /// Construct a RowMask which is invalid everywhere in the given range.
    pub fn new_invalid_between(begin: usize, end: usize) -> Self {
        unsafe { RowMask::new_unchecked(Bitmap::new(), begin, end) }
    }

    /// Construct a RowMask from given bitmap and begin.
    ///
    /// # Safety
    ///
    /// The maximum set index of the `values` must be no greater than `end - begin`.
    pub unsafe fn new_unchecked(values: Bitmap, begin: usize, end: usize) -> Self {
        Self { values, begin, end }
    }

    /// Construct a RowMask from a Boolean typed array.
    ///
    /// True-valued positions are kept by the returned mask.
    pub fn from_mask_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        match array.with_dyn(|a| a.logical_validity()) {
            LogicalValidity::AllValid(_) => {
                Self::from_mask_array_ignoring_validity(array, begin, end)
            }
            LogicalValidity::AllInvalid(_) => Ok(Self::new_invalid_between(begin, end)),
            LogicalValidity::Array(validity) => {
                let mut bits = Self::from_mask_array_ignoring_validity(array, begin, end)?;
                let validity = Self::from_mask_array_ignoring_validity(&validity, begin, end)?;
                bits.and_inplace(&validity)?;
                Ok(bits)
            }
        }
    }

    fn from_mask_array_ignoring_validity(
        array: &ArrayData,
        begin: usize,
        end: usize,
    ) -> VortexResult<Self> {
        array.with_dyn(|a| {
            a.as_bool_array()
                .ok_or_else(|| vortex_err!("Must be a bool array"))
                .map(|b| {
                    let mut bitmap = Bitmap::new();
                    for (sb, se) in b.maybe_null_slices_iter() {
                        bitmap.add_range(sb as u32..se as u32);
                    }
                    unsafe { RowMask::new_unchecked(bitmap, begin, end) }
                })
        })
    }

    /// Construct a RowMask from an integral array.
    ///
    /// The array values are interpreted as indices and those indices are kept by the returned mask.
    pub fn from_index_array(array: &ArrayData, begin: usize, end: usize) -> VortexResult<Self> {
        array.with_dyn(|a| {
            let err = || vortex_err!(InvalidArgument: "index array must be integers in the range [0, 2^32)");
            let array = a.as_primitive_array().ok_or_else(err)?;

            if !array.ptype().is_int() {
                return Err(err());
            }

            let mut bitmap = Bitmap::new();

            iterate_integer_array!(array, |$P, $iterator| {
                for batch in $iterator {
                    for index in batch.data() {
                        bitmap.add(u32::try_from(*index).map_err(|_| err())?);
                    }
                }
            });

            Ok(unsafe { RowMask::new_unchecked(bitmap, begin, end) })
        })
    }

    /// Combine the RowMask with bitmask values resulting in new RowMask containing only values true in the bitmask
    pub fn and_bitmask(self, bitmask: ArrayData) -> VortexResult<Self> {
        // If we are a dense all true bitmap just take the bitmask array
        if self.len() as u64 == self.values.cardinality() {
            if bitmask.len() != self.len() {
                vortex_bail!(
                    "Bitmask length {} does not match our length {}",
                    bitmask.len(),
                    self.values.cardinality()
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
        self.values.is_empty()
    }

    pub fn begin(&self) -> usize {
        self.begin
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn len(&self) -> usize {
        self.end - self.begin
    }

    /// Limit mask to [begin..end) range
    pub fn slice(&self, begin: usize, end: usize) -> Self {
        let range_begin = max(self.begin, begin);
        let range_end = min(self.end, end);
        let mask =
            Bitmap::from_range((range_begin - self.begin) as u32..(range_end - self.begin) as u32);
        unsafe {
            RowMask::new_unchecked(
                self.values
                    .and(&mask)
                    .add_offset(-((range_begin - self.begin) as i64)),
                range_begin,
                range_end,
            )
        }
    }

    /// Unset, in place, any bits that are unset in `other`.
    pub fn and_inplace(&mut self, other: &RowMask) -> VortexResult<()> {
        if self.begin != other.begin || self.end != other.end {
            vortex_bail!(
                "begin and ends must match: {}-{} {}-{}",
                self.begin,
                self.end,
                other.begin,
                other.end
            );
        }
        self.values.and_inplace(&other.values);
        Ok(())
    }

    /// Filter array with this `RowMask`.
    ///
    /// This function assumes that Array is no longer than the mask length and that the mask starts on same offset as the array,
    /// i.e. the beginning of the array corresponds to the beginning of the mask with begin = 0
    pub fn filter_array(&self, array: impl AsRef<ArrayData>) -> VortexResult<Option<ArrayData>> {
        let true_count = self.values.cardinality();
        if true_count == 0 {
            return Ok(None);
        }

        let array = array.as_ref();

        let sliced = if self.len() == array.len() {
            array
        } else {
            &slice(array, self.begin, self.end)?
        };

        if true_count == sliced.len() as u64 {
            return Ok(Some(sliced.clone()));
        }

        if (true_count as f64 / sliced.len() as f64) < PREFER_TAKE_TO_FILTER_DENSITY {
            let indices = self.to_indices_array()?;
            take(sliced, indices).map(Some)
        } else {
            let mask = self.to_mask_array()?;
            filter(sliced, mask).map(Some)
        }
    }

    pub fn to_indices_array(&self) -> VortexResult<ArrayData> {
        Ok(PrimitiveArray::from_vec(self.values.to_vec(), Validity::NonNullable).into_array())
    }

    pub fn to_mask_array(&self) -> VortexResult<ArrayData> {
        let bitset = self
            .values
            .to_bitset()
            .ok_or_else(|| vortex_err!("Couldn't create bitset for RowSelection"))?;

        let byte_length = self.len().div_ceil(8);
        let mut buffer = MutableBuffer::with_capacity(byte_length);
        buffer.extend_from_slice(bitset.as_slice());
        if byte_length > bitset.size_in_bytes() {
            buffer.extend_zeros(byte_length - bitset.size_in_bytes());
        }
        BoolArray::try_new(
            BooleanBuffer::new(buffer.into(), 0, self.len()),
            Validity::NonNullable,
        )
        .map(IntoArrayData::into_array)
    }

    pub fn shift(self, offset: usize) -> VortexResult<RowMask> {
        let valid_shift = self.begin >= offset;
        if !valid_shift {
            vortex_bail!(
                "Can shift RowMask by at most {}, tried to shift by {offset}",
                self.begin
            )
        }
        Ok(unsafe { RowMask::new_unchecked(self.values, self.begin - offset, self.end - offset) })
    }
}

#[cfg(test)]
mod tests {
    use croaring::Bitmap;
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::{IntoArrayData, IntoArrayVariant};

    use crate::read::mask::RowMask;

    #[rstest]
    #[case(
        RowMask::try_new((0..2).chain(9..10).collect(), 0, 10).unwrap(), (0, 1),
        RowMask::try_new((0..1).collect(), 0, 1).unwrap())]
    #[case(
        RowMask::try_new((5..8).chain(9..10).collect(), 0, 10).unwrap(), (2, 5),
        RowMask::try_new(Bitmap::new(), 2, 5).unwrap())]
    #[case(
        RowMask::try_new((0..4).collect(), 0, 10).unwrap(), (2, 5),
        RowMask::try_new((0..2).collect(), 2, 5).unwrap())]
    #[case(
        RowMask::try_new((0..3).chain(5..6).collect(), 0, 10).unwrap(), (2, 6),
        RowMask::try_new((0..1).chain(3..4).collect(), 2, 6).unwrap())]
    #[case(
        RowMask::try_new((5..10).collect(), 0, 10).unwrap(), (7, 11),
        RowMask::try_new((0..3).collect(), 7, 10).unwrap())]
    #[case(
        RowMask::try_new((1..6).collect(), 3, 9).unwrap(), (0, 5),
        RowMask::try_new((1..2).collect(), 3, 5).unwrap())]
    #[cfg_attr(miri, ignore)]
    fn slice(#[case] first: RowMask, #[case] range: (usize, usize), #[case] expected: RowMask) {
        assert_eq!(first.slice(range.0, range.1), expected);
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn test_new() {
        RowMask::try_new((5..10).collect(), 5, 10).unwrap();
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn shift_invalid() {
        RowMask::try_new((0..5).collect(), 5, 10)
            .unwrap()
            .shift(7)
            .unwrap();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shift() {
        assert_eq!(
            RowMask::try_new((0..5).collect(), 5, 10)
                .unwrap()
                .shift(5)
                .unwrap(),
            RowMask::try_new((0..5).collect(), 0, 5).unwrap()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filter_array() {
        let mask = RowMask::try_new((5..10).collect(), 0, 10).unwrap();
        let array = PrimitiveArray::from((0..20).collect::<Vec<_>>()).into_array();
        let filtered = mask.filter_array(array).unwrap().unwrap();
        assert_eq!(
            filtered.into_primitive().unwrap().maybe_null_slice::<i32>(),
            (5..10).collect::<Vec<_>>()
        );
    }
}
