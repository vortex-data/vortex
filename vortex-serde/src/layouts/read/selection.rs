use std::cmp::{max, min};
use std::fmt::{Display, Formatter};

use arrow_buffer::{BooleanBuffer, MutableBuffer};
use croaring::Bitmap;
use vortex::array::BoolArray;
use vortex::compute::{filter, slice};
use vortex::validity::Validity;
use vortex::Array;
use vortex_error::{vortex_err, VortexResult};

/// Bitmap of selected row ranges
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowSelector {
    values: Bitmap,
    begin: usize,
    end: usize,
}

impl Display for RowSelector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RowSelector [{}..{}]", self.begin, self.end)
    }
}

impl RowSelector {
    pub fn new(values: Bitmap, begin: usize, end: usize) -> Self {
        Self { values, begin, end }
    }

    pub fn from_array(array: &Array, begin: usize, end: usize) -> VortexResult<Self> {
        array.with_dyn(|a| {
            a.as_bool_array()
                .ok_or_else(|| vortex_err!("Must be a bool array"))
                .map(|b| {
                    let mut bitmap = Bitmap::new();
                    for (sb, se) in b.maybe_null_slices_iter() {
                        bitmap.add_range(sb as u32..se as u32);
                    }
                    RowSelector::new(bitmap, begin, end)
                })
        })
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

    pub fn slice(&self, begin: usize, end: usize) -> Self {
        let range_begin = max(self.begin, begin);
        let range_end = min(self.end, end);
        let mask =
            Bitmap::from_range((range_begin - self.begin) as u32..(range_end - self.begin) as u32);
        RowSelector::new(
            self.values
                .and(&mask)
                .add_offset(-((range_begin - self.begin) as i64)),
            range_begin,
            range_end,
        )
    }

    pub fn filter_array(&self, array: impl AsRef<Array>) -> VortexResult<Option<Array>> {
        let true_count = self.values.cardinality();
        if true_count == 0 {
            return Ok(None);
        }

        let array = array.as_ref();

        if true_count == array.len() as u64 {
            return Ok(Some(array.clone()));
        }

        let sliced = if self.len() == array.len() {
            array
        } else {
            &slice(array, self.begin, self.end)?
        };

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
        let predicate = BoolArray::try_new(
            BooleanBuffer::new(buffer.into(), 0, self.len()),
            Validity::NonNullable,
        )?;
        filter(sliced, predicate).map(Some)
    }

    pub fn add_offset(mut self, offset: i64) -> RowSelector {
        if offset == 0 {
            self
        } else {
            let just_shift = self.begin as i64 >= offset;
            RowSelector::new(
                if just_shift {
                    self.values
                } else {
                    // Remove last N values that were trimmed by the offset. Since we know begin is 0 new len is end - offset
                    self.values
                        .remove_range((self.end as i64 - offset) as u32..self.len() as u32);
                    self.values.add_offset(-offset)
                },
                if just_shift {
                    (self.begin as i64 - offset) as usize
                } else {
                    0
                },
                (self.end as i64 - offset) as usize,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use croaring::Bitmap;
    use rstest::rstest;

    use crate::layouts::read::selection::RowSelector;

    #[rstest]
    #[case(RowSelector::new((0..2).chain(9..10).collect(), 0, 10), (0, 1), RowSelector::new((0..1).collect(), 0, 1))]
    #[case(RowSelector::new((5..8).chain(9..10).collect(), 0, 10), (2, 5), RowSelector::new(Bitmap::new(), 2, 5))]
    #[case(RowSelector::new((0..4).collect(), 0, 10), (2, 5), RowSelector::new((0..2).collect(), 2, 5))]
    #[case(RowSelector::new((0..3).chain(5..6).collect(), 0, 10), (2, 6), RowSelector::new((0..1).chain(3..4).collect(), 2, 6))]
    #[cfg_attr(miri, ignore)]
    fn slice(
        #[case] first: RowSelector,
        #[case] range: (usize, usize),
        #[case] expected: RowSelector,
    ) {
        assert_eq!(first.slice(range.0, range.1), expected);
    }
}
