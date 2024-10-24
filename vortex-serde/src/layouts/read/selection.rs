#![allow(dead_code)]
use std::cmp::{max, min};
use std::fmt::{Display, Formatter};

use arrow_buffer::{BooleanBuffer, MutableBuffer};
use croaring::Bitmap;
use vortex::array::BoolArray;
use vortex::compute::filter;
use vortex::validity::Validity;
use vortex::Array;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

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
                        bitmap.add_range((sb + begin) as u32..(se + begin) as u32);
                    }
                    RowSelector::new(bitmap, 0, end)
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

    pub fn intersect(mut self, other: &RowSelector) -> Self {
        self.values.and_inplace(&other.values);
        RowSelector::new(
            self.values,
            max(self.begin, other.begin),
            min(self.end, other.end),
        )
    }

    pub fn concatenate(mut self, other: &RowSelector) -> Self {
        assert_eq!(
            self.end, other.begin,
            "Can only concatenate consecutive selectors"
        );
        self.values.or_inplace(&other.values);
        RowSelector::new(
            self.values,
            min(self.begin, other.begin),
            max(self.end, other.end),
        )
    }

    pub fn filter_array(&self, array: impl AsRef<Array>) -> VortexResult<Option<Array>> {
        if self.begin != 0 {
            vortex_bail!("Cannot filter arrays with absolute row selections")
        }

        if self.values.cardinality() == 0 {
            return Ok(None);
        }

        let array = array.as_ref();

        if self.values.cardinality() == array.len() as u64 {
            return Ok(Some(array.clone()));
        }

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
        filter(array, predicate).map(Some)
    }

    pub fn offset(self, offset: i64) -> RowSelector {
        if offset == 0 {
            self
        } else {
            RowSelector::new(
                self.values.add_offset(-offset),
                if self.begin as i64 > offset {
                    (self.begin as i64 - offset) as usize
                } else {
                    0
                },
                (self.end as i64 - offset) as usize,
            )
        }
    }

    pub fn advance(mut self, up_to_row: usize) -> Option<RowSelector> {
        if up_to_row >= self.end {
            None
        } else {
            self.values.remove_range(0..up_to_row as u32);
            Some(RowSelector::new(self.values, self.begin, self.end))
        }
    }
}

#[cfg(test)]
mod tests {
    use croaring::Bitmap;
    use rstest::rstest;

    use crate::layouts::read::selection::RowSelector;

    #[rstest]
    #[case(RowSelector::new((0..2).chain(9..10).collect(),0, 10), RowSelector::new((0..1).collect(),0, 10), RowSelector::new((0..1).collect(),0, 10))]
    #[case(RowSelector::new((5..8).chain(9..10).collect(),0, 10), RowSelector::new((2..5).collect(),0, 10), RowSelector::new(Bitmap::new(),0, 10))]
    #[case(RowSelector::new((0..4).collect(),0, 10), RowSelector::new((0..1).chain(2..3).chain(3..5).collect(),0, 10), RowSelector::new((0..1).chain(2..4).collect(),0, 10))]
    #[case(RowSelector::new((0..3).chain(5..6).collect(),0, 10), RowSelector::new((2..6).collect(),0, 10), RowSelector::new((2..3).chain(5..6).collect(),0, 10))]
    #[cfg_attr(miri, ignore)]
    fn intersection(
        #[case] first: RowSelector,
        #[case] second: RowSelector,
        #[case] expected: RowSelector,
    ) {
        assert_eq!(first.intersect(&second), expected);
    }
}
