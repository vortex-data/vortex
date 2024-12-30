use std::sync::Arc;

use vortex_array::stats::ArrayStatistics;
use vortex_error::VortexResult;

use crate::read::buffered::ReadMasked;
use crate::{LayoutReader, PollRead, Prune, RowMask};

/// Reads an array out of a [`LayoutReader`] as a [`RowMask`].
///
/// Similar to `ReadArray`, this wraps a layout to read an array, but `ReadRowMask` will interpret
/// that array as a `RowMask`, and performs some optimizations to apply pruning first.
pub(crate) struct ReadRowMask {
    layout: Arc<dyn LayoutReader>,
}

impl ReadRowMask {
    pub(crate) fn new(layout: Arc<dyn LayoutReader>) -> Self {
        Self { layout }
    }
}

impl ReadMasked for ReadRowMask {
    type Value = RowMask;

    /// Read given mask out of the reader
    fn read_masked(&self, mask: &RowMask) -> VortexResult<Option<PollRead<RowMask>>> {
        let can_prune = self.layout.poll_prune(mask.begin(), mask.end())?;

        match can_prune {
            PollRead::ReadMore(messages) => {
                return Ok(Some(PollRead::ReadMore(messages)));
            }
            PollRead::Value(Prune::CanPrune) => return Ok(None),
            PollRead::Value(Prune::CannotPrune) => {}
        };

        if let Some(rs) = self.layout.poll_read(mask)? {
            return match rs {
                PollRead::ReadMore(messages) => Ok(Some(PollRead::ReadMore(messages))),
                PollRead::Value(batch) => {
                    // If the mask is all FALSE we can safely discard it
                    if batch
                        .statistics()
                        .compute_true_count()
                        .map(|true_count| true_count == 0)
                        .unwrap_or(false)
                    {
                        return Ok(None);
                    }
                    // Combine requested mask with the result of filter read
                    Ok(Some(PollRead::Value(mask.and_bitmask(batch)?)))
                }
            };
        }
        Ok(None)
    }
}

/// Takes all row sub-ranges, and produces a set of row ranges that aren't filtered out by the provided mask.
pub struct SplitsAccumulator {
    ranges: Box<dyn Iterator<Item = (usize, usize)> + Send>,
    row_mask: Option<RowMask>,
}

pub struct SplitsIntoIter {
    ranges: Box<dyn Iterator<Item = (usize, usize)> + Send>,
    row_mask: Option<RowMask>,
}

impl SplitsAccumulator {
    pub fn new(
        ranges: impl Iterator<Item = (usize, usize)> + Send + 'static,
        row_mask: Option<RowMask>,
    ) -> Self {
        let ranges = Box::new(ranges);
        Self { ranges, row_mask }
    }
}

impl IntoIterator for SplitsAccumulator {
    type Item = VortexResult<RowMask>;

    type IntoIter = SplitsIntoIter;

    /// Creates an iterator of row masks to be read from the underlying file.
    fn into_iter(self) -> Self::IntoIter {
        SplitsIntoIter {
            ranges: self.ranges,
            row_mask: self.row_mask,
        }
    }
}

impl Iterator for SplitsIntoIter {
    type Item = VortexResult<RowMask>;

    fn next(&mut self) -> Option<Self::Item> {
        // Find next range that's not filtered out by supplied row_mask
        for (begin, end) in self.ranges.as_mut() {
            return if let Some(ref row_mask) = self.row_mask {
                let sliced = match row_mask.slice(begin, end) {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e)),
                };

                // If the range is all false, we take the next one
                if sliced.is_all_false() {
                    continue;
                }
                Some(Ok(sliced))
            } else {
                Some(Ok(RowMask::new_valid_between(begin, end)))
            };
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::FilterMask;
    use vortex_error::VortexResult;

    use crate::read::splits::SplitsAccumulator;
    use crate::RowMask;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filters_empty() {
        let mask_iter = SplitsAccumulator::new(
            [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)].into_iter(),
            Some(
                RowMask::try_new(
                    FilterMask::from_iter([
                        false, false, false, false, true, true, false, false, false, false,
                    ]),
                    0,
                    10,
                )
                .unwrap(),
            ),
        );

        let actual = mask_iter
            .into_iter()
            .collect::<VortexResult<Vec<_>>>()
            .unwrap();
        let expected = vec![RowMask::new_valid_between(4, 6)];

        assert_eq!(actual, expected);
    }
}
