use std::collections::BTreeSet;
use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::read::buffered::RowMaskReader;
use crate::{BatchRead, LayoutReader, MessageRead, PruningRead, RowMask, SplitRead};

pub struct RowMaskLayoutReader {
    layout: Box<dyn LayoutReader>,
}

impl RowMaskLayoutReader {
    pub fn new(layout: Box<dyn LayoutReader>) -> Self {
        Self { layout }
    }
}

impl RowMaskReader<RowMask> for RowMaskLayoutReader {
    /// Read given mask out of the reader
    fn read_mask(&self, mask: &RowMask) -> VortexResult<Option<MessageRead<RowMask>>> {
        let can_prune = self.layout.can_prune(mask.begin(), mask.end())?;

        match can_prune {
            PruningRead::ReadMore(messages) => {
                return Ok(Some(SplitRead::ReadMore(messages)));
            }
            PruningRead::Value(true) => return Ok(None),
            PruningRead::Value(false) => {}
        };

        if let Some(rs) = self.layout.read_selection(mask)? {
            return match rs {
                BatchRead::ReadMore(messages) => Ok(Some(SplitRead::ReadMore(messages))),
                BatchRead::Value(batch) => {
                    // If the mask is all FALSE we can safely discard it
                    if batch
                        .statistics()
                        .compute_true_count()
                        .vortex_expect("must be a bool array if it's a result of a filter")
                        == 0
                    {
                        return Ok(None);
                    }
                    // Combine requested mask with the result of filter read
                    Ok(Some(SplitRead::Value(mask.and_bitmask(batch)?)))
                }
            };
        }
        Ok(None)
    }
}

enum FixedSplitState {
    Ranges(Box<dyn Iterator<Item = (usize, usize)> + Send>),
    Splits(BTreeSet<usize>),
}

pub struct FixedSplitIterator {
    splits: FixedSplitState,
    row_mask: Option<RowMask>,
}

impl FixedSplitIterator {
    pub fn new(row_count: u64, row_mask: Option<RowMask>) -> Self {
        let mut splits = BTreeSet::new();
        splits.insert(row_count as usize);
        Self {
            splits: FixedSplitState::Splits(splits),
            row_mask,
        }
    }

    pub fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        match &mut self.splits {
            FixedSplitState::Ranges(_) => {
                vortex_bail!("Can't insert additional splits if we started producing row ranges")
            }
            FixedSplitState::Splits(s) => {
                s.append(splits);
                Ok(())
            }
        }
    }
}

impl Iterator for FixedSplitIterator {
    type Item = VortexResult<RowMask>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.splits {
            FixedSplitState::Ranges(ranges) => {
                // Find next range that's not filtered out by supplied row_mask
                for (begin, end) in ranges {
                    return if let Some(ref row_mask) = self.row_mask {
                        let sliced = match row_mask.slice(begin, end) {
                            Ok(s) => s,
                            Err(e) => return Some(Err(e)),
                        };

                        if sliced.is_empty() {
                            continue;
                        }
                        Some(Ok(sliced))
                    } else {
                        Some(Ok(RowMask::new_valid_between(begin, end)))
                    };
                }
                None
            }
            FixedSplitState::Splits(s) => {
                self.splits = FixedSplitState::Ranges(Box::new(
                    mem::take(s).into_iter().tuple_windows::<(usize, usize)>(),
                ));
                self.next()
            }
        }
    }
}

impl Stream for FixedSplitIterator {
    type Item = VortexResult<RowMask>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.next())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vortex_array::array::BoolArray;
    use vortex_array::IntoArrayData;
    use vortex_error::VortexResult;

    use crate::read::splits::FixedSplitIterator;
    use crate::RowMask;

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn register_after_start() {
        let mut mask_iter = FixedSplitIterator::new(10, None);
        mask_iter
            .additional_splits(&mut BTreeSet::from([0, 1, 2]))
            .unwrap();
        assert!(mask_iter.next().is_some());
        mask_iter
            .additional_splits(&mut BTreeSet::from([5]))
            .unwrap();
        mask_iter.next();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn filters_empty() {
        let mut mask_iter = FixedSplitIterator::new(
            10,
            Some(
                RowMask::try_new(
                    BoolArray::from_iter([
                        false, false, false, false, true, true, false, false, false, false,
                    ])
                    .into_array(),
                    0,
                    10,
                )
                .unwrap(),
            ),
        );
        mask_iter
            .additional_splits(&mut BTreeSet::from([0, 2, 4, 6, 8, 10]))
            .unwrap();
        assert_eq!(
            mask_iter.collect::<VortexResult<Vec<_>>>().unwrap(),
            vec![RowMask::new_valid_between(4, 6)]
        );
    }
}
