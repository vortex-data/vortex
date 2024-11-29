use std::collections::{BTreeSet, VecDeque};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};

use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use super::PruningRead;
use crate::{BatchRead, LayoutReader, MessageLocator, RowMask};

pub enum SplitMask {
    ReadMore(Vec<MessageLocator>),
    Mask(RowMask),
}

/// Iterator over row ranges of a vortex file with bitmaps of valid values in those ranges
pub trait MaskIterator: Iterator<Item = VortexResult<SplitMask>> + Send {
    /// Register additional horizontal row boundaries to split the generated layout on
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()>;
}

enum PruningSplitState {
    Ranges(VecDeque<(usize, usize)>),
    Splits(BTreeSet<usize>),
}

pub struct PruningSplitIterator {
    reader: Box<dyn LayoutReader>,
    in_progress_masks: VecDeque<RowMask>,
    registered_splits: AtomicBool,
    splits: PruningSplitState,
    row_mask: Option<RowMask>,
}

impl PruningSplitIterator {
    pub fn new(reader: Box<dyn LayoutReader>, row_count: u64, row_mask: Option<RowMask>) -> Self {
        let mut splits = BTreeSet::new();
        splits.insert(row_count as usize);
        Self {
            reader,
            in_progress_masks: VecDeque::new(),
            registered_splits: AtomicBool::new(false),
            splits: PruningSplitState::Splits(splits),
            row_mask,
        }
    }

    /// Read given mask out of the reader
    fn read_mask(&mut self, mask: RowMask) -> VortexResult<Option<SplitMask>> {
        if let Some(rs) = self.reader.read_selection(&mask)? {
            return match rs {
                BatchRead::ReadMore(rm) => {
                    // If the reader needs more data we put the mask back into queue for to come back to it later
                    self.in_progress_masks.push_back(mask);
                    Ok(Some(SplitMask::ReadMore(rm)))
                }
                BatchRead::Batch(batch) => {
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
                    Ok(Some(SplitMask::Mask(mask.and_bitmask(batch)?)))
                }
            };
        }
        Ok(None)
    }

    /// Return next mask that contains at least one row or request to read more data.
    fn next_mask(&mut self) -> VortexResult<Option<SplitMask>> {
        // First consider masks we have previously started reading to return them in order
        while let Some(mask) = self.in_progress_masks.pop_front() {
            if let Some(read_mask) = self.read_mask(mask)? {
                return Ok(Some(read_mask));
            }
        }

        match &mut self.splits {
            PruningSplitState::Ranges(ranges) => {
                while let Some((begin, end)) = ranges.pop_front() {
                    // check if we can prune the entire range (because stats indicate that it's all false)
                    let can_prune = self.reader.can_prune(begin, end)?;

                    match can_prune {
                        PruningRead::ReadMore(messages) => {
                            ranges.push_front((begin, end));
                            return Ok(Some(SplitMask::ReadMore(messages)));
                        }
                        PruningRead::CanPrune(true) => continue,
                        PruningRead::CanPrune(false) => {}
                    };

                    // we couldn't prune the whole range, but we can still read/apply the row_mask
                    let Some(ref row_mask) = self.row_mask else {
                        return self.read_mask(RowMask::new_valid_between(begin, end));
                    };

                    // we masked everything out, so move on
                    let sliced = row_mask.slice(begin, end)?;
                    if sliced.is_empty() {
                        continue;
                    }

                    return self.read_mask(sliced);
                }

                Ok(None)
            }
            PruningSplitState::Splits(s) => {
                // FIXME(DK): is this a spinlock waiting for one thread to add_splits?
                if !self.registered_splits.swap(true, Ordering::SeqCst) {
                    self.reader.add_splits(0, s)?;
                    self.splits = PruningSplitState::Ranges(
                        mem::take(s)
                            .into_iter()
                            .tuple_windows::<(usize, usize)>()
                            .collect::<VecDeque<_>>(),
                    );
                }
                self.next_mask()
            }
        }
    }
}

impl MaskIterator for PruningSplitIterator {
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        match &mut self.splits {
            PruningSplitState::Ranges(_) => {
                vortex_bail!(
                    "Can't insert additional splits, we've already started producing row ranges"
                )
            }
            PruningSplitState::Splits(s) => {
                s.append(splits);
                Ok(())
            }
        }
    }
}

impl Iterator for PruningSplitIterator {
    type Item = VortexResult<SplitMask>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_mask().transpose()
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
}

impl MaskIterator for FixedSplitIterator {
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
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
    type Item = VortexResult<SplitMask>;

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
                        Some(Ok(SplitMask::Mask(sliced)))
                    } else {
                        Some(Ok(SplitMask::Mask(RowMask::new_valid_between(begin, end))))
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vortex_array::array::BoolArray;
    use vortex_array::IntoArrayData;
    use vortex_error::VortexResult;

    use crate::read::splits::{FixedSplitIterator, MaskIterator, SplitMask};
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
            mask_iter
                .map(|split| split.map(|mask| match mask {
                    SplitMask::ReadMore(_) => unreachable!("Will never read more"),
                    SplitMask::Mask(m) => m,
                }))
                .collect::<VortexResult<Vec<_>>>()
                .unwrap(),
            vec![RowMask::new_valid_between(4, 6)]
        );
    }
}
