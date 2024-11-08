use std::collections::{BTreeSet, VecDeque};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};

use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::file::{BatchRead, LayoutReader, Message, RowMask};

pub enum SplitMask {
    ReadMore(Vec<Message>),
    Mask(RowMask),
}

enum SplitState {
    Ranges(Box<dyn Iterator<Item = (usize, usize)> + Send>),
    Splits(BTreeSet<usize>),
}

pub trait MaskIterator: Iterator<Item = VortexResult<SplitMask>> + Send {
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()>;
}

pub struct FilteringRowSplitIterator {
    reader: Box<dyn LayoutReader>,
    static_splits: FixedSplitIterator,
    in_progress_masks: VecDeque<RowMask>,
    registered_splits: AtomicBool,
}

impl FilteringRowSplitIterator {
    pub fn new(reader: Box<dyn LayoutReader>, row_count: u64, row_mask: Option<RowMask>) -> Self {
        let static_splits = FixedSplitIterator::new(row_count, row_mask);
        Self {
            reader,
            static_splits,
            in_progress_masks: VecDeque::new(),
            registered_splits: AtomicBool::new(false),
        }
    }

    fn read_mask(&mut self, mask: RowMask) -> VortexResult<Option<SplitMask>> {
        if let Some(rs) = self.reader.read_selection(&mask)? {
            return match rs {
                BatchRead::ReadMore(rm) => {
                    self.in_progress_masks.push_back(mask);
                    Ok(Some(SplitMask::ReadMore(rm)))
                }
                BatchRead::Batch(batch) => {
                    if batch
                        .statistics()
                        .compute_true_count()
                        .vortex_expect("must be a bool array if it's a result of a filter")
                        == 0
                    {
                        return Ok(None);
                    }
                    Ok(Some(SplitMask::Mask(mask.with_values(batch)?)))
                }
            };
        }
        Ok(None)
    }

    fn next_mask(&mut self) -> VortexResult<Option<SplitMask>> {
        if !self.registered_splits.swap(true, Ordering::SeqCst) {
            let mut own_splits = BTreeSet::new();
            self.reader.add_splits(0, &mut own_splits)?;
            self.static_splits.additional_splits(&mut own_splits)?;
        }

        while let Some(mask) = self.in_progress_masks.pop_front() {
            if let Some(read_mask) = self.read_mask(mask)? {
                return Ok(Some(read_mask));
            }
        }

        while let Some(mask) = self.static_splits.next() {
            match mask? {
                SplitMask::ReadMore(_) => {
                    unreachable!("StaticSplitProducer never returns ReadMore")
                }
                SplitMask::Mask(m) => {
                    if let Some(read_mask) = self.read_mask(m)? {
                        return Ok(Some(read_mask));
                    }
                }
            }
        }
        Ok(None)
    }
}

impl MaskIterator for FilteringRowSplitIterator {
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        self.static_splits.additional_splits(splits)
    }
}

impl Iterator for FilteringRowSplitIterator {
    type Item = VortexResult<SplitMask>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_mask().transpose()
    }
}

pub struct FixedSplitIterator {
    splits: SplitState,
    row_mask: Option<RowMask>,
}

impl FixedSplitIterator {
    pub fn new(row_count: u64, row_mask: Option<RowMask>) -> Self {
        let mut splits = BTreeSet::new();
        splits.insert(row_count as usize);
        Self {
            splits: SplitState::Splits(splits),
            row_mask,
        }
    }
}

impl MaskIterator for FixedSplitIterator {
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        match &mut self.splits {
            SplitState::Ranges(_) => {
                vortex_bail!("Can't insert additional splits if we started producing row ranges")
            }
            SplitState::Splits(s) => {
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
            SplitState::Ranges(ranges) => {
                for (begin, end) in ranges {
                    return if let Some(ref row_mask) = self.row_mask {
                        if row_mask.slice(begin, end).is_empty() {
                            continue;
                        }
                        Some(Ok(SplitMask::Mask(row_mask.slice(begin, end))))
                    } else {
                        Some(Ok(SplitMask::Mask(RowMask::new_valid_between(begin, end))))
                    };
                }
                None
            }
            SplitState::Splits(s) => {
                self.splits = SplitState::Ranges(Box::new(
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

    use vortex_error::VortexResult;

    use crate::file::read::splits::{FixedSplitIterator, MaskIterator, SplitMask};
    use crate::file::RowMask;

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
        let mut mask_iter =
            FixedSplitIterator::new(10, Some(RowMask::try_new((4..6).collect(), 0, 10).unwrap()));
        mask_iter
            .additional_splits(&mut BTreeSet::from([2, 4, 6, 8, 10]))
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
