use std::collections::{BTreeSet, VecDeque};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};

use itertools::Itertools;
use vortex_array::stats::ArrayStatistics;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::{BatchRead, LayoutReader, MessageLocator, RowMask};

pub enum SplitMask {
    ReadMore(Vec<MessageLocator>),
    Mask(RowMask),
}

enum SplitState {
    Ranges(Box<dyn Iterator<Item = (usize, usize)> + Send>),
    Splits(BTreeSet<usize>),
}

/// Iterator over row ranges of a vortex file with bitmaps of valid values in those ranges
pub trait MaskIterator: Iterator<Item = VortexResult<SplitMask>> + Send {
    /// Register additional horizontal row boundaries to split the generated layout on
    fn additional_splits(&mut self, splits: &mut BTreeSet<usize>) -> VortexResult<()>;
}

/// MaskIterator that reads boolean arrays out of provided reader and further filters generated masks
///
/// Arrays returned by the reader must be of boolean dtype.
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

    /// Return next not all false mask or request to read more data.
    fn next_mask(&mut self) -> VortexResult<Option<SplitMask>> {
        if !self.registered_splits.swap(true, Ordering::SeqCst) {
            let mut own_splits = BTreeSet::new();
            self.reader.add_splits(0, &mut own_splits)?;
            self.static_splits.additional_splits(&mut own_splits)?;
        }

        // First consider masks we have previously started reading to return them in order
        while let Some(mask) = self.in_progress_masks.pop_front() {
            if let Some(read_mask) = self.read_mask(mask)? {
                return Ok(Some(read_mask));
            }
        }

        // Lastly take next statically generated mask and perform read with it on our reader
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
                // Find next range that's not filtered out by supplied row_mask
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
