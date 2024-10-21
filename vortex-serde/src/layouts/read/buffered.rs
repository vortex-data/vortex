use std::collections::VecDeque;
use std::mem;

use croaring::Bitmap;
use vortex::array::ChunkedArray;
use vortex::compute::slice;
use vortex::{Array, ArrayDType, IntoArray};
use vortex_error::{vortex_bail, VortexResult};

use crate::layouts::read::selection::RowSelector;
use crate::layouts::read::{LayoutReader, ReadResult};
use crate::layouts::{Message, RangeResult};

pub type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);
pub type RangedArray = ((usize, usize), Array);

#[derive(Debug)]
pub struct BufferedArrayReader {
    layouts: VecDeque<RangedLayoutReader>,
    arrays: VecDeque<RangedArray>,
    next_range_offset: usize,
}

impl BufferedArrayReader {
    pub fn new(layouts: VecDeque<RangedLayoutReader>) -> Self {
        Self {
            layouts,
            arrays: VecDeque::new(),
            next_range_offset: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.layouts.is_empty() && self.arrays.is_empty()
    }

    pub fn next_range(&mut self) -> VortexResult<RangeResult> {
        if self.next_range_offset == self.layouts.len() {
            return Ok(RangeResult::Rows(None));
        }

        match self.layouts[self.next_range_offset].1.next_range()? {
            RangeResult::ReadMore(m) => Ok(RangeResult::ReadMore(m)),
            RangeResult::Rows(r) => match r {
                None => {
                    self.next_range_offset += 1;
                    self.next_range()
                }
                Some(rs) => {
                    let layout_range = self.layouts[self.next_range_offset].0;
                    let offset = rs.offset(-(layout_range.0 as i64));
                    if offset.end() == layout_range.1 {
                        self.next_range_offset += 1;
                    }
                    Ok(RangeResult::Rows(Some(offset)))
                }
            },
        }
    }

    pub fn read_next(&mut self, selection: RowSelector) -> VortexResult<Option<ReadResult>> {
        if self.is_empty() {
            return Ok(None);
        }

        if let Some(rr) = buffer_read(&mut self.layouts, selection, |range, read| match read {
            ReadResult::ReadMore(_) => unreachable!("Handled by outside closure"),
            ReadResult::Selector(_) => unreachable!("Can't be a selector"),
            ReadResult::Batch(a) => self.arrays.push_back((range, a)),
        })? {
            match rr {
                read_more @ ReadResult::ReadMore(..) => return Ok(Some(read_more)),
                ReadResult::Batch(_) | ReadResult::Selector(_) => {
                    unreachable!("Buffer should only produce ReadMore")
                }
            }
        }
        self.next_range_offset = 0;

        let mut result = mem::take(&mut self.arrays);
        match result.len() {
            0 | 1 => Ok(result.pop_front().map(|(_, a)| a).map(ReadResult::Batch)),
            _ => {
                let dtype = result[0].1.dtype().clone();
                Ok(Some(ReadResult::Batch(
                    ChunkedArray::try_new(result.into_iter().map(|(_, a)| a).collect(), dtype)?
                        .into_array(),
                )))
            }
        }
    }

    pub fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        if self
            .arrays
            .front()
            .map(|((begin, _), _)| up_to_row < *begin)
            .or_else(|| {
                self.layouts
                    .front()
                    .map(|((begin, _), _)| up_to_row < *begin)
            })
            .unwrap_or(false)
        {
            vortex_bail!("Can't advance backwards {up_to_row}")
        }

        let mut new_arrays = mem::take(&mut self.arrays)
            .into_iter()
            .skip_while(|((_, end), _)| *end < up_to_row)
            .collect::<VecDeque<_>>();
        if let Some(((begin, end), carr)) = new_arrays.pop_front() {
            let slice_end = carr.len();
            let sliced = slice(carr, slice_end - (end - up_to_row), slice_end)?;

            new_arrays.push_front(((begin, end), sliced));
        };
        self.arrays = new_arrays;

        let mut new_layouts = mem::take(&mut self.layouts)
            .into_iter()
            .skip_while(|((_, end), _)| *end < up_to_row)
            .collect::<VecDeque<_>>();
        let res = if let Some(((begin, end), mut l)) = new_layouts.pop_front() {
            let advance = l.advance(up_to_row - begin);
            new_layouts.push_front(((begin, end), l));
            advance
        } else {
            Ok(vec![])
        };
        self.next_range_offset = 0;
        self.layouts = new_layouts;
        res
    }
}

fn buffer_read<F: FnMut((usize, usize), ReadResult)>(
    layouts: &mut VecDeque<RangedLayoutReader>,
    selection: RowSelector,
    mut consumer: F,
) -> VortexResult<Option<ReadResult>> {
    while let Some(((begin, end), mut layout)) = layouts.pop_front() {
        // This selection doesn't know about rows in this chunk, we should put it back and wait for another request with different range
        if selection.end() <= begin {
            layouts.push_front(((begin, end), layout));
            return Ok(None);
        }
        let layout_selection =
            RowSelector::new(Bitmap::from_range(begin as u32..end as u32), begin, end)
                .intersect(&selection)
                .offset(begin as i64);
        if let Some(rr) = layout.read_next(layout_selection)? {
            layouts.push_front(((begin, end), layout));
            match rr {
                read_more @ ReadResult::ReadMore(..) => {
                    return Ok(Some(read_more));
                }
                ReadResult::Batch(a) => consumer((begin, end), ReadResult::Batch(a)),
                ReadResult::Selector(s) => consumer((begin, end), ReadResult::Selector(s)),
            }
        } else {
            if end > selection.end() && begin < selection.end() {
                layouts.push_front(((begin, end), layout));
                return Ok(None);
            }
            continue;
        }
    }
    Ok(None)
}

#[derive(Debug)]
pub struct BufferedSelectorReader {
    layouts: VecDeque<RangedLayoutReader>,
    selectors: VecDeque<RowSelector>,
    next_range_offset: usize,
}

impl BufferedSelectorReader {
    pub fn new(layouts: VecDeque<RangedLayoutReader>) -> Self {
        Self {
            layouts,
            selectors: VecDeque::new(),
            next_range_offset: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.layouts.is_empty() && self.selectors.is_empty()
    }

    pub fn next_range(&mut self) -> VortexResult<RangeResult> {
        if self.next_range_offset == self.layouts.len() {
            return Ok(RangeResult::Rows(None));
        }

        match self.layouts[self.next_range_offset].1.next_range()? {
            RangeResult::ReadMore(m) => Ok(RangeResult::ReadMore(m)),
            RangeResult::Rows(r) => match r {
                None => {
                    self.next_range_offset += 1;
                    self.next_range()
                }
                Some(rs) => {
                    let layout_range = self.layouts[self.next_range_offset].0;
                    let offset = rs.offset(-(layout_range.0 as i64));
                    if offset.end() == layout_range.1 {
                        self.next_range_offset += 1;
                    }
                    Ok(RangeResult::Rows(Some(offset)))
                }
            },
        }
    }

    pub fn read_next(&mut self, selection: RowSelector) -> VortexResult<Option<ReadResult>> {
        if self.is_empty() {
            return Ok(None);
        }

        if let Some(rr) = buffer_read(
            &mut self.layouts,
            selection,
            |(begin, _), read| match read {
                ReadResult::ReadMore(_) => unreachable!("Handled by outside closure"),
                ReadResult::Selector(s) => self.selectors.push_back(s.offset(-(begin as i64))),
                ReadResult::Batch(_) => unreachable!("Can't be an array"),
            },
        )? {
            match rr {
                read_more @ ReadResult::ReadMore(..) => return Ok(Some(read_more)),
                ReadResult::Batch(_) | ReadResult::Selector(_) => {
                    unreachable!("Buffer should only produce ReadMore")
                }
            }
        }
        self.next_range_offset = 0;

        Ok(mem::take(&mut self.selectors)
            .into_iter()
            .reduce(|acc, a| acc.concatenate(&a))
            .map(ReadResult::Selector))
    }

    pub fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        if self
            .selectors
            .front()
            .map(|s| up_to_row < s.begin())
            .or_else(|| {
                self.layouts
                    .front()
                    .map(|((begin, _), _)| up_to_row < *begin)
            })
            .unwrap_or(false)
        {
            vortex_bail!("Can't advance backwards to {up_to_row}")
        }

        let mut new_selectors = mem::take(&mut self.selectors)
            .into_iter()
            .skip_while(|s| s.end() < up_to_row)
            .collect::<VecDeque<_>>();
        if let Some(s) = new_selectors.pop_front() {
            if let Some(rs) = s.advance(up_to_row) {
                new_selectors.push_front(rs);
            }
        };
        self.selectors = new_selectors;

        let mut new_layouts = mem::take(&mut self.layouts)
            .into_iter()
            .skip_while(|((_, end), _)| *end < up_to_row)
            .collect::<VecDeque<_>>();
        let res = if let Some(((begin, end), mut l)) = new_layouts.pop_front() {
            let advance = l.advance(up_to_row - begin);
            new_layouts.push_front(((begin, end), l));
            advance
        } else {
            Ok(vec![])
        };
        self.next_range_offset = 0;
        self.layouts = new_layouts;
        res
    }
}
