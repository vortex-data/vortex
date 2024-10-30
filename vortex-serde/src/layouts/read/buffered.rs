use std::collections::VecDeque;
use std::mem;

use croaring::Bitmap;
use vortex::array::ChunkedArray;
use vortex::{Array, ArrayDType, IntoArray};
use vortex_error::VortexResult;

use crate::layouts::read::selection::RowSelector;
use crate::layouts::read::{LayoutReader, ReadResult};
use crate::layouts::Message;

pub type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);

#[derive(Debug)]
pub struct BufferedLayoutReader {
    layouts: VecDeque<RangedLayoutReader>,
    arrays: Vec<Array>,
}

impl BufferedLayoutReader {
    pub fn new(layouts: VecDeque<RangedLayoutReader>) -> Self {
        Self {
            layouts,
            arrays: Vec::new(),
        }
    }

    fn buffer_read(&mut self, selection: RowSelector) -> VortexResult<Option<Vec<Message>>> {
        while let Some(((begin, end), mut layout)) = self.layouts.pop_front() {
            // This selection doesn't know about rows in this chunk, we should put it back and wait for another request with different range
            if selection.end() <= begin || selection.begin() > end {
                self.layouts.push_front(((begin, end), layout));
                return Ok(None);
            }
            let layout_selection =
                RowSelector::new(Bitmap::from_range(begin as u32..end as u32), begin, end)
                    .intersect(&selection)
                    .offset(begin as i64);
            if let Some(rr) = layout.read_selection(layout_selection)? {
                match rr {
                    ReadResult::ReadMore(m) => {
                        self.layouts.push_front(((begin, end), layout));
                        return Ok(Some(m));
                    }
                    ReadResult::Batch(a) => {
                        self.arrays.push(a);
                        if end > selection.end() {
                            self.layouts.push_front(((begin, end), layout));
                            return Ok(None);
                        }
                    }
                }
            } else {
                if end > selection.end() && begin < selection.end() {
                    self.layouts.push_front(((begin, end), layout));
                    return Ok(None);
                }
                continue;
            }
        }
        Ok(None)
    }

    pub fn read_next(&mut self, selection: RowSelector) -> VortexResult<Option<ReadResult>> {
        if let Some(bufs) = self.buffer_read(selection)? {
            return Ok(Some(ReadResult::ReadMore(bufs)));
        }

        let mut result = mem::take(&mut self.arrays);
        match result.len() {
            0 | 1 => Ok(result.pop().map(ReadResult::Batch)),
            _ => {
                let dtype = result[0].dtype().clone();
                Ok(Some(ReadResult::Batch(
                    ChunkedArray::try_new(result, dtype)?.into_array(),
                )))
            }
        }
    }
}
