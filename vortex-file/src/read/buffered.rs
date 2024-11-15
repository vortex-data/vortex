use std::collections::VecDeque;
use std::mem;

use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::read::mask::RowMask;
use crate::read::{BatchRead, LayoutReader, MessageLocator};

pub type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);

/// Layout reader that continues reading children until all rows referenced in the mask have been handled
#[derive(Debug)]
pub struct BufferedLayoutReader {
    layouts: VecDeque<RangedLayoutReader>,
    arrays: Vec<ArrayData>,
}

impl BufferedLayoutReader {
    pub fn new(layouts: VecDeque<RangedLayoutReader>) -> Self {
        Self {
            layouts,
            arrays: Vec::new(),
        }
    }

    // TODO(robert): Support out of order reads
    fn buffer_read(&mut self, mask: &RowMask) -> VortexResult<Option<Vec<MessageLocator>>> {
        while let Some(((begin, end), layout)) = self.layouts.pop_front() {
            if mask.begin() <= begin && begin < mask.end()
                || mask.begin() < end && end <= mask.end()
            {
                self.layouts.push_front(((begin, end), layout));
                break;
            }
        }

        while let Some(((begin, end), mut layout)) = self.layouts.pop_front() {
            // This selection doesn't know about rows in this chunk, we should put it back and wait for another request with different range
            if mask.end() <= begin || mask.begin() > end {
                self.layouts.push_front(((begin, end), layout));
                return Ok(None);
            }
            let layout_selection = mask.slice(begin, end).shift(begin)?;
            if let Some(rr) = layout.read_selection(&layout_selection)? {
                match rr {
                    BatchRead::ReadMore(m) => {
                        self.layouts.push_front(((begin, end), layout));
                        return Ok(Some(m));
                    }
                    BatchRead::Batch(a) => {
                        self.arrays.push(a);
                        if end > mask.end() {
                            self.layouts.push_front(((begin, end), layout));
                            return Ok(None);
                        }
                    }
                }
            } else {
                if end > mask.end() && begin < mask.end() {
                    self.layouts.push_front(((begin, end), layout));
                    return Ok(None);
                }
                continue;
            }
        }
        Ok(None)
    }

    pub fn read_next(&mut self, mask: &RowMask) -> VortexResult<Option<BatchRead>> {
        if let Some(bufs) = self.buffer_read(mask)? {
            return Ok(Some(BatchRead::ReadMore(bufs)));
        }

        let mut result = mem::take(&mut self.arrays);
        match result.len() {
            0 | 1 => Ok(result.pop().map(BatchRead::Batch)),
            _ => {
                let dtype = result[0].dtype().clone();
                Ok(Some(BatchRead::Batch(
                    ChunkedArray::try_new(result, dtype)?.into_array(),
                )))
            }
        }
    }
}
