use std::collections::VecDeque;
use std::mem;

use vortex_array::array::ChunkedArray;
use vortex_array::compute::unary::scalar_at;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::{VortexExpect as _, VortexResult};
use vortex_scalar::BoolScalar;

use crate::read::mask::RowMask;
use crate::read::{BatchRead, LayoutReader, MessageLocator};

pub type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);

/// Layout reader that continues reading children until all rows referenced in the mask have been handled
#[derive(Debug)]
pub struct BufferedLayoutReader {
    splits: Vec<(usize, (usize, usize))>,
    layouts: VecDeque<RangedLayoutReader>,
    arrays: Vec<ArrayData>,
    chunk_mask: Option<ArrayData>,
}

impl BufferedLayoutReader {
    pub fn new(layouts: VecDeque<RangedLayoutReader>, chunk_mask: Option<ArrayData>) -> Self {
        Self {
            splits: layouts
                .iter()
                .map(|((begin, end), _)| (*begin, *end))
                .enumerate()
                .collect::<Vec<_>>(),
            layouts,
            arrays: Vec::new(),
            chunk_mask,
        }
    }

    // TODO(robert): Support out of order reads
    fn buffer_read(&mut self, mask: &RowMask) -> VortexResult<Option<Vec<MessageLocator>>> {
        while let Some(((begin, end), layout)) = self.layouts.pop_front() {
            if mask.end() > begin && mask.begin() <= end {
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

    pub fn is_pruned(&self, begin: usize, end: usize) -> VortexResult<bool> {
        // println!(
        //     "Buffered::is_pruned {}-{} {}",
        //     begin,
        //     end,
        //     self.chunk_mask.is_some()
        // );
        let Some(ref chunk_mask) = self.chunk_mask else {
            return Ok(false);
        };

        for (index, (chunk_begin, chunk_end)) in self.splits.iter() {
            if !(*chunk_begin == begin && *chunk_end == end) {
                continue;
            }

            let chunk_is_pruned = BoolScalar::try_from(&scalar_at(chunk_mask, *index)?)?
                .value()
                .vortex_expect("chunk_mask should be nonnullable");
            // println!(
            //     "is_pruned: index {} {}-{} {}",
            //     index, begin, end, chunk_is_pruned
            // );
            return Ok(chunk_is_pruned);
        }
        // println!("could not find {} {} in layouts", begin, end);
        Ok(false)
    }
}
