use std::collections::VecDeque;

use vortex_array::array::ChunkedArray;
use vortex_array::compute::slice;
use vortex_array::{Array, ArrayDType, IntoArray};
use vortex_error::VortexResult;

use crate::layouts::read::{LayoutReader, ReadResult};

#[derive(Debug)]
pub struct BufferedReader {
    layouts: VecDeque<Box<dyn LayoutReader>>,
    arrays: VecDeque<Array>,
    batch_size: usize,
}

impl BufferedReader {
    pub fn new(layouts: VecDeque<Box<dyn LayoutReader>>, batch_size: usize) -> Self {
        Self {
            layouts,
            arrays: Default::default(),
            batch_size,
        }
    }

    fn is_empty(&self) -> bool {
        self.layouts.is_empty() && self.arrays.is_empty()
    }

    fn buffered_row_count(&self) -> usize {
        self.arrays.iter().map(Array::len).sum()
    }

    fn buffer(&mut self) -> VortexResult<Option<ReadResult>> {
        while self.buffered_row_count() < self.batch_size {
            if let Some(mut layout) = self.layouts.pop_front() {
                if let Some(rr) = layout.read_next()? {
                    self.layouts.push_front(layout);
                    match rr {
                        read_more @ ReadResult::ReadMore(..) => {
                            return Ok(Some(read_more));
                        }
                        ReadResult::Batch(a) => self.arrays.push_back(a),
                    }
                } else {
                    continue;
                }
            } else {
                return Ok(None);
            }
        }
        Ok(None)
    }

    pub fn read(&mut self) -> VortexResult<Option<ReadResult>> {
        if self.is_empty() {
            return Ok(None);
        }

        if let Some(rr) = self.buffer()? {
            match rr {
                read_more @ ReadResult::ReadMore(..) => return Ok(Some(read_more)),
                ReadResult::Batch(_) => {
                    unreachable!("Batches should be handled inside the buffer call")
                }
            }
        }

        let mut rows_to_read = if self.layouts.is_empty() {
            usize::min(self.batch_size, self.buffered_row_count())
        } else {
            self.batch_size
        };

        let mut result = Vec::new();

        while rows_to_read != 0 {
            match self.arrays.pop_front() {
                None => break,
                Some(array) => {
                    if array.len() > rows_to_read {
                        let taken = slice(&array, 0, rows_to_read)?;
                        let leftover = slice(&array, rows_to_read, array.len())?;
                        self.arrays.push_front(leftover);
                        rows_to_read -= taken.len();
                        result.push(taken);
                    } else {
                        rows_to_read -= array.len();
                        result.push(array);
                    }
                }
            }
        }

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
