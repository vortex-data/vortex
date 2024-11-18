use std::collections::VecDeque;
use std::mem;

use vortex_array::array::ChunkedArray;
use vortex_array::compute::unary::scalar_at;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_scalar::BoolScalar;

use crate::pruning::PruningPredicate;
use crate::read::mask::RowMask;
use crate::read::{BatchRead, LayoutReader, MessageLocator, Scan};

pub type RangedLayoutReader = ((usize, usize), Box<dyn LayoutReader>);

/// Layout reader that continues reading children until all rows referenced in the mask have been handled
#[derive(Debug)]
pub struct BufferedLayoutReader {
    metadata_reader: Option<MetadataReader>,
    layouts: VecDeque<(usize, RangedLayoutReader)>,
    arrays: Vec<ArrayData>,
    n_chunks: usize,
    scan: Scan,
    chunk_mask: Option<ArrayData>,
}

#[derive(Debug)]
pub enum MetadataReader {
    NoMetadata,
    NotYetRead(Box<dyn LayoutReader>),
    Read(ArrayData),
}

impl BufferedLayoutReader {
    pub fn new(
        metadata_reader: MetadataReader,
        layouts: VecDeque<RangedLayoutReader>,
        scan: Scan,
    ) -> Self {
        let n_chunks = layouts.len();
        Self {
            metadata_reader: Some(metadata_reader),
            layouts: layouts.into_iter().enumerate().collect::<VecDeque<_>>(),
            arrays: Vec::new(),
            n_chunks,
            scan,
            chunk_mask: None,
        }
    }

    fn ensure_pruning_mask(&mut self) -> VortexResult<Option<Vec<MessageLocator>>> {
        if self.chunk_mask.is_some() {
            return Ok(None);
        }

        let metadata = match mem::take(&mut self.metadata_reader) {
            metadata_reader @ Some(MetadataReader::NoMetadata) => {
                self.metadata_reader = metadata_reader;
                None
            }
            Some(MetadataReader::NotYetRead(mut reader)) => {
                match reader.read_selection(&RowMask::new_valid_between(0, self.n_chunks))? {
                    Some(BatchRead::ReadMore(messages)) => {
                        self.metadata_reader = Some(MetadataReader::NotYetRead(reader));
                        return Ok(Some(messages));
                    }
                    Some(BatchRead::Batch(array)) => {
                        self.metadata_reader = Some(MetadataReader::Read(array.clone()));
                        Some(array)
                    }
                    None => {
                        vortex_bail!("unexpected end of stream while reading metadata array")
                    }
                }
            }
            Some(MetadataReader::Read(array)) => {
                self.metadata_reader = Some(MetadataReader::Read(array.clone()));
                Some(array)
            }
            None => vortex_bail!("Called buffer_read while buffer_read was running"),
        };

        self.chunk_mask = self
            .scan
            .expr
            .as_ref()
            .zip(metadata)
            .and_then(|(expression, metadata)| {
                PruningPredicate::try_new(expression).map(|p| p.evaluate(&metadata))
            })
            .transpose()?
            .flatten();

        Ok(None)
    }

    // TODO(robert): Support out of order reads
    fn buffer_read(&mut self, mask: &RowMask) -> VortexResult<Option<Vec<MessageLocator>>> {
        if let Some(requested_messages) = self.ensure_pruning_mask()? {
            return Ok(Some(requested_messages));
        }

        while let Some((index, ((begin, end), layout))) = self.layouts.pop_front() {
            let chunk_is_pruned = self
                .chunk_mask
                .as_ref()
                .map(|chunk_mask| -> VortexResult<_> {
                    Ok(BoolScalar::try_from(&scalar_at(chunk_mask, index)?)?
                        .value()
                        .vortex_expect("chunk_mask should be nonnullable"))
                })
                .transpose()?
                .unwrap_or(false);
            if chunk_is_pruned || mask.end() > begin && mask.begin() <= end {
                self.layouts.push_front((index, ((begin, end), layout)));
                break;
            }
        }

        while let Some((index, ((begin, end), mut layout))) = self.layouts.pop_front() {
            // This selection doesn't know about rows in this chunk, we should put it back and wait for another request with different range
            if mask.end() <= begin || mask.begin() > end {
                self.layouts.push_front((index, ((begin, end), layout)));
                return Ok(None);
            }
            let layout_selection = mask.slice(begin, end).shift(begin)?;
            if let Some(rr) = layout.read_selection(&layout_selection)? {
                match rr {
                    BatchRead::ReadMore(m) => {
                        self.layouts.push_front((index, ((begin, end), layout)));
                        return Ok(Some(m));
                    }
                    BatchRead::Batch(a) => {
                        self.arrays.push(a);
                        if end > mask.end() {
                            self.layouts.push_front((index, ((begin, end), layout)));
                            return Ok(None);
                        }
                    }
                }
            } else {
                if end > mask.end() && begin < mask.end() {
                    self.layouts.push_front((index, ((begin, end), layout)));
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
