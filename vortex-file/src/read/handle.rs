use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use futures::stream;
use vortex_dtype::DType;
use vortex_error::{VortexResult, VortexUnwrap as _};
use vortex_io::{IoDispatcher, VortexReadAt};

use super::splits::SplitsAccumulator;
use super::{LayoutMessageCache, LayoutReader, LazyDType, RowMask, VortexReadArrayStream};
use crate::read::buffered::{BufferedLayoutReader, ReadArray};
use crate::read::splits::ReadRowMask;

#[derive(Clone)]
pub struct VortexReadHandle<R> {
    input: R,
    dtype: Arc<LazyDType>,
    row_count: u64,
    splits: BTreeSet<usize>,
    layout_reader: Arc<dyn LayoutReader>,
    filter_reader: Option<Arc<dyn LayoutReader>>,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    row_mask: Option<RowMask>,
    io_dispatcher: Arc<IoDispatcher>,
}

impl<R: VortexReadAt + Unpin> VortexReadHandle<R> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        input: R,
        layout_reader: Arc<dyn LayoutReader>,
        filter_reader: Option<Arc<dyn LayoutReader>>,
        messages_cache: Arc<RwLock<LayoutMessageCache>>,
        dtype: Arc<LazyDType>,
        row_count: u64,
        row_mask: Option<RowMask>,
        dispatcher: Arc<IoDispatcher>,
    ) -> VortexResult<Self> {
        let mut reader_splits = BTreeSet::new();
        layout_reader.add_splits(0, &mut reader_splits)?;
        if let Some(ref fr) = filter_reader {
            fr.add_splits(0, &mut reader_splits)?;
        }

        Ok(Self {
            input,
            dtype,
            row_count,
            messages_cache,
            row_mask,
            layout_reader,
            filter_reader,
            splits: reader_splits,
            io_dispatcher: dispatcher,
        })
    }

    /// Returns the type of the file's top-level array.
    pub fn dtype(&self) -> &DType {
        // FIXME(ngates): why is this allowed to unwrap?
        self.dtype.value().vortex_unwrap()
    }

    /// Returns the total row count of the Vortex file, before any filtering.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    pub fn splits(&self) -> &BTreeSet<usize> {
        &self.splits
    }

    pub fn into_stream(self) -> VortexResult<VortexReadArrayStream<R>> {
        let mut split_accumulator = SplitsAccumulator::new(self.row_count, self.row_mask);
        split_accumulator.append_splits(self.splits);
        let splits_stream = stream::iter(split_accumulator);

        // Set up a stream of RowMask that result from applying a filter expression over the file.
        let mask_iterator = if let Some(fr) = self.filter_reader {
            Box::new(BufferedLayoutReader::new(
                self.input.clone(),
                self.io_dispatcher.clone(),
                splits_stream,
                ReadRowMask::new(fr),
                self.messages_cache.clone(),
            )) as _
        } else {
            Box::new(splits_stream) as _
        };

        // Set up a stream of result ArrayData that result from applying the filter and projection
        // expressions over the file.
        let array_reader = BufferedLayoutReader::new(
            self.input,
            self.io_dispatcher,
            mask_iterator,
            ReadArray::new(self.layout_reader),
            self.messages_cache,
        );

        Ok(VortexReadArrayStream::new(
            self.dtype,
            self.row_count,
            array_reader,
        ))
    }

    pub fn ranged_stream(
        mut self,
        begin: usize,
        end: usize,
    ) -> VortexResult<VortexReadArrayStream<R>> {
        self.row_mask = match self.row_mask {
            Some(mask) => Some(mask.and_rowmask(RowMask::new_valid_between(begin, end))?),
            None => Some(RowMask::new_valid_between(begin, end)),
        };

        self.into_stream()
    }
}
