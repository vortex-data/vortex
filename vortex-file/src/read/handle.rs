use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use futures::stream;
use itertools::Itertools;
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
    splits: Vec<(usize, usize)>,
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
        io_dispatcher: Arc<IoDispatcher>,
    ) -> VortexResult<Self> {
        let mut reader_splits = BTreeSet::new();
        layout_reader.add_splits(0, &mut reader_splits)?;
        if let Some(ref fr) = filter_reader {
            fr.add_splits(0, &mut reader_splits)?;
        }

        reader_splits.insert(row_count as usize);

        let splits = reader_splits.into_iter().tuple_windows().collect();

        Ok(Self {
            input,
            dtype,
            row_count,
            splits,
            layout_reader,
            filter_reader,
            messages_cache,
            row_mask,
            io_dispatcher,
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

    /// Returns a set of row splits in the file, that can be used to inform on how to split it horizontally.
    pub fn splits(&self) -> &[(usize, usize)] {
        &self.splits
    }

    /// Create a stream over all data from the handle
    pub fn into_stream(self) -> VortexReadArrayStream<R> {
        let splits_vec = Vec::from_iter(self.splits().iter().copied());
        let split_accumulator = SplitsAccumulator::new(splits_vec.into_iter(), self.row_mask);

        let splits_stream = stream::iter(split_accumulator);

        // Set up a stream of RowMask that result from applying a filter expression over the file.
        let mask_iterator = if let Some(fr) = &self.filter_reader {
            Box::new(BufferedLayoutReader::new(
                self.input.clone(),
                self.io_dispatcher.clone(),
                splits_stream,
                ReadRowMask::new(fr.clone()),
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

        VortexReadArrayStream::new(self.dtype, self.row_count, array_reader)
    }

    /// Create a stream over a specific row range from the handle
    pub fn stream_range(
        mut self,
        begin: usize,
        end: usize,
    ) -> VortexResult<VortexReadArrayStream<R>> {
        self.row_mask = match self.row_mask {
            Some(mask) => Some(mask.and_rowmask(RowMask::new_valid_between(begin, end))?),
            None => Some(RowMask::new_valid_between(begin, end)),
        };

        Ok(self.into_stream())
    }
}
