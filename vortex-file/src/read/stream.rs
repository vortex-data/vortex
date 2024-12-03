use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};

use futures::Stream;
use futures_util::{StreamExt, TryStreamExt};
use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult, VortexUnwrap};
use vortex_io::{IoDispatcher, VortexReadAt};

use crate::read::buffered::{BufferedLayoutReader, ReadArray};
use crate::read::cache::LayoutMessageCache;
use crate::read::mask::RowMask;
use crate::read::splits::{FixedSplitIterator, ReadRowMask};
use crate::read::LayoutReader;
use crate::LazyDType;

/// An asynchronous Vortex file that returns a [`Stream`] of [`ArrayData`]s.
///
/// The file may be read from any source implementing [`VortexReadAt`], such
/// as memory, disk, and object storage.
///
/// Use [VortexReadBuilder][crate::read::builder::VortexReadBuilder] to build one
/// from a reader.
pub struct VortexFileArrayStream<R> {
    dtype: Arc<LazyDType>,
    row_count: u64,
    array_reader: BufferedLayoutReader<
        R,
        Box<dyn Stream<Item = VortexResult<RowMask>> + Send + Unpin>,
        ArrayData,
        ReadArray,
    >,
}

impl<R: VortexReadAt + Unpin> VortexFileArrayStream<R> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        input: R,
        layout_reader: Box<dyn LayoutReader>,
        filter_reader: Option<Box<dyn LayoutReader>>,
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
        let mut split_iterator = FixedSplitIterator::new(row_count, row_mask);
        split_iterator.additional_splits(&mut reader_splits)?;

        let mask_iterator = if let Some(fr) = filter_reader {
            Box::new(BufferedLayoutReader::new(
                input.clone(),
                dispatcher.clone(),
                split_iterator,
                ReadRowMask::new(fr),
                messages_cache.clone(),
            )) as _
        } else {
            Box::new(split_iterator) as _
        };

        let array_reader = BufferedLayoutReader::new(
            input,
            dispatcher,
            mask_iterator,
            ReadArray::new(layout_reader),
            messages_cache,
        );

        Ok(Self {
            dtype,
            row_count,
            array_reader,
        })
    }

    pub fn dtype(&self) -> &DType {
        self.dtype.value().vortex_unwrap()
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }
}

impl<R: VortexReadAt + Unpin> Stream for VortexFileArrayStream<R> {
    type Item = VortexResult<ArrayData>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.array_reader.poll_next_unpin(cx)
    }
}

impl<R: VortexReadAt + Unpin> VortexFileArrayStream<R> {
    pub async fn read_all(self) -> VortexResult<ArrayData> {
        let dtype = self.dtype().clone();
        let vecs: Vec<ArrayData> = self.try_collect().await?;
        if vecs.len() == 1 {
            vecs.into_iter().next().ok_or_else(|| {
                vortex_panic!(
                    "Should be impossible: vecs.len() == 1 but couldn't get first element"
                )
            })
        } else {
            ChunkedArray::try_new(vecs, dtype).map(|e| e.into_array())
        }
    }
}
