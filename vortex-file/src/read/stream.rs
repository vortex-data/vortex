use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use futures_util::{StreamExt, TryStreamExt};
use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexResult, VortexUnwrap};
use vortex_io::VortexReadAt;

use crate::read::buffered::{BufferedLayoutReader, ReadArray};
use crate::read::mask::RowMask;
use crate::LazyDType;

/// An asynchronous Vortex file that returns a [`Stream`] of [`ArrayData`]s.
///
/// The file may be read from any source implementing [`VortexReadAt`], such
/// as memory, disk, and object storage.
///
/// Use [VortexReadBuilder][crate::read::builder::VortexReadBuilder] to build one
/// from a reader.
pub struct VortexReadArrayStream<R> {
    dtype: Arc<LazyDType>,
    row_count: u64,
    array_reader: BufferedLayoutReader<
        R,
        Box<dyn Stream<Item = VortexResult<RowMask>> + Send + Unpin>,
        ArrayData,
        ReadArray,
    >,
}

impl<R: VortexReadAt + Unpin> VortexReadArrayStream<R> {
    pub(crate) fn new(
        dtype: Arc<LazyDType>,
        row_count: u64,
        array_reader: BufferedLayoutReader<
            R,
            Box<dyn Stream<Item = VortexResult<RowMask>> + Send + Unpin>,
            ArrayData,
            ReadArray,
        >,
    ) -> Self {
        Self {
            dtype,
            row_count,
            array_reader,
        }
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

    /// Read the whole stream into a single [`ArrayData`].
    pub async fn read_all(self) -> VortexResult<ArrayData> {
        let dtype = self.dtype().clone();
        let arrays = self.try_collect::<Vec<_>>().await?;
        if arrays.len() == 1 {
            arrays.into_iter().next().ok_or_else(|| {
                vortex_panic!(
                    "Should be impossible: vecs.len() == 1 but couldn't get first element"
                )
            })
        } else {
            ChunkedArray::try_new(arrays, dtype).map(|e| e.into_array())
        }
    }
}

impl<R: VortexReadAt + Unpin> Stream for VortexReadArrayStream<R> {
    type Item = VortexResult<ArrayData>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.array_reader.poll_next_unpin(cx)
    }
}
