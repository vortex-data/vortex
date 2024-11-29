use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{ready, Context, Poll};

use futures::future::BoxFuture;
use futures::FutureExt as _;
use vortex_array::ArrayData;
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_io::{Dispatch as _, IoDispatcher, VortexReadAt};

use super::stream::{read_ranges, StreamMessages};
use super::{LayoutMessageCache, LayoutReader, MessageLocator, MetadataRead};
use crate::read::stream::Message;

pub struct MetadataFetcher<R: VortexReadAt> {
    input: R,
    dispatcher: Arc<IoDispatcher>,
    root_layout: Box<dyn LayoutReader>,
    layout_cache: Arc<RwLock<LayoutMessageCache>>,
    state: State,
}

enum State {
    Initial,
    Reading(BoxFuture<'static, VortexResult<StreamMessages>>),
}

impl<R: VortexReadAt + Unpin> MetadataFetcher<R> {
    pub fn fetch(
        input: R,
        dispatcher: Arc<IoDispatcher>,
        root_layout: Box<dyn LayoutReader>,
        layout_cache: Arc<RwLock<LayoutMessageCache>>,
    ) -> Self {
        Self {
            input,
            dispatcher,
            root_layout,
            layout_cache,
            state: State::Initial,
        }
    }

    /// Schedule an asynchronous read of several byte ranges.
    ///
    /// IO is scheduled on the provided IO dispatcher.
    fn read_ranges(
        &self,
        ranges: Vec<MessageLocator>,
    ) -> BoxFuture<'static, VortexResult<StreamMessages>> {
        let reader = self.input.clone();

        let result_rx = self
            .dispatcher
            .dispatch(move || async move { read_ranges(reader, ranges).await })
            .vortex_expect("dispatch async task");

        result_rx
            .map(|res| match res {
                Ok(result) => result,
                Err(e) => vortex_bail!("dispatcher channel canceled: {e}"),
            })
            .boxed()
    }
}

impl<R: VortexReadAt + Unpin> Future for MetadataFetcher<R> {
    type Output = VortexResult<Option<Vec<Option<ArrayData>>>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            match &mut self.state {
                State::Initial => match self.root_layout.read_metadata()? {
                    MetadataRead::ReadMore(messages) => {
                        let read_future = self.read_ranges(messages);
                        self.state = State::Reading(read_future);
                    }
                    MetadataRead::Batches(array_data) => {
                        return Poll::Ready(Ok(Some(array_data)));
                    }
                    MetadataRead::None => {
                        return Poll::Ready(Ok(None));
                    }
                },
                State::Reading(ref mut f) => {
                    let messages = ready!(f.poll_unpin(cx))?;

                    match self.layout_cache.write() {
                        Ok(mut cache) => {
                            for Message(message_id, bytes) in messages.into_iter() {
                                cache.set(message_id, bytes);
                            }
                        }
                        Err(poison) => {
                            vortex_panic!("Failed to write to message cache: {poison}")
                        }
                    }

                    self.state = State::Initial;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, RwLock};

    use vortex_array::array::{ChunkedArray, StructArray};
    use vortex_array::compute::scalar_at;
    use vortex_array::{ArrayDType as _, ArrayData, IntoArrayData as _};
    use vortex_buffer::{Buffer, BufferString};
    use vortex_io::IoDispatcher;

    use crate::metadata::MetadataFetcher;
    use crate::{
        read_initial_bytes, read_layout_from_initial, LayoutDeserializer, LayoutMessageCache,
        RelativeLayoutCache, Scan, VortexFileWriter,
    };

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_metadata_works() {
        let name_chunk1 = ArrayData::from_iter(vec![
            Some("Joseph".to_owned()),
            Some("James".to_owned()),
            Some("Angela".to_owned()),
        ]);
        let age_chunk1 = ArrayData::from_iter(vec![Some(25_i32), Some(31), None]);
        let name_chunk2 = ArrayData::from_iter(vec![
            Some("Pharrell".to_owned()),
            Some("Khalil".to_owned()),
            Some("Mikhail".to_owned()),
            None,
        ]);
        let age_chunk2 = ArrayData::from_iter(vec![Some(57_i32), Some(18), None, Some(32)]);

        let chunk1 = StructArray::from_fields(&[("name", name_chunk1), ("age", age_chunk1)])
            .unwrap()
            .into_array();
        let chunk2 = StructArray::from_fields(&[("name", name_chunk2), ("age", age_chunk2)])
            .unwrap()
            .into_array();
        let dtype = chunk1.dtype().clone();

        let array = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
            .unwrap()
            .into_array();

        let buffer = Vec::new();
        let written_bytes = VortexFileWriter::new(buffer)
            .write_array_columns(array)
            .await
            .unwrap()
            .finalize()
            .await
            .unwrap();
        let written_bytes = Buffer::from(written_bytes);

        let n_bytes = written_bytes.len();
        let initial_read = read_initial_bytes(&written_bytes, n_bytes as u64)
            .await
            .unwrap();
        let lazy_dtype = Arc::new(initial_read.lazy_dtype().unwrap());
        let layout_deserializer = LayoutDeserializer::default();
        let layout_message_cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let layout_reader = read_layout_from_initial(
            &initial_read,
            &layout_deserializer,
            Scan::empty(),
            RelativeLayoutCache::new(layout_message_cache.clone(), lazy_dtype.clone()),
        )
        .unwrap();
        let io = IoDispatcher::default();
        let metadata_table = MetadataFetcher::fetch(
            written_bytes,
            io.into(),
            layout_reader,
            layout_message_cache,
        )
        .await
        .unwrap();

        assert!(metadata_table.is_some());
        let metadata_table = metadata_table.unwrap();
        assert!(metadata_table.len() == 2);
        assert!(metadata_table.iter().all(Option::is_some));

        let name_metadata_table = metadata_table[0]
            .as_ref()
            .unwrap()
            .as_struct_array()
            .unwrap();

        let min = name_metadata_table.field_by_name("min").unwrap();
        let chunk1_min = scalar_at(&min, 0).unwrap();
        let chunk2_min = scalar_at(&min, 1).unwrap();
        assert_eq!(
            chunk1_min.as_utf8().value(),
            Some(BufferString::from("Angela"))
        );
        assert_eq!(
            chunk2_min.as_utf8().value(),
            Some(BufferString::from("Khalil"))
        );

        let max = name_metadata_table.field_by_name("max").unwrap();
        let chunk1_max = scalar_at(&max, 0).unwrap();
        let chunk2_max = scalar_at(&max, 1).unwrap();
        assert_eq!(
            chunk1_max.as_utf8().value(),
            Some(BufferString::from("Joseph"))
        );
        assert_eq!(
            chunk2_max.as_utf8().value(),
            Some(BufferString::from("Pharrell"))
        );

        let null_count = name_metadata_table.field_by_name("null_count").unwrap();
        let chunk1_null_count = scalar_at(&null_count, 0).unwrap();
        let chunk2_null_count = scalar_at(&null_count, 1).unwrap();
        assert_eq!(
            chunk1_null_count.as_primitive().typed_value::<u64>(),
            Some(0)
        );
        assert_eq!(
            chunk2_null_count.as_primitive().typed_value::<u64>(),
            Some(1)
        );

        let age_metadata_table = metadata_table[1]
            .as_ref()
            .unwrap()
            .as_struct_array()
            .unwrap();

        let min = age_metadata_table.field_by_name("min").unwrap();
        let chunk1_min = scalar_at(&min, 0).unwrap();
        let chunk2_min = scalar_at(&min, 1).unwrap();
        assert_eq!(chunk1_min.as_primitive().typed_value::<i32>(), Some(25));
        assert_eq!(chunk2_min.as_primitive().typed_value::<i32>(), Some(18));

        let max = age_metadata_table.field_by_name("max").unwrap();
        let chunk1_max = scalar_at(&max, 0).unwrap();
        let chunk2_max = scalar_at(&max, 1).unwrap();
        assert_eq!(chunk1_max.as_primitive().typed_value::<i32>(), Some(31));
        assert_eq!(chunk2_max.as_primitive().typed_value::<i32>(), Some(57));

        let null_count = age_metadata_table.field_by_name("null_count").unwrap();
        let chunk1_null_count = scalar_at(&null_count, 0).unwrap();
        let chunk2_null_count = scalar_at(&null_count, 1).unwrap();
        assert_eq!(
            chunk1_null_count.as_primitive().typed_value::<u64>(),
            Some(1)
        );
        assert_eq!(
            chunk2_null_count.as_primitive().typed_value::<u64>(),
            Some(1)
        );
    }
}
