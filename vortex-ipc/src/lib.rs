//! Vortex IPC messages and associated readers and writers.
//!
//! Vortex provides an IPC messaging format to exchange array data over a streaming
//! interface. The format emits message headers in FlatBuffer format, along with their
//! data buffers.
//!
//! This crate provides both in-memory message representations for holding IPC messages
//! before/after serialization, as well as streaming readers and writers that sit on top
//! of any type implementing `VortexRead` or `VortexWrite` respectively.

pub mod messages;
pub mod stream_reader;
pub mod stream_writer;

/// All messages in Vortex are aligned to start at a multiple of 64 bytes.
///
/// This is a multiple of the native alignment for all PTypes,
/// thus all buffers allocated with this alignment are naturally aligned
/// for any data we may put inside of it.
pub const ALIGNMENT: usize = 64;

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod test {
    use std::sync::Arc;

    use bytes::Bytes;
    use futures_executor::block_on;
    use futures_util::{pin_mut, StreamExt, TryStreamExt};
    use itertools::Itertools;
    use vortex_array::array::{ChunkedArray, PrimitiveArray, PrimitiveEncoding};
    use vortex_array::encoding::EncodingVTable;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::{ArrayDType, Context, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_io::VortexBufReader;

    use crate::stream_reader::StreamArrayReader;
    use crate::stream_writer::StreamArrayWriter;

    fn write_ipc<A: IntoArrayData>(array: A) -> Vec<u8> {
        block_on(async {
            StreamArrayWriter::new(vec![])
                .write_array(array.into_array())
                .await
                .unwrap()
                .into_inner()
        })
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_empty_index() -> VortexResult<()> {
        let data = PrimitiveArray::from((0i32..3_000_000).collect_vec());
        let buffer = write_ipc(data);

        let indices = PrimitiveArray::from(vec![1, 2, 10]).into_array();

        let ctx = Arc::new(Context::default());
        let stream_reader =
            StreamArrayReader::try_new(VortexBufReader::new(Bytes::from(buffer)), ctx)
                .await
                .unwrap()
                .load_dtype()
                .await
                .unwrap();
        let reader = stream_reader.into_array_stream();

        let result_iter = reader.take_rows(indices)?;
        pin_mut!(result_iter);

        let _result = block_on(async { result_iter.next().await.unwrap().unwrap() });
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_write_read_chunked() -> VortexResult<()> {
        let indices = PrimitiveArray::from(vec![
            10u32, 11, 12, 13, 100_000, 2_999_999, 2_999_999, 3_000_000,
        ])
        .into_array();

        // NB: the order is reversed here to ensure we aren't grabbing indexes instead of values
        let data = PrimitiveArray::from((0i32..3_000_000).rev().collect_vec()).into_array();
        let data2 =
            PrimitiveArray::from((3_000_000i32..6_000_000).rev().collect_vec()).into_array();
        let chunked = ChunkedArray::try_new(vec![data.clone(), data2], data.dtype().clone())?;
        let buffer = write_ipc(chunked);
        let buffer = Buffer::from(buffer);

        let ctx = Arc::new(Context::default());
        let stream_reader = StreamArrayReader::try_new(VortexBufReader::new(buffer), ctx)
            .await
            .unwrap()
            .load_dtype()
            .await
            .unwrap();

        let take_iter = stream_reader.into_array_stream().take_rows(indices)?;
        pin_mut!(take_iter);

        let next = block_on(async { take_iter.try_next().await })?.expect("Expected a chunk");
        assert_eq!(next.encoding().id(), PrimitiveEncoding.id());

        assert_eq!(
            next.into_primitive().unwrap().maybe_null_slice::<i32>(),
            vec![2999989, 2999988, 2999987, 2999986, 2899999, 0, 0]
        );
        assert_eq!(
            block_on(async { take_iter.try_next().await })?
                .expect("Expected a chunk")
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            vec![5999999]
        );

        Ok(())
    }
}
