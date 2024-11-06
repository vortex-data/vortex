#![doc(html_logo_url = "/vortex/docs/_static/vortex_spiral_logo.svg")]

use message_reader::*;
use message_writer::*;

pub mod chunked_reader;
mod dtype_reader;
pub mod io;
pub mod layouts;
mod message_reader;
mod message_writer;
mod messages;
pub mod stream_reader;
pub mod stream_writer;
pub use dtype_reader::*;

pub const ALIGNMENT: usize = 64;

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod test {
    use std::sync::Arc;

    use futures_executor::block_on;
    use futures_util::io::Cursor;
    use futures_util::{pin_mut, StreamExt, TryStreamExt};
    use itertools::Itertools;
    use vortex_array::array::{ChunkedArray, PrimitiveArray, PrimitiveEncoding};
    use vortex_array::encoding::ArrayEncoding;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::{ArrayDType, Context, IntoArray};
    use vortex_error::VortexResult;

    use crate::io::FuturesAdapter;
    use crate::stream_reader::StreamArrayReader;
    use crate::stream_writer::StreamArrayWriter;

    fn write_ipc<A: IntoArray>(array: A) -> Vec<u8> {
        block_on(async {
            StreamArrayWriter::new(vec![])
                .write_array(array.into_array())
                .await
                .unwrap()
                .into_inner()
        })
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_empty_index() -> VortexResult<()> {
        let data = PrimitiveArray::from((0i32..3_000_000).collect_vec());
        let buffer = write_ipc(data);

        let indices = PrimitiveArray::from(vec![1, 2, 10]).into_array();

        let ctx = Arc::new(Context::default());
        let stream_reader = block_on(async {
            StreamArrayReader::try_new(FuturesAdapter(Cursor::new(buffer)), ctx)
                .await
                .unwrap()
                .load_dtype()
                .await
                .unwrap()
        });
        let reader = stream_reader.into_array_stream();

        let result_iter = reader.take_rows(indices)?;
        pin_mut!(result_iter);

        let _result = block_on(async { result_iter.next().await.unwrap().unwrap() });
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_write_read_chunked() -> VortexResult<()> {
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

        let ctx = Arc::new(Context::default());
        let stream_reader = block_on(async {
            StreamArrayReader::try_new(FuturesAdapter(Cursor::new(buffer)), ctx)
                .await
                .unwrap()
                .load_dtype()
                .await
                .unwrap()
        });

        let take_iter = stream_reader.into_array_stream().take_rows(indices)?;
        pin_mut!(take_iter);

        let next = block_on(async { take_iter.try_next().await })?.expect("Expected a chunk");
        assert_eq!(next.encoding().id(), PrimitiveEncoding.id());

        assert_eq!(
            next.as_primitive().maybe_null_slice::<i32>(),
            vec![2999989, 2999988, 2999987, 2999986, 2899999, 0, 0]
        );
        assert_eq!(
            block_on(async { take_iter.try_next().await })?
                .expect("Expected a chunk")
                .as_primitive()
                .maybe_null_slice::<i32>(),
            vec![5999999]
        );

        Ok(())
    }
}
