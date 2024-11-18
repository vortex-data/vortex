use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::Int32Type;
use arrow_array::PrimitiveArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::{ArrayData, Context, IntoCanonical};
use vortex_buffer::Buffer;
use vortex_io::VortexBufReader;

use crate::stream_reader::StreamArrayReader;
use crate::stream_writer::StreamArrayWriter;

#[tokio::test]
async fn broken_data() {
    let arrow_arr: PrimitiveArray<Int32Type> = [Some(1), Some(2), Some(3), None].iter().collect();
    let vortex_arr = ArrayData::from_arrow(&arrow_arr, true);
    let written = StreamArrayWriter::new(Vec::new())
        .write_array(vortex_arr)
        .await
        .unwrap()
        .into_inner();
    let written = Buffer::from(written);
    let reader =
        StreamArrayReader::try_new(VortexBufReader::new(written), Arc::new(Context::default()))
            .await
            .unwrap();
    let arr = reader
        .load_dtype()
        .await
        .unwrap()
        .into_array_stream()
        .collect_chunked()
        .await
        .unwrap();
    let round_tripped = arr.into_canonical().unwrap().into_arrow().unwrap();
    assert_eq!(&arrow_arr, round_tripped.as_primitive::<Int32Type>());
}
