#![no_main]

use bytes::Bytes;
use futures_util::StreamExt;
use libfuzzer_sys::{fuzz_target, Corpus};
use vortex_array::array::ChunkedArray;
use vortex_array::compute::{compare, Operator};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_file::{LayoutDeserializer, VortexFileWriter, VortexReadBuilder};

fuzz_target!(|array_data: ArrayData| -> Corpus {
    if !array_data.dtype().is_struct() {
        return Corpus::Reject;
    }

    let buf = Vec::new();
    let mut writer = VortexFileWriter::new(buf);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async move {
        writer = writer
            .write_array_columns(array_data.clone())
            .await
            .unwrap();
        let written = Bytes::from(writer.finalize().await.unwrap());

        let stream = VortexReadBuilder::new(written, LayoutDeserializer::default())
            .build()
            .await
            .unwrap()
            .into_stream();

        let output = stream.map(|a| a.unwrap()).collect::<Vec<_>>().await;
        let output = ChunkedArray::try_new(output, array_data.dtype().clone())
            .unwrap()
            .into_array();

        assert_eq!(array_data.len(), output.len());

        let cmp_result = compare(&array_data, output, Operator::Eq).unwrap();

        let true_count = cmp_result
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .count_set_bits();

        assert_eq!(true_count, array_data.len())
    });

    Corpus::Keep
});
