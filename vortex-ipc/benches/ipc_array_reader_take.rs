#![allow(clippy::unwrap_used)]
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::async_executor::FuturesExecutor;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use futures_executor::block_on;
use futures_util::{pin_mut, TryStreamExt};
use itertools::Itertools;
use vortex_array::array::{ChunkedArray, PrimitiveArray};
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_array::{Context, IntoArrayData};
use vortex_io::VortexBufReader;
use vortex_ipc::stream_reader::StreamArrayReader;
use vortex_ipc::stream_writer::StreamArrayWriter;

// 100 record batches, 100k rows each
// take from the first 20 batches and last batch
// compare with arrow
fn ipc_array_reader_take(c: &mut Criterion) {
    let ctx = Arc::new(Context::default());

    let indices = (0..20)
        .map(|i| i * 100_000 + 1)
        .chain([98 * 100_000 + 1])
        .collect_vec();
    let mut group = c.benchmark_group("ipc_array_reader_take");

    group.bench_function("vortex", |b| {
        let array = ChunkedArray::from_iter(
            (0..100i32)
                .map(|i| vec![i; 100_000])
                .map(|vec| PrimitiveArray::from_vec(vec, Validity::AllValid).into_array()),
        )
        .into_array();

        let buffer = block_on(async { StreamArrayWriter::new(vec![]).write_array(array).await })
            .unwrap()
            .into_inner();

        let buffer = Bytes::from(buffer);

        let indices = indices.clone().into_array();

        b.to_async(FuturesExecutor).iter(|| async {
            let stream_reader =
                StreamArrayReader::try_new(VortexBufReader::new(buffer.clone()), ctx.clone())
                    .await
                    .unwrap()
                    .load_dtype()
                    .await
                    .unwrap();
            let stream = stream_reader
                .into_array_stream()
                .take_rows(indices.clone())
                .unwrap();
            pin_mut!(stream);

            while let Some(arr) = stream.try_next().await.unwrap() {
                black_box(arr);
            }
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = ipc_array_reader_take
);
criterion_main!(benches);
