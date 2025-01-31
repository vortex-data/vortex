//! Benchmark for the `bytes_at` operation on a VarBinView.
//! This measures the performance of accessing an individual byte-slice in a VarBinViewArray.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use parquet::data_type::AsBytes;
use vortex::array::{VarBinArray, VarBinViewArray};
use vortex::buffer::{buffer, ByteBuffer};
use vortex::dtype::{DType, Nullability};
use vortex::ipc::iterator::{ArrayIteratorIPC, SyncIPCReader};
use vortex::iter::ArrayIteratorExt;
use vortex::validity::Validity;
use vortex::{Context, IntoArray, IntoArrayVariant};

fn array_data_fixture() -> VarBinArray {
    VarBinArray::try_new(
        buffer![0i32, 5i32, 10i32, 15i32, 20i32].into_array(),
        ByteBuffer::copy_from(b"helloworldhelloworld".as_bytes()),
        DType::Utf8(Nullability::NonNullable),
        Validity::NonNullable,
    )
    .unwrap()
}

fn array_view_fixture() -> VarBinViewArray {
    let array_data = array_data_fixture();

    let buffer = array_data
        .into_array()
        .into_array_iterator()
        .write_ipc(vec![])
        .unwrap();

    SyncIPCReader::try_new(buffer.as_slice(), Arc::new(Context::default()))
        .unwrap()
        .into_array_data()
        .unwrap()
        .into_varbinview()
        .unwrap()
}

fn benchmark(c: &mut Criterion) {
    c.bench_with_input(
        BenchmarkId::new("bytes_at", "array_data"),
        &array_data_fixture(),
        |b, array| {
            b.iter(|| array.bytes_at(3));
        },
    );

    c.bench_with_input(
        BenchmarkId::new("bytes_at", "array_view"),
        &array_view_fixture(),
        |b, array| {
            b.iter(|| array.bytes_at(3));
        },
    );
}

criterion_group!(bench_bytes_at, benchmark);
criterion_main!(bench_bytes_at);
