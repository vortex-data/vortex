// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::field_path;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::ByteBuffer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);

    session
});

#[tokio::test]
async fn test_file_roundtrip() {
    // Create a simple roundtrip
    let nums = PrimitiveArray::from_iter((0..1024).cycle().take(16_384)).into_array();

    let a_array = StructArray::new(
        FieldNames::from(["raw", "compressed"]),
        vec![nums.clone(), nums.clone()],
        16_384,
        Validity::NonNullable,
    )
    .into_array();

    let b_array = PrimitiveArray::from_iter((1024..2048).cycle().take(16_384)).into_array();

    let data = StructArray::new(
        FieldNames::from(["a", "b"]),
        vec![a_array, b_array],
        16_384,
        Validity::NonNullable,
    )
    .into_array();

    // Create a writer which by default uses the BtrBlocks compressor for a.compressed, but leaves
    // the b and the a.raw columns uncompressed.
    let default_strategy = Arc::new(CompressingStrategy::new(
        FlatLayoutStrategy::default(),
        BtrBlocksCompressor::default(),
    ));

    let writer = Arc::new(
        TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), default_strategy)
            .with_field_writer(field_path!(a.raw), Arc::new(FlatLayoutStrategy::default()))
            .with_field_writer(field_path!(b), Arc::new(FlatLayoutStrategy::default())),
    );

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(writer)
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    // Read the data back
    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");

    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        let next = next.expect("next");
        let next = next.to_struct();
        let a = next.unmasked_field_by_name("a").unwrap().to_struct();
        let b = next.unmasked_field_by_name("b").unwrap();

        let raw = a.unmasked_field_by_name("raw").unwrap();
        let compressed = a.unmasked_field_by_name("compressed").unwrap();

        assert!(raw.is_canonical());
        assert!(!compressed.is_canonical());

        assert!(b.is_canonical());
        assert!(raw.nbytes() > compressed.nbytes());
    }
}

/// Regression test: writing a Dict<ListView> where the list has
/// Validity::Array(BoolArray) and the dict codes are nullable used to fail
/// with "Array vortex.fill_null does not support serialization".
#[tokio::test]
async fn test_dict_listview_validity_roundtrip() {
    let elements = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5]).into_array();
    let offsets = PrimitiveArray::from_iter(vec![0u32, 2, 4]).into_array();
    let sizes = PrimitiveArray::from_iter(vec![2u32, 2, 1]).into_array();
    let list_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
    let listview = ListViewArray::new(elements, offsets, sizes, list_validity).into_array();

    let codes = PrimitiveArray::new(
        vortex_buffer::buffer![0u32, 0, 1, 0, 2],
        Validity::from_iter(vec![true, false, true, true, true]),
    )
    .into_array();

    let dict = DictArray::new(codes, listview).into_array();

    let data = StructArray::from_fields(&[("col", dict)])
        .expect("from_fields")
        .into_array();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await
        .expect("write should not fail with fill_null serialization error");

    let bytes = ByteBuffer::from(bytes);
    let vxf = SESSION.open_options().open_buffer(bytes).expect("open");

    let stream = vxf
        .scan()
        .expect("scan")
        .into_stream()
        .expect("into_stream");
    pin_mut!(stream);

    let chunk = stream
        .next()
        .await
        .unwrap()
        .expect("read back should succeed");
    vortex_array::assert_arrays_eq!(data, chunk);
    assert!(stream.next().await.is_none(), "expected a single chunk");
}
