// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_dtype::FieldNames;
use vortex_dtype::field_path;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&mut session);

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
    let default_strategy = Arc::new(CompressingStrategy::new_btrblocks(
        FlatLayoutStrategy::default(),
        false,
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
