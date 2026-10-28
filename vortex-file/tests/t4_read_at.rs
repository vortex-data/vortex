// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Vortex file written to an object store and read back through t4, which caches the store in
//! memory and on a local drive: with Vortex's coalescing of nearby reads and without it, the first
//! scan fetching from the store and the second reading t4's tiers.

#![cfg(all(feature = "t4", unix))]
#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;

use futures::TryStreamExt;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::PutPayload;
use object_store::memory::InMemory;
use object_store::path::Path;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::CoalesceConfig;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::t4::T4Caching;
use vortex_io::t4::T4ReadAt;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

const MIB: u64 = 1 << 20;
const ROWS: usize = 262_144;

/// Two chunks of a column that compresses well and one that barely does, so that the file spans
/// several of t4's 1 MiB blocks.
fn test_array() -> ArrayRef {
    let cycle = PrimitiveArray::from_iter((0..1024).cycle().take(ROWS)).into_array();
    let scrambled = PrimitiveArray::from_iter(
        (0..ROWS as u64).map(|row| row.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
    .into_array();
    let chunk = StructArray::new(
        FieldNames::from(["cycle", "scrambled"]),
        vec![cycle, scrambled],
        ROWS,
        Validity::NonNullable,
    )
    .into_array();
    ChunkedArray::from_iter([chunk.clone(), chunk]).into_array()
}

#[rstest]
#[case::coalesced(Some(CoalesceConfig::object_storage()))]
#[case::each_read_as_asked(None)]
#[tokio::test]
async fn roundtrip_through_t4(#[case] coalesce_config: Option<CoalesceConfig>) -> VortexResult<()> {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(TokioRuntime::current());
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    let expected = test_array();
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, expected.to_array_stream())
        .await?;
    let written = buffer.freeze().into_bytes();
    assert!(
        written.len() as u64 > 2 * MIB,
        "the file spans several of t4's blocks"
    );

    let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let path = Path::from("data/test.vortex");
    store.put(&path, PutPayload::from(written)).await?;
    let dir = tempfile::tempdir()?;
    let cache = T4Caching::builder(store)
        .with_memory_capacity(4 * MIB)
        .with_disk_file(dir.path().join("t4.cache"), 8 * MIB)
        .build()
        .map_err(|error| vortex_err!("{error}"))?;
    let reader = Arc::new(
        T4ReadAt::new(cache.open(&path).await?, session.handle())
            .with_coalesce_config(coalesce_config),
    );

    for _ in 0..2 {
        let chunks = session
            .open_options()
            .open_read(Arc::clone(&reader))
            .await?
            .scan()?
            .into_array_stream()?
            .try_collect::<Vec<_>>()
            .await?;
        let result = ChunkedArray::try_new(chunks, expected.dtype().clone())?;
        assert_arrays_eq!(result, expected, &mut session.create_execution_ctx());
    }

    Ok(())
}
