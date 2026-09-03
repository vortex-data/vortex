// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The write and read paths spawn background tasks — the layout writer and the segment-source I/O
//! driver — that only make progress once the caller drives them concurrently. This exercises both
//! on `SingleThreadRuntime`, the runtime used where no thread pool or JavaScript event loop is
//! available, such as WebAssembly.

#![expect(clippy::tests_outside_test_module)]

use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;

mod common;

use common::enable_all_registered_array_encodings;

const ROWS: usize = 16_384;

fn test_array() -> ArrayRef {
    let nums = PrimitiveArray::from_iter((0..1024).cycle().take(ROWS)).into_array();
    let chunk = StructArray::new(
        FieldNames::from(["a", "b"]),
        vec![nums.clone(), nums],
        ROWS,
        Validity::NonNullable,
    )
    .into_array();
    ChunkedArray::from_iter([chunk.clone(), chunk]).into_array()
}

#[test]
fn roundtrip_on_single_thread_runtime() -> VortexResult<()> {
    let runtime = SingleThreadRuntime::default();

    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_handle(runtime.handle());
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    let buffer = runtime.block_on(async {
        let mut buffer = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut buffer, test_array().to_array_stream())
            .await?;
        Ok::<_, VortexError>(buffer.freeze())
    })?;

    // `open_read` drives reads through a spawned I/O driver, unlike the in-memory `open_buffer`.
    let chunks = runtime.block_on(async {
        session
            .open_options()
            .open_read(buffer)
            .await?
            .scan()?
            .into_array_stream()?
            .try_collect::<Vec<_>>()
            .await
    })?;

    assert_eq!(
        chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
        2 * ROWS
    );

    Ok(())
}
