// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::multi::MultiFileSession;
use vortex_file::multi::open_cached;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

#[tokio::test]
async fn open_cached_rereads_rewritten_file_without_version() -> VortexResult<()> {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with::<MultiFileSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    let mut before = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut before, buffer![1u32].into_array().to_array_stream())
        .await?;
    let before = ByteBuffer::from(before);
    let file = open_cached(
        &session,
        Some("rewritten.vortex"),
        Arc::new(before.clone()),
        Some(before.len() as u64),
        &|options| options,
    )
    .await?;
    assert_eq!(file.row_count(), 1);

    let mut after = ByteBufferMut::empty();
    session
        .write_options()
        .write(
            &mut after,
            buffer![2u32, 3u32].into_array().to_array_stream(),
        )
        .await?;
    let after = ByteBuffer::from(after);
    let file = open_cached(
        &session,
        Some("rewritten.vortex"),
        Arc::new(after.clone()),
        Some(after.len() as u64),
        &|options| options,
    )
    .await?;
    assert_eq!(file.row_count(), 2);
    Ok(())
}
