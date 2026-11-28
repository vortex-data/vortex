// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression test for [issue #12020]: a scan must not allocate in proportion to the row count
//! that a file declares.
//!
//! A non-nullable struct with no fields is written as a layout with no children and no segments,
//! so nothing in the file contradicts its row count. One corrupted byte makes that count huge,
//! and the file is still valid: it holds that many empty rows.
//!
//! [issue #12020]: https://github.com/vortex-data/vortex/issues/12020

use std::mem::size_of;
use std::sync::LazyLock;

use futures::StreamExt;
use futures::TryStreamExt;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::scan::split_by::DEFAULT_MAX_SPLIT_ROWS;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    session
});

/// The row count of the valid file. Its bytes are unlikely to appear anywhere else in the file.
const ROWS: u64 = 0x1234_5678;

#[rstest]
// The corruption that the fuzzer found: about 850 billion rows, 8.5 million default splits.
#[case::issue_12020(4, 0xC6)]
// The largest row counts a single byte can produce.
#[case::top_byte(7, 0xFF)]
#[tokio::test]
async fn scan_of_corrupt_zero_field_row_count_is_lazy(
    #[case] byte: usize,
    #[case] flip: u8,
) -> VortexResult<()> {
    let array = StructArray::try_new(
        FieldNames::empty(),
        vec![],
        usize::try_from(ROWS)?,
        Validity::NonNullable,
    )?
    .into_array();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await?;

    let mut bytes = ByteBuffer::from(buf).as_slice().to_vec();
    let pattern = ROWS.to_le_bytes();
    let positions = bytes
        .windows(size_of::<u64>())
        .enumerate()
        .filter_map(|(i, window)| (window == pattern).then_some(i))
        .collect::<Vec<_>>();
    assert_eq!(positions.len(), 1, "expected one row count in the file");
    bytes[positions[0] + byte] ^= flip;

    let file = SESSION
        .open_options()
        .open_buffer(ByteBuffer::from(bytes))?;
    let corrupt_rows = ROWS ^ (u64::from(flip) << (8 * byte));
    assert_eq!(file.row_count(), corrupt_rows);

    // The scan streams its splits, so the first ones arrive without preparing the rest.
    let batches: Vec<ArrayRef> = file.scan()?.into_stream()?.take(3).try_collect().await?;
    assert_eq!(batches.len(), 3);
    for batch in batches {
        assert_eq!(batch.dtype(), array.dtype());
        assert!(batch.len() as u64 <= DEFAULT_MAX_SPLIT_ROWS);
    }
    Ok(())
}
