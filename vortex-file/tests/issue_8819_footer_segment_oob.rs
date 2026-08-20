// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression test for [issue #8819]: opening a Vortex file whose footer segment map declares a
//! segment extending past the end of the file must return a [`vortex_error::VortexError`] rather
//! than panicking while slicing the backing buffer during file open (the reported repro dies
//! while opening the file, before any array decode).
//!
//! [issue #8819]: https://github.com/vortex-data/vortex/issues/8819

#![expect(clippy::tests_outside_test_module)]

use std::mem::size_of;
use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    session
});

#[tokio::test]
async fn open_buffer_rejects_out_of_bounds_footer_segment() {
    // Write a valid file to obtain a well-formed footer.
    let mut buf = ByteBufferMut::empty();
    let array = Buffer::from((0i32..256).collect::<Vec<i32>>()).into_array();
    SESSION
        .write_options()
        .write(&mut buf, array.to_array_stream())
        .await
        .expect("write");
    let valid = ByteBuffer::from(buf);

    // Open the valid file to read its real segment map. We pick the longest segment so its
    // `(offset, length)` byte pattern is unlikely to collide with the file's data below.
    let file = SESSION
        .open_options()
        .open_buffer(valid.clone())
        .expect("valid file opens");
    let target = *file
        .footer()
        .segment_map()
        .iter()
        .max_by_key(|segment| segment.length)
        .expect("file must contain at least one segment");

    // Rewrite the segment's declared length in the footer flatbuffer so it extends past the end of
    // the file. A `SegmentSpec` is stored as a FlatBuffer struct with the `u64` offset immediately
    // followed by the `u32` length.
    let mut bytes = valid.as_slice().to_vec();
    let mut pattern = target.offset.to_le_bytes().to_vec();
    pattern.extend_from_slice(&target.length.to_le_bytes());
    let positions = bytes
        .windows(pattern.len())
        .enumerate()
        .filter_map(|(i, window)| (window == pattern.as_slice()).then_some(i))
        .collect::<Vec<_>>();
    assert_eq!(
        positions.len(),
        1,
        "expected a uniquely locatable segment spec"
    );
    let length_offset = positions[0] + size_of::<u64>();
    bytes[length_offset..length_offset + size_of::<u32>()].copy_from_slice(&u32::MAX.to_le_bytes());

    // Opening the corrupted file must return an error rather than panicking while slicing.
    match SESSION.open_options().open_buffer(ByteBuffer::from(bytes)) {
        Ok(_) => panic!("open_buffer must reject an out-of-bounds footer segment"),
        Err(err) => assert!(
            err.to_string().contains("out of bounds") || err.to_string().contains("past the end"),
            "unexpected error: {err}"
        ),
    }
}
