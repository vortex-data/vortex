// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end test of the `WasmLayout` write + read path against real layout machinery.
//!
//! An identity kernel reconstructs its output by asking the host for child 0 — the data layout
//! written by a `FlatLayoutStrategy`. Reading back through `WasmReader` therefore yields the
//! original array, exercising: writing the kernel at end-of-file, the child layout, eager child
//! decode, the `vx_decode_child` host import, and the `CanonicalMessage` round trip.

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_wasm::WasmLayoutStrategy;

/// Identity kernel: returns child 0 verbatim.
const IDENTITY_KERNEL_WAT: &str = r#"
(module
  (import "vortex_host" "vx_decode_child" (func $decode_child (param i32 i32) (result i32)))
  (memory (export "memory") 4)
  (global $next (mut i32) (i32.const 1024))
  (func $alloc (export "vx_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $next))
    (global.set $next (i32.add (global.get $next) (local.get $len)))
    (local.get $p))
  (func (export "vx_decode") (param $in_ptr i32) (param $in_len i32) (result i32)
    (local $out i32) (local $child_off i32) (local $child_len i32) (local $res i32)
    (local.set $out (call $alloc (i32.const 8)))
    (drop (call $decode_child (i32.const 0) (local.get $out)))
    (local.set $child_off (i32.load (local.get $out)))
    (local.set $child_len (i32.load offset=4 (local.get $out)))
    (local.set $res (call $alloc (i32.add (i32.const 4) (local.get $child_len))))
    (i32.store (local.get $res) (local.get $child_len))
    (memory.copy (i32.add (local.get $res) (i32.const 4)) (local.get $child_off) (local.get $child_len))
    (local.get $res)))
"#;

#[test]
fn wasm_layout_round_trips_through_identity_kernel() {
    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);

        let array = PrimitiveArray::new(buffer![7i32, 11, 13, 17, 19, 23], Validity::NonNullable)
            .into_array();
        let dtype = array.dtype().clone();
        let expected: Vec<u8> = [7i32, 11, 13, 17, 19, 23]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let kernel = ByteBuffer::from(wat::parse_str(IDENTITY_KERNEL_WAT).expect("valid wat"));
        let strategy = WasmLayoutStrategy::new(
            kernel,
            "test.identity",
            Arc::new(FlatLayoutStrategy::default()) as Arc<dyn LayoutStrategy>,
        );

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let layout = strategy
            .write_stream(
                ArrayContext::empty(),
                Arc::<TestSegments>::clone(&segments),
                SequentialStreamAdapter::new(dtype, array.to_array_stream().sequenced(ptr))
                    .sendable(),
                eof,
                &session,
            )
            .await
            .expect("write");

        assert_eq!(layout.row_count(), 6);

        let row_count = layout.row_count();
        let reader = layout
            .new_reader("wasm".into(), segments, &session, &Default::default())
            .expect("reader");
        let out = reader
            .projection_evaluation(
                &(0..row_count),
                &root(),
                MaskFuture::new_true(row_count as usize),
            )
            .expect("projection")
            .await
            .expect("decode");

        assert_eq!(out.len(), 6);
        assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
    });
}
