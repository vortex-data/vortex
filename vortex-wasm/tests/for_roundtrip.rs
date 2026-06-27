// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end test of a real (non-identity) encoding: Frame-of-Reference (FoR).
//!
//! Write side ([`ForEncoder`]): pick a reference (the minimum), store it in the payload, and store
//! `value - reference` as the child input. Read side (the FoR kernel): read the reference from the
//! payload, decode the child deltas via `vx_decode_child`, and return `reference + delta`.
//!
//! This exercises the payload segment, a transformed (genuinely smaller-valued) child, and a
//! kernel that does real arithmetic — not just a passthrough.

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
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
use vortex_wasm::WasmEncoded;
use vortex_wasm::WasmEncoder;
use vortex_wasm::WasmLayoutStrategy;

/// Frame-of-Reference kernel for `i32`.
///
/// Payload layout: `[i32 reference]` (4 bytes). Output[i] = reference + delta[i].
const FOR_KERNEL_WAT: &str = r#"
(module
  (import "vortex_host" "vx_decode_child" (func $decode_child (param i32 i32) (result i32)))
  (memory (export "memory") 16)
  (global $next (mut i32) (i32.const 1024))
  (func $alloc (export "vx_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $next))
    (global.set $next (i32.add (global.get $next) (local.get $len)))
    (local.get $p))
  (func (export "vx_decode") (param $in_ptr i32) (param $in_len i32) (result i32)
    (local $ref i32)
    (local $out8 i32)
    (local $child_off i32)
    (local $n i32)
    (local $data_off i32)
    (local $total i32)
    (local $res i32)
    (local $msg i32)
    (local $buf i32)
    (local $i i32)
    ;; reference is the first 4 bytes of the payload
    (local.set $ref (i32.load (local.get $in_ptr)))
    ;; decode child 0 (the deltas) into guest memory
    (local.set $out8 (call $alloc (i32.const 8)))
    (drop (call $decode_child (i32.const 0) (local.get $out8)))
    (local.set $child_off (i32.load (local.get $out8)))
    ;; element count = length field (u64 low word) at child_off + 4
    (local.set $n (i32.load offset=4 (local.get $child_off)))
    ;; child buffer data begins after the 20-byte message header + 16-byte buffer entry header
    (local.set $data_off (i32.add (local.get $child_off) (i32.const 36)))
    ;; result = [u32 len][20-byte msg header][16-byte buffer header][n*4 data]
    (local.set $total (i32.add (i32.const 40) (i32.mul (local.get $n) (i32.const 4))))
    (local.set $res (call $alloc (local.get $total)))
    (local.set $msg (i32.add (local.get $res) (i32.const 4)))
    (local.set $buf (i32.add (local.get $msg) (i32.const 36)))
    (i32.store (local.get $res) (i32.sub (local.get $total) (i32.const 4)))
    ;; message header: kind=2 (Primitive), ptype=6 (I32), validity=0
    (i32.store8 (local.get $msg) (i32.const 2))
    (i32.store8 offset=1 (local.get $msg) (i32.const 6))
    (i32.store8 offset=2 (local.get $msg) (i32.const 0))
    (i32.store8 offset=3 (local.get $msg) (i32.const 0))
    (i64.store offset=4 (local.get $msg) (i64.extend_i32_u (local.get $n)))
    (i32.store offset=12 (local.get $msg) (i32.const 1))
    (i32.store offset=16 (local.get $msg) (i32.const 0))
    ;; buffer entry header: len = n*4, alignment_exponent = 2
    (i64.store offset=20 (local.get $msg) (i64.extend_i32_u (i32.mul (local.get $n) (i32.const 4))))
    (i32.store8 offset=28 (local.get $msg) (i32.const 2))
    ;; for each element: buf[i] = reference + delta[i]
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (i32.store
          (i32.add (local.get $buf) (i32.mul (local.get $i) (i32.const 4)))
          (i32.add
            (local.get $ref)
            (i32.load (i32.add (local.get $data_off) (i32.mul (local.get $i) (i32.const 4))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
    (local.get $res)))
"#;

/// Host-side Frame-of-Reference encoder for `i32`.
struct ForEncoder;

impl WasmEncoder for ForEncoder {
    fn encode(
        &self,
        chunk: vortex_array::ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!(
                "ForEncoder example only supports i32, got {}",
                primitive.ptype()
            );
        }
        let values = primitive.as_slice::<i32>();
        let reference = values.iter().copied().min().unwrap_or(0);
        let deltas: Vec<i32> = values.iter().map(|v| v - reference).collect();

        let payload = ByteBuffer::from(reference.to_le_bytes().to_vec());
        let child =
            PrimitiveArray::new(Buffer::copy_from(&deltas), Validity::NonNullable).into_array();
        Ok(WasmEncoded { payload, child })
    }
}

#[test]
fn for_encoding_round_trips() {
    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);

        let values = vec![1000i32, 1005, 1002, 1010, 1001, 1000, 1234];
        let array =
            PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
        let dtype = array.dtype().clone();
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let kernel = ByteBuffer::from(wat::parse_str(FOR_KERNEL_WAT).expect("valid wat"));
        let strategy = WasmLayoutStrategy::new(
            kernel,
            "test.for",
            Arc::new(ForEncoder),
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

        let row_count = layout.row_count();
        assert_eq!(row_count, values.len() as u64);

        let reader = layout
            .new_reader("for".into(), segments, &session, &Default::default())
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

        assert_eq!(out.len(), values.len());
        assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
    });
}
