// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end test of decoding a *nullable* column through a WASM kernel.
//!
//! A nullable canonical primitive is a values buffer plus a validity bitmap. The encoder splits a
//! nullable `i32` column into (a) a non-nullable values child and (b) a validity bitmap stored in
//! the payload. The kernel reassembles them into a nullable primitive `CanonicalMessage` (values
//! in buffer 0, bitmap in buffer 1, `validity = Bitmap`), which the host turns back into a
//! `Validity::Array`.

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
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

/// Nullable passthrough kernel for `i32`.
///
/// Payload: `[u32 len][validity bitmap]` (`ceil(len/8)` bitmap bytes, LSB-first, 1 = valid).
/// Child 0: the `i32` values. Output: a nullable `i32` (values + bitmap).
const NULLABLE_KERNEL_WAT: &str = r#"
(module
  (import "vortex_host" "vx_decode_child" (func $decode_child (param i32 i32) (result i32)))
  (memory (export "memory") 64)
  (global $next (mut i32) (i32.const 1024))
  (func $alloc (export "vx_alloc") (param $len i32) (result i32)
    (local $p i32)
    (local.set $p (global.get $next))
    (global.set $next (i32.add (global.get $next) (local.get $len)))
    (local.get $p))
  (func (export "vx_decode") (param $in_ptr i32) (param $in_len i32) (result i32)
    (local $n i32) (local $bm_len i32)
    (local $out8 i32) (local $child_off i32) (local $vdata i32)
    (local $total i32) (local $res i32) (local $msg i32) (local $buf0 i32)
    (local $buf1hdr i32) (local $buf1 i32)
    (local.set $n (i32.load (local.get $in_ptr)))
    (local.set $bm_len (i32.shr_u (i32.add (local.get $n) (i32.const 7)) (i32.const 3)))
    ;; decode child 0 (the i32 values)
    (local.set $out8 (call $alloc (i32.const 8)))
    (drop (call $decode_child (i32.const 0) (local.get $out8)))
    (local.set $child_off (i32.load (local.get $out8)))
    (local.set $vdata (i32.add (local.get $child_off) (i32.const 36)))
    ;; result = [u32 len][20 hdr][16 buf0 hdr][n*4 values][16 buf1 hdr][bm_len bitmap]
    (local.set $total
      (i32.add (i32.add (i32.const 56) (i32.mul (local.get $n) (i32.const 4)))
               (i32.add (i32.const 16) (local.get $bm_len))))
    (local.set $res (call $alloc (local.get $total)))
    (local.set $msg (i32.add (local.get $res) (i32.const 4)))
    (local.set $buf0 (i32.add (local.get $msg) (i32.const 36)))
    (i32.store (local.get $res) (i32.sub (local.get $total) (i32.const 4)))
    ;; message header: kind=2 (Primitive), ptype=6 (I32), validity=3 (Bitmap), nbuffers=2
    (i32.store8 (local.get $msg) (i32.const 2))
    (i32.store8 offset=1 (local.get $msg) (i32.const 6))
    (i32.store8 offset=2 (local.get $msg) (i32.const 3))
    (i32.store8 offset=3 (local.get $msg) (i32.const 0))
    (i64.store offset=4 (local.get $msg) (i64.extend_i32_u (local.get $n)))
    (i32.store offset=12 (local.get $msg) (i32.const 2))
    (i32.store offset=16 (local.get $msg) (i32.const 0))
    ;; buffer 0 (values): len = n*4, alignment_exponent = 2
    (i64.store offset=20 (local.get $msg) (i64.extend_i32_u (i32.mul (local.get $n) (i32.const 4))))
    (i32.store8 offset=28 (local.get $msg) (i32.const 2))
    (memory.copy (local.get $buf0) (local.get $vdata) (i32.mul (local.get $n) (i32.const 4)))
    ;; buffer 1 (validity bitmap): len = bm_len, alignment_exponent = 0
    (local.set $buf1hdr (i32.add (local.get $buf0) (i32.mul (local.get $n) (i32.const 4))))
    (i64.store (local.get $buf1hdr) (i64.extend_i32_u (local.get $bm_len)))
    (i32.store8 offset=8 (local.get $buf1hdr) (i32.const 0))
    (local.set $buf1 (i32.add (local.get $buf1hdr) (i32.const 16)))
    (memory.copy (local.get $buf1) (i32.add (local.get $in_ptr) (i32.const 4)) (local.get $bm_len))
    (local.get $res)))
"#;

/// Host encoder: split a nullable i32 column into a non-nullable values child and a payload
/// carrying the length and validity bitmap.
struct NullableEncoder;

impl WasmEncoder for NullableEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!("NullableEncoder example only supports i32");
        }
        let n = primitive.len();
        let values = primitive.buffer_handle().to_host_sync();

        // Materialize the validity bitmap (offset 0, 1 = valid).
        let bits = primitive.validity()?.execute_mask(n, ctx)?.to_bit_buffer();
        let bitmap = &bits.inner().as_slice()[..n.div_ceil(8)];

        let mut payload = Vec::with_capacity(4 + bitmap.len());
        payload.extend_from_slice(&(n as u32).to_le_bytes());
        payload.extend_from_slice(bitmap);

        let child = PrimitiveArray::from_byte_buffer(values, PType::I32, Validity::NonNullable)
            .into_array();
        Ok(WasmEncoded {
            payload: ByteBuffer::from(payload),
            child,
        })
    }
}

#[test]
fn nullable_column_round_trips() {
    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);

        // Positions 1 and 3 are null.
        let validity = Validity::from_iter([true, false, true, false, true]);
        let array = PrimitiveArray::new(Buffer::copy_from(&[100i32, 200, 300, 400, 500]), validity)
            .into_array();
        let dtype = array.dtype().clone();

        let kernel = ByteBuffer::from(wat::parse_str(NULLABLE_KERNEL_WAT).expect("valid wat"));
        let strategy = WasmLayoutStrategy::new(
            kernel,
            "test.nullable",
            Arc::new(NullableEncoder),
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
        assert_eq!(row_count, 5);

        let reader = layout
            .new_reader("nullable".into(), segments, &session, &Default::default())
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

        assert_eq!(out.len(), 5);

        // Validity round-trips.
        let mut ctx = session.create_execution_ctx();
        let bits = out
            .validity()
            .unwrap()
            .execute_mask(5, &mut ctx)
            .unwrap()
            .to_bit_buffer();
        let valid: Vec<bool> = (0..5).map(|i| bits.value(i)).collect();
        assert_eq!(valid, vec![true, false, true, false, true]);

        // Values at all positions survive (including the null slots).
        let expected: Vec<u8> = [100i32, 200, 300, 400, 500]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
    });
}
