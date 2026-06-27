// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end test of Frame-of-Reference + bit packing — a composed real encoding with genuine
//! on-disk size reduction.
//!
//! Write side ([`ForBitpackEncoder`]): reference = minimum, deltas = `value - reference` packed
//! into the minimum number of bits, stored as a `u8` child. The reference, bit width, and length
//! live in the payload. Read side (the WAT kernel below): unpack the deltas and add the reference.
//!
//! The packed child is asserted to be materially smaller than the raw column.

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
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
use vortex_wasm_guest::bitpack;

/// FoR + bit-packing kernel for `i32`.
///
/// Payload: `[i32 reference][u8 bit_width][u32 len]`. Child 0: `u8` LSB-first packed deltas.
const FOR_BITPACK_KERNEL_WAT: &str = r#"
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
    (local $ref i32) (local $bw i32) (local $n i32)
    (local $out8 i32) (local $child_off i32) (local $data_off i32)
    (local $total i32) (local $res i32) (local $msg i32) (local $buf i32)
    (local $i i32) (local $b i32) (local $bitpos i32) (local $v i32) (local $bit i32)
    ;; payload = [i32 ref][u8 bw][u32 n]
    (local.set $ref (i32.load (local.get $in_ptr)))
    (local.set $bw (i32.load8_u offset=4 (local.get $in_ptr)))
    (local.set $n (i32.load offset=5 (local.get $in_ptr)))
    ;; decode child 0 (packed u8 deltas)
    (local.set $out8 (call $alloc (i32.const 8)))
    (drop (call $decode_child (i32.const 0) (local.get $out8)))
    (local.set $child_off (i32.load (local.get $out8)))
    (local.set $data_off (i32.add (local.get $child_off) (i32.const 36)))
    ;; result = [u32 len][20 hdr][16 buf hdr][n*4 data]
    (local.set $total (i32.add (i32.const 40) (i32.mul (local.get $n) (i32.const 4))))
    (local.set $res (call $alloc (local.get $total)))
    (local.set $msg (i32.add (local.get $res) (i32.const 4)))
    (local.set $buf (i32.add (local.get $msg) (i32.const 36)))
    (i32.store (local.get $res) (i32.sub (local.get $total) (i32.const 4)))
    (i32.store8 (local.get $msg) (i32.const 2))
    (i32.store8 offset=1 (local.get $msg) (i32.const 6))
    (i32.store8 offset=2 (local.get $msg) (i32.const 0))
    (i32.store8 offset=3 (local.get $msg) (i32.const 0))
    (i64.store offset=4 (local.get $msg) (i64.extend_i32_u (local.get $n)))
    (i32.store offset=12 (local.get $msg) (i32.const 1))
    (i32.store offset=16 (local.get $msg) (i32.const 0))
    (i64.store offset=20 (local.get $msg) (i64.extend_i32_u (i32.mul (local.get $n) (i32.const 4))))
    (i32.store8 offset=28 (local.get $msg) (i32.const 2))
    ;; unpack: for each element, gather bit_width LSB-first bits
    (local.set $bitpos (i32.const 0))
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $v (i32.const 0))
        (local.set $b (i32.const 0))
        (block $bdone
          (loop $bloop
            (br_if $bdone (i32.ge_u (local.get $b) (local.get $bw)))
            (local.set $bit
              (i32.and
                (i32.shr_u
                  (i32.load8_u
                    (i32.add (local.get $data_off) (i32.shr_u (local.get $bitpos) (i32.const 3))))
                  (i32.and (local.get $bitpos) (i32.const 7)))
                (i32.const 1)))
            (local.set $v (i32.or (local.get $v) (i32.shl (local.get $bit) (local.get $b))))
            (local.set $bitpos (i32.add (local.get $bitpos) (i32.const 1)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $bloop)))
        (i32.store
          (i32.add (local.get $buf) (i32.mul (local.get $i) (i32.const 4)))
          (i32.add (local.get $ref) (local.get $v)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
    (local.get $res)))
"#;

/// Host-side FoR + bit-packing encoder for `i32`.
struct ForBitpackEncoder;

impl ForBitpackEncoder {
    fn encode_i32(values: &[i32]) -> (ByteBuffer, ArrayRef) {
        let reference = values.iter().copied().min().unwrap_or(0);
        let deltas: Vec<u32> = values
            .iter()
            .map(|v| v.wrapping_sub(reference) as u32)
            .collect();
        let bw = bitpack::bit_width(deltas.iter().copied().max().unwrap_or(0));
        let packed = bitpack::pack(&deltas, bw);

        let mut payload = Vec::with_capacity(9);
        payload.extend_from_slice(&reference.to_le_bytes());
        payload.push(bw);
        payload.extend_from_slice(&(values.len() as u32).to_le_bytes());

        let child =
            PrimitiveArray::new(Buffer::copy_from(&packed), Validity::NonNullable).into_array();
        (ByteBuffer::from(payload), child)
    }
}

impl WasmEncoder for ForBitpackEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!("ForBitpackEncoder example only supports i32");
        }
        let (payload, child) = Self::encode_i32(primitive.as_slice::<i32>());
        Ok(WasmEncoded { payload, child })
    }
}

#[test]
fn for_bitpack_encoding_reduces_size() {
    // 1024 values within a 6-bit window of the reference => 6 bits each instead of 32.
    let values: Vec<i32> = (0..1024).map(|i| 10_000 + (i % 64)).collect();
    let (_payload, child) = ForBitpackEncoder::encode_i32(&values);
    let packed_bytes = child.buffers()[0].len();
    let raw_bytes = values.len() * 4;
    assert_eq!(packed_bytes, bitpack::packed_len(values.len(), 6));
    assert!(
        packed_bytes * 4 < raw_bytes,
        "expected >4x reduction: packed={packed_bytes} raw={raw_bytes}"
    );
}

#[test]
fn for_bitpack_round_trips() {
    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);

        let values: Vec<i32> = (0..1024).map(|i| 10_000 + (i % 64)).collect();
        let array =
            PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
        let dtype = array.dtype().clone();
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let kernel = ByteBuffer::from(wat::parse_str(FOR_BITPACK_KERNEL_WAT).expect("valid wat"));
        let strategy = WasmLayoutStrategy::new(
            kernel,
            "test.for-bitpack",
            Arc::new(ForBitpackEncoder),
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
            .new_reader(
                "for-bitpack".into(),
                segments,
                &session,
                &Default::default(),
            )
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
