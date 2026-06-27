// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests of the WASM encoding pipeline using *real* compiled kernels.
//!
//! Each test pairs a host-side [`WasmEncoder`] with a kernel `.wasm` built from the matching
//! `vortex-wasm-guest` example (committed under `tests/fixtures/`, see that directory's `README`).
//! The kernels return their output as Arrow C Data Interface structs, which the host imports back
//! into a Vortex array — exercising the full [`WasmLayoutStrategy`] -> [`WasmReader`] -> Arrow
//! boundary, not just a hand-written WAT stand-in.

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
use vortex_wasm::IdentityEncoder;
use vortex_wasm::WasmEncoded;
use vortex_wasm::WasmEncoder;
use vortex_wasm::WasmLayoutStrategy;
use vortex_wasm_guest::bitpack;

/// The identity kernel: returns child 0 unchanged (`examples/identity-kernel`).
const IDENTITY_KERNEL: &[u8] = include_bytes!("fixtures/identity_kernel.wasm");
/// The Frame-of-Reference kernel for `i32` (`examples/for-kernel`).
const FOR_KERNEL: &[u8] = include_bytes!("fixtures/for_kernel.wasm");
/// The FoR + bit-packing kernel for `i32` (`examples/for-bitpack-kernel`).
const FOR_BITPACK_KERNEL: &[u8] = include_bytes!("fixtures/for_bitpack_kernel.wasm");

/// Write `array` through a [`WasmLayoutStrategy`] with `kernel`/`encoder`, then decode the whole
/// column back through a [`WasmReader`].
fn round_trip(
    kernel: &'static [u8],
    encoding_id: &str,
    encoder: Arc<dyn WasmEncoder>,
    array: ArrayRef,
) -> ArrayRef {
    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);

        let dtype = array.dtype().clone();
        let strategy = WasmLayoutStrategy::new(
            ByteBuffer::from(kernel.to_vec()),
            encoding_id,
            encoder,
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
        let reader = layout
            .new_reader(encoding_id.into(), segments, &session, &Default::default())
            .expect("reader");
        reader
            .projection_evaluation(
                &(0..row_count),
                &root(),
                MaskFuture::new_true(row_count as usize),
            )
            .expect("projection")
            .await
            .expect("decode")
    })
}

/// Collect the validity of `array` as a `Vec<bool>` of length `len`.
fn validity_bools(array: &ArrayRef, len: usize) -> Vec<bool> {
    let mut ctx = array_session().create_execution_ctx();
    let bits = array
        .validity()
        .expect("validity")
        .execute_mask(len, &mut ctx)
        .expect("mask")
        .to_bit_buffer();
    (0..len).map(|i| bits.value(i)).collect()
}

#[test]
fn identity_round_trips_non_nullable() {
    let values = vec![10i32, 20, 30, 40];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let out = round_trip(
        IDENTITY_KERNEL,
        "test.identity",
        Arc::new(IdentityEncoder),
        array,
    );

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

#[test]
fn identity_round_trips_nullable() {
    // Positions 1 and 3 are null. The identity kernel preserves the child's validity bitmap, so the
    // whole nullable column survives the Arrow boundary in both directions.
    let validity = Validity::from_iter([true, false, true, false, true]);
    let values = [100i32, 200, 300, 400, 500];
    let array = PrimitiveArray::new(Buffer::copy_from(values), validity).into_array();
    let out = round_trip(
        IDENTITY_KERNEL,
        "test.identity",
        Arc::new(IdentityEncoder),
        array,
    );

    assert_eq!(out.len(), 5);
    assert_eq!(
        validity_bools(&out, 5),
        vec![true, false, true, false, true]
    );
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

/// Frame-of-Reference encoder for `i32`: store the minimum as the payload reference and the
/// per-element deltas as the child.
struct ForEncoder;

impl WasmEncoder for ForEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!("ForEncoder only supports i32, got {}", primitive.ptype());
        }
        let values = primitive.as_slice::<i32>();
        let reference = values.iter().copied().min().unwrap_or(0);
        let deltas: Vec<i32> = values.iter().map(|v| v - reference).collect();

        let payload = ByteBuffer::from(reference.to_le_bytes().to_vec());
        let child =
            PrimitiveArray::new(Buffer::copy_from(&deltas), Validity::NonNullable).into_array();
        Ok(WasmEncoded {
            payload,
            child: Some(child),
        })
    }
}

#[test]
fn for_round_trips() {
    let values = vec![1000i32, 1005, 1002, 1010, 1001, 1000, 1234];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let out = round_trip(FOR_KERNEL, "test.for", Arc::new(ForEncoder), array);

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

/// FoR + bit-packing encoder for `i32`: the entire encoded form is the opaque payload
/// `[i32 reference][u8 bit_width][u32 len][packed deltas…]`, so there is **no child**.
struct ForBitpackEncoder;

impl ForBitpackEncoder {
    fn encode_i32(values: &[i32]) -> ByteBuffer {
        let reference = values.iter().copied().min().unwrap_or(0);
        let deltas: Vec<u32> = values
            .iter()
            .map(|v| v.wrapping_sub(reference) as u32)
            .collect();
        let bw = bitpack::bit_width(deltas.iter().copied().max().unwrap_or(0));
        let packed = bitpack::pack(&deltas, bw);

        let mut payload = Vec::with_capacity(9 + packed.len());
        payload.extend_from_slice(&reference.to_le_bytes());
        payload.push(bw);
        payload.extend_from_slice(&(values.len() as u32).to_le_bytes());
        payload.extend_from_slice(&packed);
        ByteBuffer::from(payload)
    }
}

impl WasmEncoder for ForBitpackEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!("ForBitpackEncoder only supports i32");
        }
        Ok(WasmEncoded {
            payload: Self::encode_i32(primitive.as_slice::<i32>()),
            child: None,
        })
    }
}

#[test]
fn for_bitpack_reduces_size() {
    // 1024 values within a 6-bit window of the reference => 6 bits each instead of 32.
    let values: Vec<i32> = (0..1024).map(|i| 10_000 + (i % 64)).collect();
    let payload = ForBitpackEncoder::encode_i32(&values);
    let packed_bytes = payload.len() - 9; // minus the [i32 ref][u8 bw][u32 len] header
    assert_eq!(packed_bytes, bitpack::packed_len(values.len(), 6));
    assert!(
        packed_bytes * 4 < values.len() * 4,
        "expected >4x reduction: packed={packed_bytes} raw={}",
        values.len() * 4
    );
}

#[test]
fn for_bitpack_round_trips() {
    let values: Vec<i32> = (0..1024).map(|i| 10_000 + (i % 64)).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let out = round_trip(
        FOR_BITPACK_KERNEL,
        "test.for-bitpack",
        Arc::new(ForBitpackEncoder),
        array,
    );

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}
