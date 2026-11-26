// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Untrusted wire-input fuzzing for IPC and protobuf boundaries.
//!
//! Feeds arbitrary and bit-flipped bytes into `vortex-ipc` message readers and protobuf
//! `Expr`/`Scalar`/`ScalarValue` deserializers, ensuring corrupt inputs return errors rather than
//! panicking or looping infinitely.

use std::io::Cursor;
use std::sync::LazyLock;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use prost::Message as _;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::eq;
use vortex::expr::get_item;
use vortex::expr::gt;
use vortex::expr::lit;
use vortex::expr::proto::ExprSerializeProtoExt;
use vortex::expr::root;
use vortex::ipc::messages::BufMessageReader;
use vortex::ipc::messages::DecoderMessage;
use vortex::ipc::messages::EncoderMessage;
use vortex::ipc::messages::MessageEncoder;
use vortex::ipc::messages::SyncMessageReader;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::proto::expr as pb_expr;
use vortex_array::proto::scalar as pb_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexExpect;

use crate::SESSION;

/// Upper bound on rows executed from decoded IPC arrays to avoid slow or large canonicalization in
/// fuzzing. `ArrayParts::decode` intentionally runs before this check so lazy deserializers still
/// see untrusted large `row_count` values and exercise their length/overflow validation.
const MAX_WIRE_ROWS: usize = 4096;
/// Upper bound on IPC messages drained from an untrusted stream.
const MAX_IPC_STEPS: usize = 64;
/// Upper bound on raw untrusted payload length generated via [`Arbitrary`].
const MAX_RAW_PAYLOAD_LEN: usize = 4096;

static SEED_IPC_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let array = PrimitiveArray::from_iter([1i32, 2, 3, 4]).into_array();
    let mut encoder = MessageEncoder::new(SESSION.clone());
    let mut bytes = Vec::new();
    for buf in encoder
        .encode(EncoderMessage::DType(array.dtype()))
        .vortex_expect("seed IPC dtype must encode")
    {
        bytes.extend_from_slice(buf.as_ref());
    }
    for buf in encoder
        .encode(EncoderMessage::Array(&array))
        .vortex_expect("seed IPC array must encode")
    {
        bytes.extend_from_slice(buf.as_ref());
    }
    bytes
});

static SEED_EXPR_PROTO_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let expr = and(
        gt(get_item("a", root()), lit(1i32)),
        eq(get_item("b", root()), lit(2i32)),
    );
    expr.serialize_proto()
        .vortex_expect("seed expression must serialize to proto")
        .encode_to_vec()
});

static SEED_SCALAR_VALUE_DTYPE: LazyLock<DType> = LazyLock::new(|| {
    DType::Struct(
        StructFields::new(
            FieldNames::from(["id", "label"]),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
        ),
        Nullability::Nullable,
    )
});

static SEED_SCALAR_PROTO_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let scalar = Scalar::struct_(
        SEED_SCALAR_VALUE_DTYPE.clone(),
        vec![Scalar::from(42i32), Scalar::from("vortex")],
    );
    pb_scalar::Scalar::from(&scalar).encode_to_vec()
});

static SEED_SCALAR_VALUE_PROTO_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let scalar = Scalar::struct_(
        SEED_SCALAR_VALUE_DTYPE.clone(),
        vec![Scalar::from(42i32), Scalar::from("vortex")],
    );
    pb_scalar::ScalarValue::from(scalar.value().vortex_expect("seed scalar is non-null"))
        .encode_to_vec()
});

/// Which wire boundary to exercise.
#[derive(Debug, Clone, Copy)]
pub enum WireTarget {
    /// `vortex-ipc` message readers.
    Ipc,
    /// Protobuf `Expr` decoding.
    Expr,
    /// Protobuf `Scalar` decoding.
    Scalar,
    /// Protobuf `ScalarValue` decoding.
    ScalarValue,
}

impl<'a> Arbitrary<'a> for WireTarget {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        u.choose(&[Self::Ipc, Self::Expr, Self::Scalar, Self::ScalarValue])
            .copied()
    }
}

/// Fuzz input for untrusted wire-format decoding safety.
#[derive(Debug)]
pub struct FuzzWireSafety {
    /// Wire boundary to exercise.
    pub target: WireTarget,
    /// Whether `payload` was bit-flipped from a valid seed rather than generated raw.
    pub flip: bool,
    /// Raw or mutated wire payload bytes.
    pub payload: Vec<u8>,
}

impl<'a> Arbitrary<'a> for FuzzWireSafety {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let target = WireTarget::arbitrary(u)?;
        let flip = bool::arbitrary(u)?;
        let payload = if flip {
            let seed: &[u8] = match target {
                WireTarget::Ipc => &SEED_IPC_BYTES,
                WireTarget::Expr => &SEED_EXPR_PROTO_BYTES,
                WireTarget::Scalar => &SEED_SCALAR_PROTO_BYTES,
                WireTarget::ScalarValue => &SEED_SCALAR_VALUE_PROTO_BYTES,
            };
            mutate_seed(seed, u)?
        } else {
            let len = u.int_in_range(0usize..=MAX_RAW_PAYLOAD_LEN)?;
            let available = len.min(u.len());
            u.bytes(available)?.to_vec()
        };
        Ok(Self {
            target,
            flip,
            payload,
        })
    }
}

fn mutate_seed(seed: &[u8], u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<u8>> {
    let mut bytes = seed.to_vec();
    if !bytes.is_empty() {
        let flips = u.int_in_range(1usize..=8)?;
        for _ in 0..flips {
            let idx = u.choose_index(bytes.len())?;
            let bit = u.int_in_range(0u8..=7)?;
            bytes[idx] ^= 1u8 << bit;
        }
        if u.arbitrary::<bool>()? {
            let new_len = u.int_in_range(0usize..=bytes.len())?;
            bytes.truncate(new_len);
        }
    }
    Ok(bytes)
}

/// Exercise the selected wire boundary with untrusted bytes.
///
/// Corrupt inputs must return `Err` cleanly rather than panicking or looping infinitely.
pub fn run_wire_safety(input: FuzzWireSafety) {
    match input.target {
        WireTarget::Ipc => fuzz_ipc_bytes(&input.payload),
        WireTarget::Expr => fuzz_expr_proto_bytes(&input.payload),
        WireTarget::Scalar => fuzz_scalar_proto_bytes(&input.payload),
        WireTarget::ScalarValue => fuzz_scalar_value_proto_bytes(&input.payload),
    }
}

fn decode_ipc_message(msg: DecoderMessage, current_dtype: &mut DType) {
    match msg {
        DecoderMessage::DType(fb_dtype) => {
            if let Ok(dtype) = DType::from_flatbuffer(fb_dtype, &SESSION) {
                *current_dtype = dtype;
            }
        }
        DecoderMessage::Array((parts, ctx, row_count)) => {
            if let Ok(array) = parts.decode(current_dtype, row_count, &ctx, &SESSION)
                && row_count <= MAX_WIRE_ROWS
            {
                let mut exec_ctx = SESSION.create_execution_ctx();
                drop(array.execute::<Canonical>(&mut exec_ctx));
            }
        }
        DecoderMessage::Buffer(_) => {}
    }
}

/// Feed untrusted bytes into `vortex-ipc` message readers with a bounded step drain.
fn fuzz_ipc_bytes(bytes: &[u8]) {
    let mut current_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    for item in BufMessageReader::new(bytes).take(MAX_IPC_STEPS) {
        match item {
            Ok(msg) => decode_ipc_message(msg, &mut current_dtype),
            Err(_) => break,
        }
    }

    let mut sync_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    for item in SyncMessageReader::new(Cursor::new(bytes)).take(MAX_IPC_STEPS) {
        match item {
            Ok(msg) => decode_ipc_message(msg, &mut sync_dtype),
            Err(_) => break,
        }
    }
}

/// Feed untrusted bytes into protobuf `Expr` decoding and [`Expression::from_proto`].
fn fuzz_expr_proto_bytes(bytes: &[u8]) {
    if let Ok(proto) = pb_expr::Expr::decode(bytes) {
        drop(Expression::from_proto(&proto, &SESSION));
    }
}

/// Feed untrusted bytes into protobuf `Scalar` deserialization.
fn fuzz_scalar_proto_bytes(bytes: &[u8]) {
    if let Ok(proto) = pb_scalar::Scalar::decode(bytes) {
        drop(Scalar::from_proto(&proto, &SESSION));
    }
}

/// Feed untrusted bytes into protobuf `ScalarValue` deserialization.
fn fuzz_scalar_value_proto_bytes(bytes: &[u8]) {
    drop(ScalarValue::from_proto_bytes(
        bytes,
        &SEED_SCALAR_VALUE_DTYPE,
        &SESSION,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_wire_seeds_decode_cleanly() {
        let buf_messages: Vec<_> = BufMessageReader::new(SEED_IPC_BYTES.as_slice())
            .collect::<Result<_, _>>()
            .vortex_expect("seed IPC bytes must decode via BufMessageReader");
        assert_eq!(buf_messages.len(), 2);

        let sync_messages: Vec<_> = SyncMessageReader::new(Cursor::new(SEED_IPC_BYTES.as_slice()))
            .collect::<Result<_, _>>()
            .vortex_expect("seed IPC bytes must decode via SyncMessageReader");
        assert_eq!(sync_messages.len(), 2);

        let expr_proto = pb_expr::Expr::decode(SEED_EXPR_PROTO_BYTES.as_slice())
            .vortex_expect("seed expr proto must decode");
        Expression::from_proto(&expr_proto, &SESSION)
            .vortex_expect("seed expr must deserialize from proto");

        let scalar_proto = pb_scalar::Scalar::decode(SEED_SCALAR_PROTO_BYTES.as_slice())
            .vortex_expect("seed scalar proto must decode");
        Scalar::from_proto(&scalar_proto, &SESSION)
            .vortex_expect("seed scalar must deserialize from proto");

        let scalar_value = ScalarValue::from_proto_bytes(
            &SEED_SCALAR_VALUE_PROTO_BYTES,
            &SEED_SCALAR_VALUE_DTYPE,
            &SESSION,
        )
        .vortex_expect("seed scalar value must deserialize from proto");
        assert!(scalar_value.is_some());

        fuzz_ipc_bytes(&SEED_IPC_BYTES);
        fuzz_expr_proto_bytes(&SEED_EXPR_PROTO_BYTES);
        fuzz_scalar_proto_bytes(&SEED_SCALAR_PROTO_BYTES);
        fuzz_scalar_value_proto_bytes(&SEED_SCALAR_VALUE_PROTO_BYTES);
    }

    #[test]
    fn test_ipc_overflowing_row_count_returns_error() {
        let mut messages = BufMessageReader::new(SEED_IPC_BYTES.as_slice());
        let DecoderMessage::DType(fb_dtype) = messages
            .next()
            .vortex_expect("expected DType message")
            .vortex_expect("seed DType must decode")
        else {
            unreachable!("first seed IPC message is DType");
        };
        let dtype =
            DType::from_flatbuffer(fb_dtype, &SESSION).vortex_expect("seed DType must deserialize");

        let DecoderMessage::Array((parts, ctx, row_count)) = messages
            .next()
            .vortex_expect("expected Array message")
            .vortex_expect("seed Array must decode")
        else {
            unreachable!("second seed IPC message is Array");
        };

        let overflowing_row_count = row_count ^ (1usize << 62);
        assert!(
            parts
                .decode(&dtype, overflowing_row_count, &ctx, &SESSION)
                .is_err()
        );
    }
}
