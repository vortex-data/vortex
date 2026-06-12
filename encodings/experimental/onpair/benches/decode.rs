// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Decode-path microbenchmarks for the OnPair Vortex array.
//!
//! * `decompress_into` — the upstream `onpair::decompress_into` decoder hot
//!   loop, fed by pre-materialised [`DecodeInputs`]. Measures the inner loop
//!   only (no child `execute`, no allocation).
//! * `canonicalize_to_varbinview` — the full Vortex
//!   `OnPair → VarBinViewArray` path callers actually hit. Includes child
//!   `execute`, the build_views step, allocation, etc.
//!
//! Each bench sweeps four corpus shapes against two row counts to surface
//! cache-pressure cliffs and per-row decode cost.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::panic,
    clippy::tests_outside_test_module,
    clippy::redundant_clone,
    clippy::missing_safety_doc,
    clippy::unwrap_used,
    clippy::expect_used
)]

use std::mem::MaybeUninit;
use std::sync::LazyLock;

use divan::Bencher;
use onpair::Parts;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_mask::Mask;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArray;
use vortex_onpair::OnPairArraySlotsExt;

/// Host-resident decode inputs, materialised once so the decode-loop benchmark
/// measures only `onpair::decompress_into` (not child `execute`/allocation).
struct DecodeInputs {
    dict_bytes: ByteBuffer,
    dict_offsets: Buffer<u32>,
    codes: Buffer<u16>,
    bits: u32,
}

impl DecodeInputs {
    fn as_parts(&self) -> Parts<'_> {
        Parts {
            dict_bytes: self.dict_bytes.as_slice(),
            dict_offsets: self.dict_offsets.as_slice(),
            bits: self.bits,
            codes: self.codes.as_slice(),
        }
    }

    fn decompressed_len(&self) -> usize {
        onpair::decompressed_len(self.as_parts())
    }

    fn decompress_into(&self, out: &mut [MaybeUninit<u8>]) -> usize {
        onpair::decompress_into(self.as_parts(), out)
    }
}
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[derive(Copy, Clone, Debug)]
enum Shape {
    /// URL / HTTP-log shaped — high lexical overlap, ~35–45 bytes per row.
    UrlLog,
    /// Short uniform strings — 4–8 bytes per row, very low cardinality.
    Short,
    /// Long log-line shaped — ~120 bytes per row, more tokens per row.
    Long,
    /// High cardinality — every row unique.
    HighCard,
}

fn corpus(n: usize, shape: Shape) -> Vec<String> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let mut out = Vec::with_capacity(n);
    match shape {
        Shape::UrlLog => {
            let templates: &[&str] = &[
                "https://www.example.com/products/{id}",
                "https://cdn.example.com/img/{id}.webp",
                "https://api.example.com/v2/orders/{id}",
                "https://www.example.com/users/{id}/profile",
                "INFO  request_id={id} status=200 method=GET",
                "WARN  request_id={id} status=429 method=POST",
                "ERROR request_id={id} status=500 method=PUT",
            ];
            for _ in 0..n {
                let s = next();
                let pick = (s as usize) % templates.len();
                let id = s as u32;
                out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
            }
        }
        Shape::Short => {
            let templates: &[&str] = &["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"];
            for _ in 0..n {
                let s = next();
                out.push(templates[(s as usize) % templates.len()].to_string());
            }
        }
        Shape::Long => {
            let templates: &[&str] = &[
                "2026-05-14T12:34:56.789012Z INFO  request_id={id} method=GET path=/api/v1/users/{id}/profile status=200",
                "2026-05-14T12:34:56.789012Z WARN  request_id={id} method=POST path=/api/v1/users/{id}/sessions status=429",
                "2026-05-14T12:34:56.789012Z ERROR request_id={id} method=PUT  path=/api/v1/users/{id}/settings status=500",
            ];
            for _ in 0..n {
                let s = next();
                let pick = (s as usize) % templates.len();
                let id = s as u32;
                out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
            }
        }
        Shape::HighCard => {
            for i in 0..n {
                out.push(format!("row-{i:010x}-{rand:016x}", rand = next()));
            }
        }
    }
    out
}

fn compress(n: usize, shape: Shape) -> OnPairArray {
    let strings = corpus(n, shape);
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)
        .unwrap_or_else(|e| panic!("onpair_compress failed: {e}"))
}

/// Canonicalise a slot child to the decoder's native primitive width.
fn widen<T: NativePType>(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> Buffer<T> {
    arr.cast(DType::Primitive(T::PTYPE, arr.dtype().nullability()))
        .expect("cast")
        .execute::<PrimitiveArray>(ctx)
        .expect("execute")
        .into_buffer::<T>()
}

fn materialise(arr: &OnPairArray) -> (DecodeInputs, usize) {
    let mut ctx = SESSION.create_execution_ctx();
    let view = arr.as_view();
    let inputs = DecodeInputs {
        dict_bytes: view.dict_bytes().clone(),
        dict_offsets: widen::<u32>(view.dict_offsets(), &mut ctx),
        codes: widen::<u16>(view.codes(), &mut ctx),
        bits: view.bits(),
    };
    let total = inputs.decompressed_len();
    (inputs, total)
}

const CASES: &[(Shape, usize)] = &[
    (Shape::UrlLog, 100_000),
    (Shape::UrlLog, 1_000_000),
    (Shape::Short, 100_000),
    (Shape::Long, 100_000),
    (Shape::HighCard, 100_000),
];

/// Raw decode loop time, excluding child `execute` and the output allocation.
/// Hits `onpair::decompress_into` directly.
#[divan::bench(args = CASES)]
fn decompress_into_bench(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let arr = compress(n, shape);
    let (inputs, total) = materialise(&arr);
    bencher.bench_local(|| {
        let mut out: Vec<u8> = Vec::with_capacity(total);
        let written = inputs.decompress_into(out.spare_capacity_mut());
        unsafe { out.set_len(written) };
        divan::black_box(out);
    });
}

/// Full Vortex canonicalisation, including `execute<>` on every child,
/// building the view buffer + `BinaryView` list, etc.
#[divan::bench(args = CASES)]
fn canonicalize_to_varbinview(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let arr = compress(n, shape);
    bencher
        .with_inputs(|| (arr.clone().into_array(), SESSION.create_execution_ctx()))
        .bench_local_values(|(arr, mut ctx)| {
            divan::black_box(
                arr.execute::<VarBinViewArray>(&mut ctx)
                    .unwrap_or_else(|e| panic!("canonicalize failed: {e}")),
            )
        });
}

// ─── Compute kernels ─────────────────────────────────────────────────────

const COMPUTE_CASES: &[(Shape, usize)] = &[(Shape::UrlLog, 100_000), (Shape::UrlLog, 1_000_000)];

/// Filter — share-dict path. Builds a 1-in-7 mask so we keep ~14 % of
/// rows; the cost is dominated by the `codes` segment copy + offsets.
#[divan::bench(args = COMPUTE_CASES)]
fn filter_share_dict(bencher: Bencher, case: (Shape, usize)) {
    let (shape, n) = case;
    let arr = compress(n, shape);
    let mask = Mask::from_iter((0..n).map(|i| i % 7 == 0));
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_local_values(|mut ctx| {
            let result = <OnPair as FilterKernel>::filter(arr.as_view(), &mask, &mut ctx)
                .unwrap()
                .unwrap();
            divan::black_box(result);
        });
}

fn main() {
    divan::main();
}
