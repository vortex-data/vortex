// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Divan benchmarks for `vortex.pivco` (`PH`, `PHA`, `OP-PHA`) encode, canonical decode,
//! `scalar_at` (clustered vs scattered), `filter` (1%), `take` (1%), and `compare` (`== 0` and
//! `< threshold` on `OP-PHA`) across skewed, Zipf, and signed `ZigZag(Pivco)` distributions.

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_optimized;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_pivco::Pivco;
use vortex_pivco::PivcoOptions;
use vortex_session::VortexSession;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagArraySlotsExt;
use vortex_zigzag::zigzag_encode;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const THROUGHPUT_LEN: usize = 262_144;
const SCALAR_LEN: usize = 16_384;
const SCALAR_PROBES: usize = 256;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_pivco::initialize(&session);
    vortex_zigzag::initialize(&session);
    session
});

#[derive(Clone, Copy)]
enum Dist {
    SkewedGt,
    Zipf64,
    SignedZigZag,
}

impl Display for Dist {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SkewedGt => write!(f, "skewed_gt_u64"),
            Self::Zipf64 => write!(f, "zipf64_u32"),
            Self::SignedZigZag => write!(f, "signed_zigzag_i32"),
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Ph,
    Pha,
    OpPha,
}

impl Mode {
    const fn options(self) -> PivcoOptions {
        match self {
            Self::Ph => PivcoOptions::ph(),
            Self::Pha => PivcoOptions::pha(),
            Self::OpPha => PivcoOptions::op_pha(),
        }
    }
}

impl Display for Mode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ph => write!(f, "ph"),
            Self::Pha => write!(f, "pha"),
            Self::OpPha => write!(f, "op_pha"),
        }
    }
}

#[derive(Clone, Copy)]
enum AccessPattern {
    Clustered,
    Scattered,
}

impl Display for AccessPattern {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clustered => write!(f, "clustered"),
            Self::Scattered => write!(f, "scattered"),
        }
    }
}

#[derive(Clone, Copy)]
struct EncodeCase(Dist, Mode);

impl Display for EncodeCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.0, self.1)
    }
}

#[derive(Clone, Copy)]
struct ScalarCase(Dist, AccessPattern);

impl Display for ScalarCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.0, self.1)
    }
}

const ALL_DISTS: &[Dist] = &[Dist::SkewedGt, Dist::Zipf64, Dist::SignedZigZag];
const UNSIGNED_DISTS: &[Dist] = &[Dist::SkewedGt, Dist::Zipf64];

const ENCODE_DECODE_CASES: &[EncodeCase] = &[
    EncodeCase(Dist::SkewedGt, Mode::Ph),
    EncodeCase(Dist::SkewedGt, Mode::Pha),
    EncodeCase(Dist::SkewedGt, Mode::OpPha),
    EncodeCase(Dist::Zipf64, Mode::Ph),
    EncodeCase(Dist::Zipf64, Mode::Pha),
    EncodeCase(Dist::Zipf64, Mode::OpPha),
    EncodeCase(Dist::SignedZigZag, Mode::Ph),
    EncodeCase(Dist::SignedZigZag, Mode::Pha),
    EncodeCase(Dist::SignedZigZag, Mode::OpPha),
];

const SCALAR_CASES: &[ScalarCase] = &[
    ScalarCase(Dist::SkewedGt, AccessPattern::Clustered),
    ScalarCase(Dist::SkewedGt, AccessPattern::Scattered),
    ScalarCase(Dist::Zipf64, AccessPattern::Clustered),
    ScalarCase(Dist::Zipf64, AccessPattern::Scattered),
];

fn make_primitive(dist: Dist, len: usize) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(2026_1124);
    match dist {
        Dist::SkewedGt => PrimitiveArray::from_iter((0..len).map(|_| {
            let draw = rng.random_range(0..1000u32);
            if draw < 850 {
                0u64
            } else if draw < 980 {
                1u64
            } else {
                2u64
            }
        })),
        Dist::Zipf64 => PrimitiveArray::from_iter((0..len).map(|_| {
            let draw = rng.random_range(0..1000u32);
            if draw < 520 {
                0u32
            } else if draw < 720 {
                1
            } else if draw < 830 {
                2
            } else if draw < 900 {
                rng.random_range(3..8u32)
            } else if draw < 960 {
                rng.random_range(8..32u32)
            } else {
                rng.random_range(32..1_024u32)
            }
        })),
        Dist::SignedZigZag => PrimitiveArray::from_iter((0..len).map(|_| {
            let draw = rng.random_range(0..1000u32);
            let mag = if draw < 600 {
                0i32
            } else if draw < 850 {
                rng.random_range(1..16i32)
            } else {
                rng.random_range(16..512i32)
            };
            if rng.random_bool(0.5) { -mag } else { mag }
        })),
    }
}

fn encode_array(prim: &PrimitiveArray, dist: Dist, mode: Mode) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = match dist {
        Dist::SignedZigZag => {
            let zz = zigzag_encode(prim.as_view()).vortex_expect("zigzag encode");
            let unsigned = zz
                .encoded()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("execute zigzag child");
            let child = Pivco::encode_with_options(&unsigned, mode.options(), &mut ctx)
                .vortex_expect("pivco encode")
                .into_array();
            ZigZag::try_new(child)
                .vortex_expect("zigzag wrap")
                .into_array()
        }
        Dist::SkewedGt | Dist::Zipf64 => Pivco::encode_with_options(prim, mode.options(), &mut ctx)
            .vortex_expect("pivco encode")
            .into_array(),
    };
    mark_optimized(&encoded, &SESSION).vortex_expect("mark optimized");
    encoded
}

fn make_probe_indices(pattern: AccessPattern) -> Vec<usize> {
    let (base, span) = match pattern {
        AccessPattern::Clustered => (4_096, 256),
        AccessPattern::Scattered => (0, SCALAR_LEN),
    };
    let mut seed = 42u64;
    (0..SCALAR_PROBES)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            base + (usize::try_from(seed >> 32).vortex_expect("u32 fits usize") % span)
        })
        .collect()
}

fn make_filter_mask_1pct() -> Mask {
    let mut rng = StdRng::seed_from_u64(0xF117_0001);
    Mask::from_buffer(BitBuffer::from_iter(
        (0..THROUGHPUT_LEN).map(|_| rng.random_range(0..100u32) == 0),
    ))
}

fn make_take_indices_1pct() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0x7A4E_0001);
    let count = THROUGHPUT_LEN / 100;
    let len_u32 = u32::try_from(THROUGHPUT_LEN).vortex_expect("THROUGHPUT_LEN fits u32");
    let indices = PrimitiveArray::from_iter((0..count).map(|_| rng.random_range(0..len_u32)));
    let arr = indices.into_array();
    mark_optimized(&arr, &SESSION).vortex_expect("mark indices optimized");
    arr
}

fn compare_rhs(dist: Dist, threshold: bool) -> ArrayRef {
    let scalar = match (dist, threshold) {
        (Dist::SkewedGt, false) => Scalar::from(0u64),
        (Dist::SkewedGt, true) => Scalar::from(2u64),
        (Dist::Zipf64, false) => Scalar::from(0u32),
        (Dist::Zipf64, true) => Scalar::from(16u32),
        (Dist::SignedZigZag, _) => unreachable!("compare benchmarks run on unsigned OP-PHA"),
    };
    let rhs = ConstantArray::new(scalar, THROUGHPUT_LEN).into_array();
    mark_optimized(&rhs, &SESSION).vortex_expect("mark rhs optimized");
    rhs
}

#[divan::bench(args = ENCODE_DECODE_CASES)]
fn encode(bencher: Bencher, EncodeCase(dist, mode): EncodeCase) {
    let prim = make_primitive(dist, THROUGHPUT_LEN);
    mark_optimized(prim.as_array(), &SESSION).vortex_expect("mark input optimized");
    let opts = mode.options();

    bencher
        .counter(ItemsCount::new(THROUGHPUT_LEN))
        .with_inputs(|| (&prim, SESSION.create_execution_ctx()))
        .bench_values(|(prim, mut ctx)| match dist {
            Dist::SignedZigZag => {
                let zz = zigzag_encode(prim.as_view()).vortex_expect("zigzag encode");
                let unsigned = zz
                    .encoded()
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)
                    .vortex_expect("execute zigzag child");
                let child = Pivco::encode_with_options(&unsigned, opts, &mut ctx)
                    .vortex_expect("pivco encode")
                    .into_array();
                ZigZag::try_new(child)
                    .vortex_expect("zigzag wrap")
                    .into_array()
            }
            Dist::SkewedGt | Dist::Zipf64 => Pivco::encode_with_options(prim, opts, &mut ctx)
                .vortex_expect("pivco encode")
                .into_array(),
        });
}

#[divan::bench(args = ENCODE_DECODE_CASES)]
fn decode(bencher: Bencher, EncodeCase(dist, mode): EncodeCase) {
    let prim = make_primitive(dist, THROUGHPUT_LEN);
    let encoded = encode_array(&prim, dist, mode);

    bencher
        .counter(ItemsCount::new(THROUGHPUT_LEN))
        .with_inputs(|| (encoded.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(arr, mut ctx)| {
            arr.execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("pivco decode")
        });
}

#[divan::bench(args = SCALAR_CASES)]
fn scalar_at(bencher: Bencher, ScalarCase(dist, pattern): ScalarCase) {
    let prim = make_primitive(dist, SCALAR_LEN);
    let encoded = encode_array(&prim, dist, Mode::Pha);
    let indices = make_probe_indices(pattern);

    bencher
        .counter(ItemsCount::new(SCALAR_PROBES))
        .with_inputs(|| (SESSION.create_execution_ctx(), encoded.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for &idx in &indices {
                divan::black_box(
                    probe
                        .execute_scalar(idx, ctx)
                        .vortex_expect("pivco scalar_at"),
                );
            }
        });
}

#[divan::bench(args = ALL_DISTS)]
fn filter_1pct(bencher: Bencher, dist: Dist) {
    let prim = make_primitive(dist, THROUGHPUT_LEN);
    let encoded = encode_array(&prim, dist, Mode::Pha);
    let mask = make_filter_mask_1pct();

    bencher
        .counter(ItemsCount::new(THROUGHPUT_LEN))
        .with_inputs(|| {
            (
                encoded.clone(),
                mask.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, mask, mut ctx)| {
            arr.filter(mask)
                .vortex_expect("pivco filter")
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("execute filtered")
        });
}

#[divan::bench(args = ALL_DISTS)]
fn take_1pct(bencher: Bencher, dist: Dist) {
    let prim = make_primitive(dist, THROUGHPUT_LEN);
    let encoded = encode_array(&prim, dist, Mode::Pha);
    let indices = make_take_indices_1pct();

    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| {
            (
                encoded.clone(),
                indices.clone(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(arr, indices, mut ctx)| {
            arr.take(indices)
                .vortex_expect("pivco take")
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("execute taken")
        });
}

#[divan::bench(args = UNSIGNED_DISTS)]
fn compare_eq_zero(bencher: Bencher, dist: Dist) {
    let prim = make_primitive(dist, THROUGHPUT_LEN);
    let encoded = encode_array(&prim, dist, Mode::OpPha);
    let rhs = compare_rhs(dist, false);

    bencher
        .counter(ItemsCount::new(THROUGHPUT_LEN))
        .with_inputs(|| (encoded.clone(), rhs.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(arr, rhs, mut ctx)| {
            arr.compare(rhs, CompareOperator::Eq)
                .vortex_expect("pivco compare eq")
                .execute::<BoolArray>(&mut ctx)
                .vortex_expect("execute compare eq")
        });
}

#[divan::bench(args = UNSIGNED_DISTS)]
fn compare_lt(bencher: Bencher, dist: Dist) {
    let prim = make_primitive(dist, THROUGHPUT_LEN);
    let encoded = encode_array(&prim, dist, Mode::OpPha);
    let rhs = compare_rhs(dist, true);

    bencher
        .counter(ItemsCount::new(THROUGHPUT_LEN))
        .with_inputs(|| (encoded.clone(), rhs.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(arr, rhs, mut ctx)| {
            arr.compare(rhs, CompareOperator::Lt)
                .vortex_expect("pivco compare lt")
                .execute::<BoolArray>(&mut ctx)
                .vortex_expect("execute compare lt")
        });
}
