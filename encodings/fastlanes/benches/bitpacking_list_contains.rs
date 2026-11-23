// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures `list_contains` (`IN (...)`) over compressed `BitPacked` and `FoR(BitPacked)` arrays.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpacking_list_contains`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::BitWidths;
use vortex_fastlanes::FoR;
use vortex_functions::expr::list_contains;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_functions::initialize(&session);
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 65_536;

/// Pin the packed buffer to a page boundary so CodSpeed cache-line placement is deterministic.
fn page_aligned(array: BitPackedArray) -> BitPackedArray {
    let ptype = array.dtype().as_ptype();
    let parts = BitPacked::into_parts(array);
    let BitWidths::Global(bit_width) = parts.bit_widths else {
        unreachable!("bitpack_encode packs every block at one bit width")
    };
    let packed = BufferHandle::new_host(
        parts
            .packed
            .ensure_aligned(Alignment::new(4096))
            .unwrap()
            .to_host_sync()
            .aligned(Alignment::new(ptype.byte_width())),
    );
    BitPacked::try_new(
        packed,
        ptype,
        parts.validity,
        parts.patches,
        bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

fn build_bitpacked<T>(
    bit_width: u8,
    members: &[u64],
    patch_every: Option<usize>,
    from_u64: impl Fn(u64) -> T,
) -> ArrayRef
where
    T: IntegerPType + Copy,
{
    let domain = 1u64 << bit_width;
    let patch_hit = domain;
    let patch_miss = domain + 1;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut buf = BufferMut::<T>::with_capacity(LEN);
    for idx in 0..LEN {
        let value = if let Some(step) = patch_every
            && idx.is_multiple_of(step)
        {
            if (idx / step).is_multiple_of(2) {
                patch_hit
            } else {
                patch_miss
            }
        } else {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if !members.is_empty() && (state >> 32).is_multiple_of(2) {
                let candidate = members[((state >> 33) as usize) % members.len()];
                if candidate < domain {
                    candidate
                } else {
                    (state >> 40) % domain
                }
            } else {
                (state >> 40) % domain
            }
        };
        buf.push(from_u64(value));
    }

    let mut ctx = SESSION.create_execution_ctx();
    let prim = PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array();
    page_aligned(BitPackedData::encode(&prim, bit_width, &mut ctx).unwrap()).into_array()
}

fn bench_list_contains(bencher: Bencher, set: Scalar, needles: ArrayRef) {
    let expr = list_contains(lit(set), root())
        .bind(needles.dtype())
        .unwrap()
        .optimize_recursive()
        .unwrap();
    bencher
        .with_inputs(|| {
            let array = needles
                .clone()
                .apply_bound(&expr)
                .unwrap()
                .optimize_ctx(&SESSION)
                .unwrap();
            (array, SESSION.create_execution_ctx())
        })
        .bench_values(|(array, mut ctx)| array.execute::<BoolArray>(&mut ctx).unwrap());
}

fn list_scalar<T: IntegerPType + Into<Scalar>>(values: impl IntoIterator<Item = T>) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(T::PTYPE, Nullability::NonNullable)),
        values.into_iter().map(Into::into).collect(),
        Nullability::NonNullable,
    )
}

/// Non-contiguous in-domain members (stride 3 so `in_list_of` cannot rewrite to `between`).
fn strided_members(count: usize) -> Vec<u64> {
    (0..count).map(|i| 1 + (i as u64) * 3).collect()
}

#[divan::bench]
fn bitpacked_u32_m4(bencher: Bencher) {
    let members = strided_members(4);
    let needles = build_bitpacked(8, &members, None, |v| v as u32);
    let set = list_scalar(members.into_iter().map(|v| v as u32));
    bench_list_contains(bencher, set, needles);
}

#[divan::bench]
fn bitpacked_u32_out_of_domain(bencher: Bencher) {
    let bit_width = 8;
    let base = 1u64 << bit_width;
    let members: Vec<u64> = (0..4).map(|i| base + i * 3).collect();
    let needles = build_bitpacked(bit_width, &[], None, |v| v as u32);
    let set = list_scalar(members.into_iter().map(|v| v as u32));
    bench_list_contains(bencher, set, needles);
}

#[divan::bench]
fn bitpacked_u32_patched(bencher: Bencher) {
    let bit_width = 8;
    let domain = 1u64 << bit_width;
    let members = vec![1, 7, domain, domain + 4];
    let needles = build_bitpacked(bit_width, &members, Some(64), |v| v as u32);
    let set = list_scalar(members.into_iter().map(|v| v as u32));
    bench_list_contains(bencher, set, needles);
}

#[divan::bench]
fn for_bitpacked_i32_m4(bencher: Bencher) {
    let reference = 10_000i32;
    let packed_members = strided_members(4);
    let query_members: Vec<i32> = packed_members
        .iter()
        .map(|&v| reference + v as i32)
        .collect();
    let packed = build_bitpacked(8, &packed_members, None, |v| v as i32);
    let needles = FoR::try_new(packed, Scalar::from(reference))
        .unwrap()
        .into_array();
    bench_list_contains(bencher, list_scalar(query_members), needles);
}

#[divan::bench]
fn for_bitpacked_i32_below_ref(bencher: Bencher) {
    let reference = 10_000i32;
    let packed = build_bitpacked(8, &[], None, |v| v as i32);
    let needles = FoR::try_new(packed, Scalar::from(reference))
        .unwrap()
        .into_array();
    bench_list_contains(bencher, list_scalar([100i32, 200, 500, 9_999]), needles);
}
