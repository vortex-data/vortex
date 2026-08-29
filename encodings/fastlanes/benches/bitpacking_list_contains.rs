// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures compressed constant-list membership.
//!
//! FastLanes evaluates constant lists with at most four distinct non-null members during unpacking.
//! Mid-size lists use repeated packed comparisons. Larger lists decode once at a threshold that
//! depends on the physical integer width and array length. Every path runs on each real CPU feature
//! leg in CodSpeed.
//! To recalculate the thresholds, temporarily replace `min_decode_source_members` with a constant.
//! Return `usize::MAX` to force repeated comparisons. Return `5` to force decode-once.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpacking_list_contains`.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::hint::black_box;
use std::sync::Arc;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

trait BenchInt: IntegerPType + Copy + Into<Scalar> {
    fn from_counter(value: u64) -> Self;
}

impl BenchInt for u8 {
    fn from_counter(value: u64) -> Self {
        Self::try_from(value).unwrap()
    }
}

impl BenchInt for u16 {
    fn from_counter(value: u64) -> Self {
        Self::try_from(value).unwrap()
    }
}

impl BenchInt for u32 {
    fn from_counter(value: u64) -> Self {
        Self::try_from(value).unwrap()
    }
}

impl BenchInt for u64 {
    fn from_counter(value: u64) -> Self {
        value
    }
}

#[derive(Clone, Copy)]
struct PackedCase {
    name: &'static str,
    ptype: PType,
    bit_width: u8,
    len: usize,
    member_count: usize,
    member_stride: u64,
}

impl Display for PackedCase {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}_{}_w{}_n{}",
            self.name, self.ptype, self.bit_width, self.len
        )
    }
}

const fn strided_case(
    name: &'static str,
    ptype: PType,
    bit_width: u8,
    len: usize,
    count: usize,
    stride: u64,
) -> PackedCase {
    PackedCase {
        name,
        ptype,
        bit_width,
        len,
        member_count: count,
        member_stride: stride,
    }
}

const PACKED_CASES: &[PackedCase] = &[
    strided_case("direct_u8_m4", PType::U8, 6, 65_536, 4, 2),
    strided_case("direct_u16_m4", PType::U16, 12, 65_536, 4, 2),
    strided_case("direct_u32_m4", PType::U32, 20, 65_536, 4, 2),
    strided_case("direct_u64_m4", PType::U64, 40, 65_536, 4, 2),
    strided_case("generic_u8_m29", PType::U8, 6, 65_536, 29, 2),
    strided_case("decode_u8_m30", PType::U8, 6, 65_536, 30, 2),
    strided_case("generic_u16_m24", PType::U16, 8, 65_536, 24, 2),
    strided_case("decode_u16_m25", PType::U16, 8, 65_536, 25, 2),
    strided_case("generic_u32_m12", PType::U32, 8, 65_536, 12, 2),
    strided_case("decode_u32_m13", PType::U32, 8, 65_536, 13, 2),
    strided_case("decode_u64_m5", PType::U64, 40, 65_536, 5, 2),
    strided_case("short_direct_u32_m4", PType::U32, 10, 1_024, 4, 2),
    strided_case("short_generic_u8_m9", PType::U8, 6, 8_192, 9, 2),
    strided_case("short_decode_u8_m10", PType::U8, 6, 8_192, 10, 2),
    strided_case("short_generic_u16_m9", PType::U16, 8, 8_192, 9, 2),
    strided_case("short_decode_u16_m10", PType::U16, 8, 8_192, 10, 2),
    strided_case("short_generic_u32_m10", PType::U32, 8, 16_384, 10, 2),
    strided_case("short_decode_u32_m11", PType::U32, 8, 16_384, 11, 2),
    strided_case("longer_generic_u8_m10", PType::U8, 6, 16_384, 10, 2),
    strided_case("longer_generic_u16_m10", PType::U16, 8, 16_384, 10, 2),
    strided_case("longer_generic_u32_m11", PType::U32, 8, 32_768, 11, 2),
    strided_case("short_direct_u64_m4", PType::U64, 8, 8_192, 4, 2),
    strided_case("short_decode_u64_m5", PType::U64, 8, 8_192, 5, 2),
];

fn page_aligned(array: BitPackedArray) -> BitPackedArray {
    let ptype = array.dtype().as_ptype();
    let parts = BitPacked::into_parts(array);
    BitPacked::try_new(
        parts.packed.ensure_aligned(Alignment::new(4_096)).unwrap(),
        ptype,
        parts.validity,
        parts.patches,
        parts.bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

fn generated_values(case: PackedCase, members: &[u64]) -> Vec<u64> {
    let domain_size = 1u64 << case.bit_width;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..case.len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let is_hit = (state >> 32).is_multiple_of(2);
            if is_hit {
                let member_index =
                    usize::try_from(state % u64::try_from(members.len()).unwrap()).unwrap();
                members[member_index]
            } else {
                let mut candidate = state.rotate_left(17) % domain_size;
                while members.contains(&candidate) {
                    candidate = (candidate + 1) % domain_size;
                }
                candidate
            }
        })
        .collect()
}

fn list_scalar<T: BenchInt>(members: &[u64]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(T::PTYPE, Nullability::NonNullable)),
        members
            .iter()
            .map(|value| T::from_counter(*value).into())
            .collect(),
        Nullability::NonNullable,
    )
}

fn packed_input<T: BenchInt>(
    case: PackedCase,
) -> (BitPackedArray, Scalar, BoolArray, VortexSession) {
    let session = array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let members = (0..case.member_count)
        .map(|index| u64::try_from(index).unwrap() * case.member_stride)
        .collect::<Vec<_>>();
    let generated = generated_values(case, &members);
    let expected = BoolArray::from_iter(generated.iter().map(|value| members.contains(value)));
    let values: BufferMut<T> = generated.into_iter().map(T::from_counter).collect();
    let packed = page_aligned(
        BitPackedData::encode(
            &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
            case.bit_width,
            &mut ctx,
        )
        .unwrap(),
    );
    (packed, list_scalar::<T>(&members), expected, session)
}

fn bench_packed_current<T: BenchInt>(bencher: Bencher, case: PackedCase) {
    let (packed, list, expected, session) = packed_input::<T>(case);
    let contains = packed
        .into_array()
        .apply(&list_contains(lit(list), root()))
        .unwrap();
    let mut ctx = session.create_execution_ctx();
    let actual = contains.clone().execute::<BoolArray>(&mut ctx).unwrap();
    assert_arrays_eq!(actual, expected, &mut ctx);
    bencher
        .counter(ItemsCount::new(case.len))
        .bench_local(|| black_box(contains.clone().execute::<BoolArray>(&mut ctx).unwrap()));
}

macro_rules! dispatch_packed {
    ($bencher:expr, $case:expr, $function:ident) => {
        match $case.ptype {
            PType::U8 => $function::<u8>($bencher, $case),
            PType::U16 => $function::<u16>($bencher, $case),
            PType::U32 => $function::<u32>($bencher, $case),
            PType::U64 => $function::<u64>($bencher, $case),
            _ => unreachable!("benchmark case uses an unsigned integer type"),
        }
    };
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = PACKED_CASES)]
fn packed_current(bencher: Bencher, case: PackedCase) {
    dispatch_packed!(bencher, case, bench_packed_current);
}
