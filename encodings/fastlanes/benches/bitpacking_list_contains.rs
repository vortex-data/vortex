// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures compressed constant-list membership.
//!
//! FastLanes evaluates at most four distinct integer members during unpacking. Larger sets use the
//! frozen generic path. Every path runs on each real CPU feature leg in CodSpeed.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpacking_list_contains`.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::hint::black_box;
use std::sync::Arc;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayExt;
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
    patch_every: Option<usize>,
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
        patch_every: None,
    }
}

const fn patched_case(
    name: &'static str,
    ptype: PType,
    bit_width: u8,
    len: usize,
    count: usize,
    stride: u64,
    patch_every: usize,
) -> PackedCase {
    PackedCase {
        name,
        ptype,
        bit_width,
        len,
        member_count: count,
        member_stride: stride,
        patch_every: Some(patch_every),
    }
}

const PACKED_CASES: &[PackedCase] = &[
    strided_case("direct_u8_m4", PType::U8, 6, 65_536, 4, 2),
    strided_case("fallback_u8_m5", PType::U8, 6, 65_536, 5, 2),
    strided_case("direct_u16_m4", PType::U16, 8, 65_536, 4, 2),
    strided_case("fallback_u16_m5", PType::U16, 8, 65_536, 5, 2),
    strided_case("direct_u32_m4", PType::U32, 8, 65_536, 4, 2),
    strided_case("fallback_u32_m5", PType::U32, 8, 65_536, 5, 2),
    strided_case("direct_u64_m4", PType::U64, 40, 65_536, 4, 2),
    strided_case("fallback_u64_m5", PType::U64, 40, 65_536, 5, 2),
    strided_case("short_direct_u32_m4", PType::U32, 10, 1_024, 4, 2),
    strided_case("short_fallback_u32_m5", PType::U32, 10, 1_024, 5, 2),
    strided_case("wide_direct_u32_m4", PType::U32, 31, 65_536, 4, 2),
    patched_case("patch_sparse_u32_m4", PType::U32, 8, 65_536, 4, 2, 64),
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

fn generated_values(case: PackedCase, ordinary_members: &[u64]) -> Vec<u64> {
    let domain_size = 1u64 << case.bit_width;
    let patch_hit = domain_size;
    let patch_miss = patch_hit + 1;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..case.len)
        .map(|index| {
            if let Some(patch_every) = case.patch_every
                && index.is_multiple_of(patch_every)
            {
                return if (index / patch_every).is_multiple_of(2) {
                    patch_hit
                } else {
                    patch_miss
                };
            }
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let is_hit = (state >> 32).is_multiple_of(2);
            if is_hit {
                let member_index =
                    usize::try_from(state % u64::try_from(ordinary_members.len()).unwrap())
                        .unwrap();
                ordinary_members[member_index]
            } else {
                let mut candidate = state.rotate_left(17) % domain_size;
                while ordinary_members.contains(&candidate) {
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

fn frozen_generic_membership<T: BenchInt>(
    values: ArrayRef,
    members: &[u64],
) -> VortexResult<ArrayRef> {
    fn balanced_or(arrays: &[ArrayRef]) -> VortexResult<ArrayRef> {
        if let [array] = arrays {
            return Ok(array.clone());
        }
        let (left, right) = arrays.split_at(arrays.len() / 2);
        balanced_or(left)?.binary(balanced_or(right)?, Operator::Or)
    }

    let len = values.len();
    let nullability = values.dtype().nullability();
    let false_scalar = Scalar::bool(false, nullability);
    let comparisons = members
        .iter()
        .map(|member| {
            Binary::try_new(
                ConstantArray::new(T::from_counter(*member).into(), len).into_array(),
                values.clone(),
                Operator::Eq,
            )?
            .into_array()
            .fill_null(false_scalar.clone())
        })
        .collect::<VortexResult<Vec<_>>>()?;

    if comparisons.is_empty() {
        Ok(ConstantArray::new(false_scalar, len).into_array())
    } else {
        balanced_or(&comparisons)
    }
}

fn execute_generic_baseline<T: BenchInt>(
    values: &BitPackedArray,
    members: &[u64],
    ctx: &mut vortex_array::ExecutionCtx,
) -> BoolArray {
    frozen_generic_membership::<T>(values.clone().into_array(), members)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap()
}

fn packed_input<T: BenchInt>(
    case: PackedCase,
) -> (BitPackedArray, Vec<u64>, BoolArray, VortexSession) {
    let session = array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let in_domain_member_count = case.member_count - usize::from(case.patch_every.is_some());
    let ordinary_members = (0..in_domain_member_count)
        .map(|index| u64::try_from(index).unwrap() * case.member_stride)
        .collect::<Vec<_>>();
    let mut members = ordinary_members.clone();
    if case.patch_every.is_some() {
        members.push(1u64 << case.bit_width);
    }
    let generated = generated_values(case, &ordinary_members);
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
    if case.patch_every.is_some() {
        assert!(packed.patches().is_some());
    }
    (packed, members, expected, session)
}

fn bench_packed_current<T: BenchInt>(bencher: Bencher, case: PackedCase) {
    let (packed, members, expected, session) = packed_input::<T>(case);
    let contains = packed
        .into_array()
        .apply(&list_contains(lit(list_scalar::<T>(&members)), root()))
        .unwrap();
    let mut ctx = session.create_execution_ctx();
    let actual = contains.clone().execute::<BoolArray>(&mut ctx).unwrap();
    assert_arrays_eq!(actual, expected, &mut ctx);
    bencher
        .counter(ItemsCount::new(case.len))
        .bench_local(|| black_box(contains.clone().execute::<BoolArray>(&mut ctx).unwrap()));
}

fn bench_packed_generic_baseline<T: BenchInt>(bencher: Bencher, case: PackedCase) {
    let (packed, members, expected, session) = packed_input::<T>(case);
    let mut ctx = session.create_execution_ctx();
    let actual = execute_generic_baseline::<T>(&packed, &members, &mut ctx);
    assert_arrays_eq!(actual, expected, &mut ctx);
    // The frozen pre-change implementation built this array tree during execution.
    bencher
        .counter(ItemsCount::new(case.len))
        .bench_local(|| black_box(execute_generic_baseline::<T>(&packed, &members, &mut ctx)));
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

#[vortex_bench_support::cpu_features]
#[divan::bench(args = PACKED_CASES)]
fn packed_generic_baseline(bencher: Bencher, case: PackedCase) {
    dispatch_packed!(bencher, case, bench_packed_generic_baseline);
}
