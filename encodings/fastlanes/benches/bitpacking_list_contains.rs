// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares compressed list membership with two explicit fallback paths.
//!
//! The decode-once path measures the lower bound for materializing a Primitive array before
//! membership evaluation. The old-generic path freezes the former balanced equality-expression
//! fallback. Primitive cases isolate the benefit of the prepared integer membership set.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpacking_list_contains`.

#![expect(clippy::cast_possible_truncation)]
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
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
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
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
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

macro_rules! impl_bench_int {
    ($($T:ty),+) => {
        $(impl BenchInt for $T {
            fn from_counter(value: u64) -> Self {
                value as $T
            }
        })+
    };
}

impl_bench_int!(u8, u16, u32, u64);

#[derive(Clone, Copy)]
enum MemberSpec {
    Explicit(&'static [u64]),
    Stride { count: usize, stride: u64 },
}

impl MemberSpec {
    fn values(self) -> Vec<u64> {
        match self {
            Self::Explicit(values) => values.to_vec(),
            Self::Stride { count, stride } => {
                (0..count).map(|index| index as u64 * stride).collect()
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PackedCase {
    name: &'static str,
    ptype: PType,
    bit_width: u8,
    len: usize,
    members: MemberSpec,
    hit_percent: u8,
}

impl Display for PackedCase {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}_{}_w{}_n{}_hit{}",
            self.name, self.ptype, self.bit_width, self.len, self.hit_percent
        )
    }
}

const FOUR_BOUNDARY_MEMBERS: &[u64] = &[0, 1, 2, 4_095];
const FIVE_DENSE_BOUNDARY_MEMBERS: &[u64] = &[0, 1, 2, 3, 4_095];
const FIVE_SPARSE_BOUNDARY_MEMBERS: &[u64] = &[0, 1, 2, 3, 4_096];

const PACKED_CASES: &[PackedCase] = &[
    PackedCase {
        name: "short_direct_m4",
        ptype: PType::U32,
        bit_width: 10,
        len: 1_024,
        members: MemberSpec::Stride {
            count: 4,
            stride: 2,
        },
        hit_percent: 50,
    },
    PackedCase {
        name: "four_member_span4096",
        ptype: PType::U32,
        bit_width: 13,
        len: 2_048,
        members: MemberSpec::Explicit(FOUR_BOUNDARY_MEMBERS),
        hit_percent: 50,
    },
    PackedCase {
        name: "before_length_gate_m5_span4096",
        ptype: PType::U32,
        bit_width: 13,
        len: 2_047,
        members: MemberSpec::Explicit(FIVE_DENSE_BOUNDARY_MEMBERS),
        hit_percent: 50,
    },
    PackedCase {
        name: "at_length_gate_m5_span4096",
        ptype: PType::U32,
        bit_width: 13,
        len: 2_048,
        members: MemberSpec::Explicit(FIVE_DENSE_BOUNDARY_MEMBERS),
        hit_percent: 50,
    },
    PackedCase {
        name: "above_span_gate_m5_span4097",
        ptype: PType::U32,
        bit_width: 13,
        len: 2_048,
        members: MemberSpec::Explicit(FIVE_SPARSE_BOUNDARY_MEMBERS),
        hit_percent: 50,
    },
    PackedCase {
        name: "long_dense_m32",
        ptype: PType::U32,
        bit_width: 10,
        len: 65_536,
        members: MemberSpec::Stride {
            count: 32,
            stride: 2,
        },
        hit_percent: 50,
    },
    PackedCase {
        name: "long_sparse_m32",
        ptype: PType::U64,
        bit_width: 40,
        len: 65_536,
        members: MemberSpec::Stride {
            count: 32,
            stride: 10_000,
        },
        hit_percent: 50,
    },
    PackedCase {
        name: "zero_hit_m8",
        ptype: PType::U8,
        bit_width: 6,
        len: 65_536,
        members: MemberSpec::Stride {
            count: 8,
            stride: 2,
        },
        hit_percent: 0,
    },
    PackedCase {
        name: "full_hit_m8",
        ptype: PType::U16,
        bit_width: 12,
        len: 65_536,
        members: MemberSpec::Stride {
            count: 8,
            stride: 2,
        },
        hit_percent: 100,
    },
    PackedCase {
        name: "wide_packed_m8",
        ptype: PType::U32,
        bit_width: 31,
        len: 65_536,
        members: MemberSpec::Stride {
            count: 8,
            stride: 2,
        },
        hit_percent: 50,
    },
];

const OLD_GENERIC_CASES: &[PackedCase] = &[
    PACKED_CASES[0],
    PACKED_CASES[3],
    PACKED_CASES[5],
    PACKED_CASES[6],
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
    (0..case.len)
        .map(|index| {
            let is_hit = match case.hit_percent {
                0 => false,
                100 => true,
                percent => index % 100 < usize::from(percent),
            };
            if is_hit {
                members[index % members.len()]
            } else {
                let mut candidate = (index as u64 * 17 + 11) % domain_size;
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

fn packed_input<T: BenchInt>(case: PackedCase) -> (BitPackedArray, Scalar, VortexSession) {
    let session = array_session();
    vortex_fastlanes::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let members = case.members.values();
    let values: BufferMut<T> = generated_values(case, &members)
        .into_iter()
        .map(T::from_counter)
        .collect();
    let packed = page_aligned(
        BitPackedData::encode(
            &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
            case.bit_width,
            &mut ctx,
        )
        .unwrap(),
    );
    (packed, list_scalar::<T>(&members), session)
}

fn old_generic_contains(values: ArrayRef, list: &Scalar) -> ArrayRef {
    let false_scalar = Scalar::bool(false, values.dtype().nullability());
    let mut level = list
        .as_list()
        .elements()
        .vortex_expect("benchmark list is non-null")
        .iter()
        .map(|member| {
            Binary::try_new(
                ConstantArray::new(member.clone(), values.len()).into_array(),
                values.clone(),
                Operator::Eq,
            )
            .unwrap()
            .into_array()
            .fill_null(false_scalar.clone())
            .unwrap()
        })
        .collect::<Vec<_>>();

    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut arrays = level.into_iter();
        while let Some(left) = arrays.next() {
            next.push(if let Some(right) = arrays.next() {
                left.binary(right, Operator::Or).unwrap()
            } else {
                left
            });
        }
        level = next;
    }

    level.pop().vortex_expect("benchmark list is nonempty")
}

fn bench_packed_specialized<T: BenchInt>(bencher: Bencher, case: PackedCase) {
    let (packed, list, session) = packed_input::<T>(case);
    let contains = packed
        .into_array()
        .apply(&list_contains(lit(list), root()))
        .unwrap();
    let mut ctx = session.create_execution_ctx();
    bencher
        .counter(ItemsCount::new(case.len))
        .bench_local(|| black_box(contains.clone().execute::<BoolArray>(&mut ctx).unwrap()));
}

fn bench_packed_decode_once<T: BenchInt>(bencher: Bencher, case: PackedCase) {
    let (packed, list, session) = packed_input::<T>(case);
    let list = ConstantArray::new(list, case.len).into_array();
    let mut ctx = session.create_execution_ctx();
    bencher.counter(ItemsCount::new(case.len)).bench_local(|| {
        let primitive = packed
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let result = <Primitive as ListContainsElementKernel>::list_contains(
            &list,
            primitive.as_view(),
            &mut ctx,
        )
        .unwrap()
        .unwrap();
        black_box(result.execute::<BoolArray>(&mut ctx).unwrap())
    });
}

fn bench_packed_old_generic<T: BenchInt>(bencher: Bencher, case: PackedCase) {
    let (packed, list, session) = packed_input::<T>(case);
    let mut ctx = session.create_execution_ctx();
    bencher.counter(ItemsCount::new(case.len)).bench_local(|| {
        let result = old_generic_contains(packed.clone().into_array(), &list);
        black_box(result.execute::<BoolArray>(&mut ctx).unwrap())
    });
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

#[divan::bench(args = PACKED_CASES)]
fn packed_specialized(bencher: Bencher, case: PackedCase) {
    dispatch_packed!(bencher, case, bench_packed_specialized);
}

#[divan::bench(args = PACKED_CASES)]
fn packed_decode_once(bencher: Bencher, case: PackedCase) {
    dispatch_packed!(bencher, case, bench_packed_decode_once);
}

#[divan::bench(args = OLD_GENERIC_CASES)]
fn packed_old_generic(bencher: Bencher, case: PackedCase) {
    dispatch_packed!(bencher, case, bench_packed_old_generic);
}

#[cfg(not(codspeed))]
fn length_sweep_cases() -> Vec<PackedCase> {
    [
        2_048, 2_049, 2_304, 2_560, 3_072, 4_095, 4_096, 4_097, 6_144, 8_192,
    ]
    .map(|len| PackedCase {
        name: "length_sweep_m5_span4096",
        ptype: PType::U32,
        bit_width: 13,
        len,
        members: MemberSpec::Explicit(FIVE_DENSE_BOUNDARY_MEMBERS),
        hit_percent: 50,
    })
    .to_vec()
}

#[cfg(not(codspeed))]
#[divan::bench(args = length_sweep_cases())]
fn length_sweep_specialized(bencher: Bencher, case: PackedCase) {
    bench_packed_specialized::<u32>(bencher, case);
}

#[cfg(not(codspeed))]
#[divan::bench(args = length_sweep_cases())]
fn length_sweep_decode_once(bencher: Bencher, case: PackedCase) {
    bench_packed_decode_once::<u32>(bencher, case);
}

fn primitive_input<T: BenchInt>() -> (PrimitiveArray, Scalar, VortexSession) {
    const LEN: usize = 65_536;
    let case = PackedCase {
        name: "primitive",
        ptype: T::PTYPE,
        bit_width: 12,
        len: LEN,
        members: MemberSpec::Stride {
            count: 8,
            stride: 2,
        },
        hit_percent: 50,
    };
    let members = case.members.values();
    let values = generated_values(case, &members)
        .into_iter()
        .map(T::from_counter)
        .collect::<PrimitiveArray>();
    (values, list_scalar::<T>(&members), array_session())
}

macro_rules! primitive_benchmarks {
    ($module:ident, $T:ty) => {
        mod $module {
            use super::*;

            #[divan::bench]
            fn specialized(bencher: Bencher) {
                let (values, list, session) = primitive_input::<$T>();
                let len = values.len();
                let contains = values
                    .into_array()
                    .apply(&list_contains(lit(list), root()))
                    .unwrap();
                let mut ctx = session.create_execution_ctx();
                bencher.counter(ItemsCount::new(len)).bench_local(|| {
                    black_box(contains.clone().execute::<BoolArray>(&mut ctx).unwrap())
                });
            }

            #[divan::bench]
            fn old_generic(bencher: Bencher) {
                let (values, list, session) = primitive_input::<$T>();
                let len = values.len();
                let values = values.into_array();
                let mut ctx = session.create_execution_ctx();
                bencher.counter(ItemsCount::new(len)).bench_local(|| {
                    let result = old_generic_contains(values.clone(), &list);
                    black_box(result.execute::<BoolArray>(&mut ctx).unwrap())
                });
            }
        }
    };
}

primitive_benchmarks!(primitive_u8, u8);
primitive_benchmarks!(primitive_u16, u16);
primitive_benchmarks!(primitive_u32, u32);
primitive_benchmarks!(primitive_u64, u64);
