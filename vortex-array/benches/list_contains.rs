// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the Primitive constant-list membership dispatch paths.
//!
//! Primitive arrays use direct comparisons for up to four distinct members. They use binary
//! search from 10 members for 8- and 16-bit integers. The 32- and 64-bit thresholds are 11 and 13
//! members. Every path runs on each real CPU feature leg in CodSpeed.
//! To recalculate the thresholds, run this benchmark twice with temporary policy constants. Use a
//! high cutoff to force generic evaluation. Use `5` to force binary search above four members.
//!
//! Run with `cargo bench -p vortex-array --bench list_contains`.

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
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
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
struct PrimitiveCase {
    name: &'static str,
    len: usize,
    member_count: usize,
}

impl Display for PrimitiveCase {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}_m{}_n{}",
            self.name, self.member_count, self.len
        )
    }
}

const fn primitive_case(name: &'static str, len: usize, member_count: usize) -> PrimitiveCase {
    PrimitiveCase {
        name,
        len,
        member_count,
    }
}

const LONG_M1: PrimitiveCase = primitive_case("long", 65_536, 1);
const LONG_M4: PrimitiveCase = primitive_case("long", 65_536, 4);
const LONG_M9: PrimitiveCase = primitive_case("long", 65_536, 9);
const LONG_M10: PrimitiveCase = primitive_case("long", 65_536, 10);
const LONG_M11: PrimitiveCase = primitive_case("long", 65_536, 11);
const LONG_M12: PrimitiveCase = primitive_case("long", 65_536, 12);
const LONG_M13: PrimitiveCase = primitive_case("long", 65_536, 13);
const LONG_M32: PrimitiveCase = primitive_case("long", 65_536, 32);
const SHORT_M11: PrimitiveCase = primitive_case("short", 1_024, 11);
const SHORT_M13: PrimitiveCase = primitive_case("short", 1_024, 13);

const CURRENT_10: &[PrimitiveCase] = &[LONG_M1, LONG_M4, LONG_M9, LONG_M10, LONG_M32, SHORT_M11];
const CURRENT_11: &[PrimitiveCase] = &[LONG_M1, LONG_M4, LONG_M10, LONG_M11, LONG_M32, SHORT_M11];
const CURRENT_13: &[PrimitiveCase] = &[LONG_M1, LONG_M4, LONG_M12, LONG_M13, LONG_M32, SHORT_M13];

fn primitive_input<T: BenchInt>(
    case: PrimitiveCase,
) -> (PrimitiveArray, Scalar, BoolArray, VortexSession) {
    let members = (0..case.member_count)
        .map(|index| T::from_counter(u64::try_from(index).unwrap() * 2))
        .collect::<Vec<_>>();
    let domain_bits = T::PTYPE.bit_width().min(12);
    let domain_size = 1u64 << domain_bits;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let generated = (0..case.len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if (state >> 32).is_multiple_of(2) {
                let member_index =
                    usize::try_from(state % u64::try_from(members.len()).unwrap()).unwrap();
                members[member_index]
            } else {
                let mut candidate = state.rotate_left(17) % domain_size;
                while members.contains(&T::from_counter(candidate)) {
                    candidate = (candidate + 1) % domain_size;
                }
                T::from_counter(candidate)
            }
        })
        .collect::<Vec<_>>();
    let expected = BoolArray::from_iter(generated.iter().map(|value| members.contains(value)));
    let list = Scalar::list(
        Arc::new(DType::Primitive(T::PTYPE, Nullability::NonNullable)),
        members.iter().copied().map(Into::into).collect(),
        Nullability::NonNullable,
    );
    (
        PrimitiveArray::new::<T>(generated, Validity::NonNullable),
        list,
        expected,
        array_session(),
    )
}

fn bench_current<T: BenchInt>(bencher: Bencher, case: PrimitiveCase) {
    let (array, list, expected, session) = primitive_input::<T>(case);
    let expression = list_contains(lit(list), root());
    let mut ctx = session.create_execution_ctx();
    let actual = array
        .clone()
        .into_array()
        .apply(&expression)
        .unwrap()
        .execute::<BoolArray>(&mut ctx)
        .unwrap();
    assert_arrays_eq!(actual, expected, &mut ctx);

    bencher.counter(ItemsCount::new(case.len)).bench_local(|| {
        black_box(
            array
                .clone()
                .into_array()
                .apply(&expression)
                .unwrap()
                .execute::<BoolArray>(&mut ctx)
                .unwrap(),
        )
    });
}

macro_rules! primitive_benchmarks {
    ($type_name:ident, $ty:ty, $current:ident) => {
        mod $type_name {
            use super::*;

            #[vortex_bench_support::cpu_features]
            #[divan::bench(args = $current)]
            fn current(bencher: Bencher, case: PrimitiveCase) {
                bench_current::<$ty>(bencher, case);
            }
        }
    };
}

primitive_benchmarks!(u8_cases, u8, CURRENT_10);
primitive_benchmarks!(u16_cases, u16, CURRENT_10);
primitive_benchmarks!(u32_cases, u32, CURRENT_11);
primitive_benchmarks!(u64_cases, u64, CURRENT_13);
