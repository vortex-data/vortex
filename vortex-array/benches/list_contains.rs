// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the Primitive constant-list membership dispatch paths.
//!
//! Primitive arrays use direct comparisons for at most four distinct integer members. Larger sets
//! use the frozen generic path. Every path runs on each real CPU feature leg in CodSpeed.
//!
//! Run with `cargo bench -p vortex-array --bench list_contains`.

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
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
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
const LONG_M5: PrimitiveCase = primitive_case("long", 65_536, 5);
const SHORT_M4: PrimitiveCase = primitive_case("short", 1_024, 4);

const CURRENT: &[PrimitiveCase] = &[LONG_M1, LONG_M4, LONG_M5, SHORT_M4];

fn primitive_input<T: BenchInt>(
    case: PrimitiveCase,
) -> (PrimitiveArray, Vec<T>, BoolArray, VortexSession) {
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
    (
        PrimitiveArray::new::<T>(generated, Validity::NonNullable),
        members,
        expected,
        array_session(),
    )
}

fn list_scalar<T: BenchInt>(members: &[T]) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(T::PTYPE, Nullability::NonNullable)),
        members.iter().copied().map(Into::into).collect(),
        Nullability::NonNullable,
    )
}

fn frozen_generic_membership<T: BenchInt>(
    values: ArrayRef,
    members: &[T],
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
            let member: Scalar = (*member).into();
            Binary::try_new(
                ConstantArray::new(member, len).into_array(),
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
    values: &PrimitiveArray,
    members: &[T],
    ctx: &mut vortex_array::ExecutionCtx,
) -> BoolArray {
    frozen_generic_membership(values.clone().into_array(), members)
        .unwrap()
        .execute::<BoolArray>(ctx)
        .unwrap()
}

fn bench_current<T: BenchInt>(bencher: Bencher, case: PrimitiveCase) {
    let (array, members, expected, session) = primitive_input::<T>(case);
    let contains = array
        .into_array()
        .apply(&list_contains(lit(list_scalar(&members)), root()))
        .unwrap();
    let mut ctx = session.create_execution_ctx();
    let actual = contains.clone().execute::<BoolArray>(&mut ctx).unwrap();
    assert_arrays_eq!(actual, expected, &mut ctx);

    bencher
        .counter(ItemsCount::new(case.len))
        .bench_local(|| black_box(contains.clone().execute::<BoolArray>(&mut ctx).unwrap()));
}

fn bench_generic_baseline<T: BenchInt>(bencher: Bencher, case: PrimitiveCase) {
    let (array, members, expected, session) = primitive_input::<T>(case);
    let mut ctx = session.create_execution_ctx();
    let actual = execute_generic_baseline(&array, &members, &mut ctx);
    assert_arrays_eq!(actual, expected, &mut ctx);

    // The frozen pre-change implementation built this array tree during execution.
    bencher
        .counter(ItemsCount::new(case.len))
        .bench_local(|| black_box(execute_generic_baseline(&array, &members, &mut ctx)));
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

            #[vortex_bench_support::cpu_features]
            #[divan::bench(args = $current)]
            fn generic_baseline(bencher: Bencher, case: PrimitiveCase) {
                bench_generic_baseline::<$ty>(bencher, case);
            }
        }
    };
}

primitive_benchmarks!(u8_cases, u8, CURRENT);
primitive_benchmarks!(u16_cases, u16, CURRENT);
primitive_benchmarks!(u32_cases, u32, CURRENT);
primitive_benchmarks!(u64_cases, u64, CURRENT);
