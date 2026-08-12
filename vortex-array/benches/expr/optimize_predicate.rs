// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks `Expression::optimize_recursive` on lookup-style pushdown predicates:
//! an id membership test (either `list_contains` or a balanced OR of equalities) conjoined
//! with timestamp range bounds and kind filters.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::sync::Arc;

use divan::Bencher;
use divan::black_box;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::and_collect;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt_eq;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::not_eq;
use vortex_array::expr::or_collect;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;

fn main() {
    divan::main();
}

const FIELD_ID: &str = "id";
const FIELD_KIND: &str = "kind";
const FIELD_TIMESTAMP: &str = "timestamp";

const TIMESTAMP_BASE_SECS: i64 = 1_700_000_000;
const MICROS_PER_SECOND: i64 = 1_000_000;
const KIND_INCLUDED: u8 = 1;
const KIND_EXCLUDED: u8 = 2;

#[derive(Debug, Clone, Copy)]
enum IdPredicateShape {
    InList,
    BalancedOr,
}

impl Display for IdPredicateShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            IdPredicateShape::InList => write!(f, "in_list"),
            IdPredicateShape::BalancedOr => write!(f, "balanced_or"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PredicateCase {
    id_count: usize,
    shape: IdPredicateShape,
}

impl Display for PredicateCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "ids={}, shape={}", self.id_count, self.shape)
    }
}

const PREDICATE_CASES: &[PredicateCase] = &[
    PredicateCase {
        id_count: 1,
        shape: IdPredicateShape::InList,
    },
    PredicateCase {
        id_count: 1,
        shape: IdPredicateShape::BalancedOr,
    },
    PredicateCase {
        id_count: 16,
        shape: IdPredicateShape::InList,
    },
    PredicateCase {
        id_count: 16,
        shape: IdPredicateShape::BalancedOr,
    },
    PredicateCase {
        id_count: 64,
        shape: IdPredicateShape::InList,
    },
    PredicateCase {
        id_count: 64,
        shape: IdPredicateShape::BalancedOr,
    },
    PredicateCase {
        id_count: 256,
        shape: IdPredicateShape::InList,
    },
    PredicateCase {
        id_count: 256,
        shape: IdPredicateShape::BalancedOr,
    },
    PredicateCase {
        id_count: 1_024,
        shape: IdPredicateShape::InList,
    },
    PredicateCase {
        id_count: 1_024,
        shape: IdPredicateShape::BalancedOr,
    },
    PredicateCase {
        id_count: 4_096,
        shape: IdPredicateShape::InList,
    },
    PredicateCase {
        id_count: 4_096,
        shape: IdPredicateShape::BalancedOr,
    },
];

fn timestamp_dtype() -> DType {
    DType::Extension(Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased())
}

fn scope() -> DType {
    DType::Struct(
        StructFields::new(
            [FIELD_ID, FIELD_KIND, FIELD_TIMESTAMP].into(),
            vec![
                DType::Utf8(Nullability::NonNullable),
                DType::Primitive(PType::U8, Nullability::NonNullable),
                timestamp_dtype(),
            ],
        ),
        Nullability::NonNullable,
    )
}

fn id_strings(id_count: usize) -> Vec<String> {
    (0..id_count)
        .map(|idx| format!("00000000-0000-0000-0000-{idx:012x}"))
        .collect()
}

fn timestamp_lit(micros: i64) -> Expression {
    let storage = Scalar::primitive(micros, Nullability::NonNullable);
    lit(Scalar::extension_ref(
        Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased(),
        storage,
    ))
}

fn id_in_list_filter(ids: &[String]) -> Expression {
    let elements = ids
        .iter()
        .map(|id| Scalar::utf8(id.as_str(), Nullability::Nullable))
        .collect::<Vec<_>>();
    let list = Scalar::list(
        Arc::new(DType::Utf8(Nullability::Nullable)),
        elements,
        Nullability::Nullable,
    );
    list_contains(lit(list), col(FIELD_ID))
}

fn id_balanced_or_filter(ids: &[String]) -> Expression {
    or_collect(ids.iter().map(|id| eq(col(FIELD_ID), lit(id.as_str())))).unwrap()
}

fn id_filter(ids: &[String], shape: IdPredicateShape) -> Expression {
    match shape {
        IdPredicateShape::InList => id_in_list_filter(ids),
        IdPredicateShape::BalancedOr => id_balanced_or_filter(ids),
    }
}

fn lookup_predicate(predicate_case: PredicateCase) -> Expression {
    let ids = id_strings(predicate_case.id_count);
    let min_timestamp_us = TIMESTAMP_BASE_SECS * MICROS_PER_SECOND;
    let max_timestamp_us =
        (TIMESTAMP_BASE_SECS + 1) * MICROS_PER_SECOND + predicate_case.id_count as i64;

    let id_filter = and(
        id_filter(&ids, predicate_case.shape),
        not_eq(col(FIELD_KIND), lit(KIND_EXCLUDED)),
    );

    and_collect([
        id_filter,
        gt_eq(col(FIELD_TIMESTAMP), timestamp_lit(min_timestamp_us)),
        lt(col(FIELD_TIMESTAMP), timestamp_lit(max_timestamp_us)),
        eq(col(FIELD_KIND), lit(KIND_INCLUDED)),
    ])
    .unwrap()
}

#[divan::bench(args = PREDICATE_CASES)]
fn optimize_lookup_predicate(bencher: Bencher, predicate_case: &PredicateCase) {
    let scope = scope();
    let predicate = lookup_predicate(*predicate_case);

    bencher.bench(|| black_box(predicate.optimize_recursive(&scope)));
}
