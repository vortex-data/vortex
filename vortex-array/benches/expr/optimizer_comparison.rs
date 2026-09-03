// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the legacy `Expression` optimizer with the rule-driven `BoundExpression` optimizer.
//!
//! Expressions are constructed and bound outside the timed region. The benchmarks vary tree size
//! and the number of nodes that the default optimizer rules can rewrite.

#![expect(clippy::unwrap_used)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpressionOptimizer;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::and_collect;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or_collect;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

fn struct_scope() -> DType {
    DType::Struct(
        StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    )
}

#[derive(Clone, Copy, Debug)]
struct RewriteCase {
    terms: usize,
    rewrite_sites: usize,
}

impl RewriteCase {
    fn node_count(self) -> usize {
        // Each term is `eq(get_item("x", root()), literal)`, the terms are joined by OR nodes,
        // and each rewrite site adds `and(term, true)`.
        5 * self.terms - 1 + 2 * self.rewrite_sites
    }
}

impl Display for RewriteCase {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "nodes={}, rewrites={}",
            self.node_count(),
            self.rewrite_sites
        )
    }
}

const REWRITE_CASES: &[RewriteCase] = &[
    RewriteCase {
        terms: 1,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 1,
        rewrite_sites: 1,
    },
    RewriteCase {
        terms: 16,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 16,
        rewrite_sites: 4,
    },
    RewriteCase {
        terms: 16,
        rewrite_sites: 16,
    },
    RewriteCase {
        terms: 128,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 128,
        rewrite_sites: 32,
    },
    RewriteCase {
        terms: 128,
        rewrite_sites: 128,
    },
    RewriteCase {
        terms: 512,
        rewrite_sites: 0,
    },
    RewriteCase {
        terms: 512,
        rewrite_sites: 128,
    },
    RewriteCase {
        terms: 512,
        rewrite_sites: 512,
    },
];

fn build_expression(case: RewriteCase) -> Expression {
    or_collect((0..case.terms).map(|idx| {
        let term = eq(col("x"), lit(i32::try_from(idx).unwrap()));
        if idx < case.rewrite_sites {
            and(term, lit(true))
        } else {
            term
        }
    }))
    .unwrap()
}

mod builtins {
    use super::*;

    #[divan::bench(args = REWRITE_CASES)]
    fn expression(bencher: Bencher, case: &RewriteCase) {
        let scope = struct_scope();
        let expr = build_expression(*case);

        bencher
            .counter(ItemsCount::new(case.node_count()))
            .bench(|| black_box(expr.optimize_recursive(&scope).unwrap()));
    }

    #[divan::bench(args = REWRITE_CASES)]
    fn bound_expression(bencher: Bencher, case: &RewriteCase) {
        let scope = struct_scope();
        let unbound = build_expression(*case);
        let expr = unbound.bind(&scope).unwrap();
        let optimizer = BoundExpressionOptimizer::default();

        let expected = unbound
            .optimize_recursive(&scope)
            .unwrap()
            .bind(&scope)
            .unwrap();
        assert_eq!(optimizer.optimize(&expr).unwrap(), expected);

        bencher
            .counter(ItemsCount::new(case.node_count()))
            .bench(|| black_box(optimizer.optimize(&expr).unwrap()));
    }
}

const CONJUNCTION_SIZES: &[usize] = &[16, 64, 128];

fn build_range_conjunction(pairs: usize) -> Expression {
    and_collect((0..pairs).flat_map(|idx| {
        let lower = i32::try_from(idx * 2).unwrap();
        [gt_eq(col("x"), lit(lower)), lt(col("x"), lit(lower + 1))]
    }))
    .unwrap()
}

fn build_no_range_conjunction(terms: usize) -> Expression {
    and_collect((0..terms).map(|idx| eq(col("x"), lit(i32::try_from(idx).unwrap())))).unwrap()
}

mod conjunctions {
    use super::*;

    #[divan::bench(args = CONJUNCTION_SIZES)]
    fn expression_ranges(bencher: Bencher, pairs: &usize) {
        let scope = struct_scope();
        let expr = build_range_conjunction(*pairs);

        bencher
            .counter(ItemsCount::new(10 * *pairs - 1))
            .bench(|| black_box(expr.optimize_recursive(&scope).unwrap()));
    }

    #[divan::bench(args = CONJUNCTION_SIZES)]
    fn bound_expression_ranges(bencher: Bencher, pairs: &usize) {
        let scope = struct_scope();
        let unbound = build_range_conjunction(*pairs);
        let expr = unbound.bind(&scope).unwrap();
        let optimizer = BoundExpressionOptimizer::default();

        assert_eq!(
            optimizer.optimize(&expr).unwrap(),
            unbound
                .optimize_recursive(&scope)
                .unwrap()
                .bind(&scope)
                .unwrap()
        );

        bencher
            .counter(ItemsCount::new(10 * *pairs - 1))
            .bench(|| black_box(optimizer.optimize(&expr).unwrap()));
    }

    #[divan::bench(args = CONJUNCTION_SIZES)]
    fn expression_no_ranges(bencher: Bencher, terms: &usize) {
        let scope = struct_scope();
        let expr = build_no_range_conjunction(*terms);

        bencher
            .counter(ItemsCount::new(5 * *terms - 1))
            .bench(|| black_box(expr.optimize_recursive(&scope).unwrap()));
    }

    #[divan::bench(args = CONJUNCTION_SIZES)]
    fn bound_expression_no_ranges(bencher: Bencher, terms: &usize) {
        let scope = struct_scope();
        let unbound = build_no_range_conjunction(*terms);
        let expr = unbound.bind(&scope).unwrap();
        let optimizer = BoundExpressionOptimizer::default();

        assert_eq!(
            optimizer.optimize(&expr).unwrap(),
            unbound
                .optimize_recursive(&scope)
                .unwrap()
                .bind(&scope)
                .unwrap()
        );

        bencher
            .counter(ItemsCount::new(5 * *terms - 1))
            .bench(|| black_box(optimizer.optimize(&expr).unwrap()));
    }
}
