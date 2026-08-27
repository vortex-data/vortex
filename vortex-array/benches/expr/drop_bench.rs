// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::and_collect;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::not;
use vortex_array::expr::root;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// A predicate of the shape and depth that a query usually has.
fn shallow() -> Expression {
    and(
        not(eq(get_item("status", root()), lit("inactive"))),
        gt(get_item("score", root()), lit(75i32)),
    )
}

/// A balanced conjunction of `n` predicates, which is the shape [`and_collect`] builds.
fn balanced(n: usize) -> Expression {
    and_collect((0..n).map(|i| eq(get_item("x", root()), lit(i as i32)))).unwrap()
}

/// A conjunction of `n` predicates nested to the left, which gives a tree of depth `n`.
fn deep_chain(n: usize) -> Expression {
    (1..n)
        .map(|i| eq(get_item("x", root()), lit(i as i32)))
        .fold(eq(get_item("x", root()), lit(0i32)), and)
}

#[divan::bench(sample_count = 500)]
fn drop_shallow(bencher: Bencher) {
    bencher.with_inputs(shallow).bench_local_values(drop);
}

#[divan::bench(args = [8, 1024], sample_count = 500)]
fn drop_balanced(bencher: Bencher, n: usize) {
    bencher.with_inputs(|| balanced(n)).bench_local_values(drop);
}

#[divan::bench(args = [64, 1024], sample_count = 500)]
fn drop_deep_chain(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| deep_chain(n))
        .bench_local_values(drop);
}
