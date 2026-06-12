// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpr;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::or;
use vortex_array::expr::root;

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

fn build_or_chain(n: usize, scope: &DType) -> BoundExpr {
    let base = eq(get_item("x", root(scope.clone())), lit(0i32));
    (1..n).fold(base, |acc, i| {
        or(acc, eq(get_item("x", root(scope.clone())), lit(i as i32)))
    })
}

#[divan::bench(args = [200])]
fn optimize_or_chain(bencher: Bencher, n: usize) {
    let scope = struct_scope();
    let expr = build_or_chain(n, &scope);
    bencher.bench(|| expr.optimize_recursive().unwrap());
}
