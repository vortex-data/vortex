// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_dtype::{DType, FieldName, Nullability, PType, StructFields};
use vortex_expr::{get_item, pack, root};

fn main() {
    divan::main();
}

#[divan::bench(args = [100, 500, 1000, 2000])]
fn pack_return_dtype(bencher: Bencher, num_fields: usize) {
    // struct with many columns
    let field_names: Vec<FieldName> = (0..num_fields)
        .map(|i| FieldName::from(format!("col_{}", i)))
        .collect();
    let field_types = vec![DType::Primitive(PType::I64, Nullability::Nullable); num_fields];

    let struct_fields = StructFields::new(field_names.clone().into(), field_types);
    let dtype = DType::Struct(struct_fields, Nullability::NonNullable);

    let root_expr = root();
    let children: Vec<_> = field_names
        .iter()
        .map(|name| (name.clone(), get_item(name.clone(), root_expr.clone())))
        .collect();

    // pack(get_item(col) for col in cols)
    let pack_expr = pack(children, Nullability::Nullable);

    // return_dtype should be fast, it is assumed cheap in some expression simplifiers
    bencher.bench(|| pack_expr.return_dtype(&dtype).unwrap());
}
