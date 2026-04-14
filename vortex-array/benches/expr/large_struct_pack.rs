// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::get_item;
use vortex_array::expr::pack;
use vortex_array::expr::root;

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
    bencher
        .with_inputs(|| (&pack_expr, &dtype))
        .bench_refs(|(pack_expr, dtype)| pack_expr.return_dtype(dtype).unwrap());
}
