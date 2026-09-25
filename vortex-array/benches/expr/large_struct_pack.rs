// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::pack;
use vortex_array::expr::bound::root;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

#[divan::bench(args = [100, 200, 350, 500])]
fn pack_construct_bound(bencher: Bencher, num_fields: usize) {
    // struct with many columns
    let field_names: Vec<FieldName> = (0..num_fields)
        .map(|i| FieldName::from(format!("col_{}", i)))
        .collect();
    let field_types = vec![DType::Primitive(PType::I64, Nullability::Nullable); num_fields];

    let struct_fields = StructFields::new(field_names.clone().into(), field_types);
    let dtype = DType::Struct(struct_fields, Nullability::NonNullable);

    let root_expr = root(dtype);
    let children: Vec<_> = field_names
        .iter()
        .map(|name| (name.clone(), get_item(name.clone(), root_expr.clone())))
        .collect();

    bencher.bench(|| pack(children.clone(), Nullability::Nullable));
}
