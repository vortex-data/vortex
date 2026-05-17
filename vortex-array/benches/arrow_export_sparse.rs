// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for exporting sparsely-referenced `DictArray` / `ListArray` / `ListViewArray` to
//! Arrow.
//!
//! The Vortex → Arrow path runs `execute_arrow` recursively on each child. For dicts it
//! materialises every dictionary value into an Arrow array; for list/listview it materialises
//! every element. When the live codes/offsets cover only a small slice of the underlying
//! buffer, that work is mostly wasted. These benches make the savings from the prune helpers
//! visible end-to-end on the Arrow path.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use arrow_schema::DataType;
use arrow_schema::Field;
use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const VECTOR_SIZE: usize = 2048;
const LIST_SIZE: usize = 8;

fn make_varbinview_dict(num_values: usize, referenced_fraction: f64) -> DictArray {
    let mut rng = StdRng::seed_from_u64(0);
    let referenced = ((num_values as f64) * referenced_fraction).max(1.0) as usize;
    let strings: Vec<String> = (0..num_values)
        .map(|i| {
            if i % 3 == 0 {
                format!("a-longer-string-value-{i:08}")
            } else {
                format!("s{i}")
            }
        })
        .collect();
    let values = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let codes =
        PrimitiveArray::from_iter((0..VECTOR_SIZE).map(|_| rng.random_range(0..referenced) as u32))
            .into_array();
    DictArray::try_new(codes, values).unwrap()
}

fn make_varbinview_listview(referenced_fraction: f64) -> ListViewArray {
    let mut rng = StdRng::seed_from_u64(0);
    let referenced = VECTOR_SIZE * LIST_SIZE;
    let element_count = ((referenced as f64) / referenced_fraction).max(1.0) as usize;
    let strings: Vec<String> = (0..element_count)
        .map(|i| {
            if i % 3 == 0 {
                format!("a-longer-string-value-{i:08}")
            } else {
                format!("s{i}")
            }
        })
        .collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let mut offsets: Vec<u32> = Vec::with_capacity(VECTOR_SIZE);
    let mut sizes: Vec<u32> = Vec::with_capacity(VECTOR_SIZE);
    let max_offset = element_count.saturating_sub(LIST_SIZE);
    for _ in 0..VECTOR_SIZE {
        offsets.push(rng.random_range(0..=max_offset.max(1)) as u32);
        sizes.push(LIST_SIZE as u32);
    }
    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
}

fn make_varbinview_list(referenced_fraction: f64) -> ListArray {
    let referenced = VECTOR_SIZE * LIST_SIZE;
    let element_count = ((referenced as f64) / referenced_fraction).max(1.0) as usize;
    let leading = (element_count - referenced) / 2;
    let strings: Vec<String> = (0..element_count)
        .map(|i| {
            if i % 3 == 0 {
                format!("a-longer-string-value-{i:08}")
            } else {
                format!("s{i}")
            }
        })
        .collect();
    let elements = VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();
    let offsets: Buffer<u32> = (0..=VECTOR_SIZE)
        .map(|i| (leading + i * LIST_SIZE) as u32)
        .collect();
    ListArray::try_new(elements, offsets.into_array(), Validity::NonNullable).unwrap()
}

const DICT_ARGS: [(usize, f64); 6] = [
    (16_384, 0.01),
    (16_384, 1.0),
    (131_072, 0.01),
    (131_072, 0.05),
    (131_072, 0.25),
    (131_072, 1.0),
];

#[divan::bench(args = DICT_ARGS)]
fn dict_varbinview(bencher: Bencher, (num_values, referenced_fraction): (usize, f64)) {
    let dict_type = DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8View));
    let field = Field::new("v", dict_type, false);
    bencher
        .with_inputs(|| make_varbinview_dict(num_values, referenced_fraction))
        .bench_values(|dict| {
            LEGACY_SESSION
                .arrow()
                .execute_arrow(
                    dict.into_array(),
                    Some(&field),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .unwrap()
        });
}

const LIST_ARGS: [f64; 4] = [0.01, 0.05, 0.25, 1.0];

#[divan::bench(args = LIST_ARGS)]
fn list_varbinview(bencher: Bencher, referenced_fraction: f64) {
    let elements_field = Field::new("item", DataType::Utf8View, false);
    let list_type = DataType::List(elements_field.into());
    let field = Field::new("v", list_type, false);
    bencher
        .with_inputs(|| make_varbinview_list(referenced_fraction))
        .bench_values(|list| {
            LEGACY_SESSION
                .arrow()
                .execute_arrow(
                    list.into_array(),
                    Some(&field),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .unwrap()
        });
}

#[divan::bench(args = LIST_ARGS)]
fn list_view_varbinview(bencher: Bencher, referenced_fraction: f64) {
    let elements_field = Field::new("item", DataType::Utf8View, false);
    let list_view_type = DataType::ListView(elements_field.into());
    let field = Field::new("v", list_view_type, false);
    bencher
        .with_inputs(|| make_varbinview_listview(referenced_fraction))
        .bench_values(|lv| {
            LEGACY_SESSION
                .arrow()
                .execute_arrow(
                    lv.into_array(),
                    Some(&field),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .unwrap()
        });
}
