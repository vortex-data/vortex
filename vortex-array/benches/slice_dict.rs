// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

const NUM_ROWS: usize = 65_536;
const NUM_UNIQUE: usize = 50;
const SLICE_LENGTH: usize = 8_192;

fn main() {
    divan::main();
}

fn make_string_dict(num_rows: usize, num_unique: usize) -> ArrayRef {
    let unique_strings: Vec<String> = (0..num_unique)
        .map(|i| format!("service-{i:04}-environment-name"))
        .collect();

    let values =
        VarBinViewArray::from_iter_nullable_str(unique_strings.iter().map(|s| Some(s.as_str())))
            .into_array();

    let codes: Vec<u32> = (0..num_rows).map(|i| (i % num_unique) as u32).collect();
    let codes_buf: Buffer<u32> = codes.into_iter().collect();
    let codes_arr = PrimitiveArray::new(codes_buf, Validity::NonNullable).into_array();

    unsafe { DictArray::new_unchecked(codes_arr, values).into_array() }
}

#[divan::bench(name = "slice_dict/native_slice")]
fn native_slice(bencher: Bencher) {
    let source = make_string_dict(NUM_ROWS, NUM_UNIQUE);

    bencher
        .counter(ItemsCount::new(SLICE_LENGTH))
        .with_inputs(|| 0..SLICE_LENGTH)
        // Returning the ArrayRef lets divan defer Drop until after the timed sample.
        .bench_refs(|range| source.slice(range.clone()));
}
