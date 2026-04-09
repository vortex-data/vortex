// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

#[cfg(not(codspeed))]
mod benchmarks {
    use divan::Bencher;
    use divan::counter::BytesCount;
    use divan::counter::ItemsCount;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;
    use vortex_btrblocks::BtrBlocksCompressor;
    use vortex_buffer::buffer_mut;

    const NUM_ROWS: usize = 8192;
    const SEED: u64 = 42;

    const SHORT_STRINGS: &[&str] = &[
        "alpha_one",
        "bravo_two",
        "charlie_three",
        "delta_four",
        "echo_five",
        "foxtrot_six",
        "golf_seven",
        "hotel_eight",
        "india_nine",
        "juliet_ten",
    ];

    const LONG_STRINGS: &[&str] = &[
        "/path/to/some/deeply/nested/resource_a",
        "/path/to/some/deeply/nested/resource_b",
        "/path/to/another/location/item_c",
        "/data/archive/2024/collection/entry_d",
        "/data/archive/2024/collection/entry_e",
        "/workspace/project/src/module_f",
        "/workspace/project/src/module_g",
        "/workspace/project/test/fixture_h",
        "/var/log/service/output_i",
        "/tmp/scratch/workspace/temp_j",
    ];

    /// Wrap `elements` into a `ListViewArray` driven by per-entry `counts`.
    /// When `zctl`, offsets are sorted/non-overlapping. Otherwise, adjacent entries overlap by 1.
    fn wrap_listview(elements: ArrayRef, counts: &[usize], zctl: bool) -> ListViewArray {
        let mut offsets = buffer_mut![0u32; counts.len()];
        let mut sizes = buffer_mut![0u32; counts.len()];
        let mut offset = 0u32;

        for (i, &count) in counts.iter().enumerate() {
            // When !zctl, each entry (after the first) starts 1 element before the
            // previous entry ended, creating simple pairwise overlaps:
            //
            //   elements: [a, b, c, d, e, f, g, h, i]
            //   row 0:     ├────────┤              counts=[3, 3, 3]
            //   row 1:           ├────────┤        offsets=[0, 2, 5]
            //   row 2:                 ├────────┤  sizes  =[3, 4, 4]
            //                    ^           ^
            //                    shared      shared
            let overlap = if !zctl && i > 0 && offset > 0 { 1 } else { 0 };
            offsets[i] = offset - overlap;
            sizes[i] = count as u32 + overlap;
            offset += count as u32;
        }

        let mut lv = ListViewArray::new(
            elements,
            offsets.freeze().into_array(),
            sizes.freeze().into_array(),
            Validity::NonNullable,
        );
        if zctl {
            lv = unsafe { lv.with_zero_copy_to_list(true) };
        }
        lv
    }

    fn random_counts(
        rng: &mut StdRng,
        n: usize,
        range: std::ops::RangeInclusive<usize>,
    ) -> Vec<usize> {
        (0..n).map(|_| rng.random_range(range.clone())).collect()
    }

    fn random_i64_array(rng: &mut StdRng, len: usize, range: std::ops::Range<i64>) -> ArrayRef {
        let mut buf = buffer_mut![0i64; len];
        for v in buf.iter_mut() {
            *v = rng.random_range(range.clone());
        }
        buf.freeze().into_array()
    }

    fn random_str_array(rng: &mut StdRng, len: usize, pool: &[&str]) -> ArrayRef {
        let values: Vec<&str> = (0..len)
            .map(|_| pool[rng.random_range(0..pool.len())])
            .collect();
        VarBinViewArray::from_iter_str(values).into_array()
    }

    fn make_struct(names: impl Into<FieldNames>, fields: Vec<ArrayRef>, len: usize) -> ArrayRef {
        StructArray::try_new(names.into(), fields, len, Validity::NonNullable)
            .unwrap()
            .into_array()
    }

    /// Build the flat inner elements: `Struct<i64, Struct<utf8, utf8, i64>>`.
    fn build_inner_elements(rng: &mut StdRng, total_mid: usize) -> (ArrayRef, Vec<usize>) {
        let counts = random_counts(rng, total_mid, 1..=3);
        let n: usize = counts.iter().sum();

        let nested = make_struct(
            ["str_a", "str_b", "int_b"],
            vec![
                random_str_array(rng, n, SHORT_STRINGS),
                random_str_array(rng, n, LONG_STRINGS),
                random_i64_array(rng, n, 1..200),
            ],
            n,
        );
        let inner = make_struct(
            ["int_a", "nested"],
            vec![random_i64_array(rng, n, 1..500), nested],
            n,
        );
        (inner, counts)
    }

    /// Build the flat mid-level elements: `Struct<i64, utf8, ListView<Struct<...>>>`.
    fn build_mid_elements(rng: &mut StdRng, num_rows: usize, zctl: bool) -> (ArrayRef, Vec<usize>) {
        let outer_counts = random_counts(rng, num_rows, 3..=10);
        let n: usize = outer_counts.iter().sum();

        let (inner_elements, inner_counts) = build_inner_elements(rng, n);
        let inner_lv = wrap_listview(inner_elements, &inner_counts, zctl);

        let mid = make_struct(
            ["int_c", "str_c", "inner_list"],
            vec![
                random_i64_array(rng, n, 0x400000..0x7FFFFF),
                random_str_array(rng, n, LONG_STRINGS),
                inner_lv.into_array(),
            ],
            n,
        );
        (mid, outer_counts)
    }

    /// `ListView<Struct<i64, utf8, ListView<Struct<i64, Struct<utf8, utf8, i64>>>>>`
    fn build_nested_listview(num_rows: usize, layout: OffsetLayout) -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(SEED);
        let zctl = matches!(layout, OffsetLayout::Zctl);
        let (mid_elements, outer_counts) = build_mid_elements(&mut rng, num_rows, zctl);
        wrap_listview(mid_elements, &outer_counts, zctl).into_array()
    }

    #[derive(Debug, Clone, Copy)]
    enum OffsetLayout {
        Zctl,
        Overlapping,
    }

    #[divan::bench(args = [OffsetLayout::Zctl, OffsetLayout::Overlapping])]
    fn compress_listview(bencher: Bencher, layout: OffsetLayout) {
        let array = build_nested_listview(NUM_ROWS, layout);
        let nbytes = array.nbytes();
        let compressor = BtrBlocksCompressor::default();
        bencher
            .with_inputs(|| &array)
            .input_counter(|_| ItemsCount::new(NUM_ROWS))
            .input_counter(move |_| BytesCount::new(nbytes as usize))
            .bench_refs(|array| compressor.compress(array).unwrap());
    }
}

fn main() {
    divan::main()
}
