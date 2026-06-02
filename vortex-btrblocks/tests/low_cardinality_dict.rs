// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Low-cardinality dictionary integration: a primitive column of {0, 1, 2} with nulls
//! (the canonical four-state case) should compress to a *sorted* dictionary, and
//! compute over the compressed dict must agree with compute over the canonical column.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const N: usize = 100_000;

/// Build an i32 column whose non-null values are drawn from {0, 1, 2}, with ~10% nulls.
fn four_state_column() -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(0);
    let values: Vec<i32> = (0..N).map(|_| rng.random_range(0i32..3)).collect();
    let validity_bits: Vec<bool> = (0..N).map(|_| rng.random_range(0u8..10) != 0).collect();
    PrimitiveArray::new(
        values.into_iter().collect::<vortex_buffer::Buffer<i32>>(),
        Validity::from(BitBuffer::from(validity_bits)),
    )
}

/// Run `column < 2` and return the number of matching (true) rows.
fn lt_two_true_count(array: &vortex_array::ArrayRef) -> usize {
    let mut ctx = SESSION.create_execution_ctx();
    array
        .clone()
        .binary(
            ConstantArray::new(2i32, array.len()).into_array(),
            Operator::Lt,
        )
        .unwrap()
        .execute::<Mask>(&mut ctx)
        .unwrap()
        .true_count()
}

#[test]
fn low_cardinality_compresses_to_sorted_dict() {
    let mut ctx = SESSION.create_execution_ctx();
    let column = four_state_column();
    let raw_nbytes = column.clone().into_array().nbytes();

    // Default btrblocks: dict is available and should win on this column.
    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor
        .compress(&column.clone().into_array(), &mut ctx)
        .unwrap();

    // The chosen encoding is a (sorted) dictionary.
    let dict = compressed
        .as_opt::<Dict>()
        .expect("low-cardinality column should compress to a Dict");
    assert!(
        dict.has_sorted_values(),
        "btrblocks dict must emit sorted values so codes are order-preserving"
    );

    // Baseline: compress with the integer dict scheme excluded, forcing a non-dict
    // (primitive bitpack / FoR / RLE) encoding.
    let no_dict = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([IntDictScheme.id()])
        .build();
    let compressed_prim = no_dict
        .compress(&column.clone().into_array(), &mut ctx)
        .unwrap();

    let dict_nbytes = compressed.nbytes();
    let prim_nbytes = compressed_prim.nbytes();

    println!("--- low-cardinality compression ({N} rows, values {{0,1,2}} + ~10% nulls) ---");
    println!("raw i32              : {raw_nbytes:>9} bytes");
    println!(
        "btrblocks (dict)     : {dict_nbytes:>9} bytes  ({:.1}x vs raw)",
        raw_nbytes as f64 / dict_nbytes as f64
    );
    println!(
        "btrblocks (no dict)  : {prim_nbytes:>9} bytes  ({:.1}x vs raw)",
        raw_nbytes as f64 / prim_nbytes as f64
    );

    // Compute over compressed must match compute over the canonical column.
    let canonical = column.into_array();
    let expected = lt_two_true_count(&canonical);
    assert_eq!(lt_two_true_count(&compressed), expected);
    assert_eq!(lt_two_true_count(&compressed_prim), expected);
}
