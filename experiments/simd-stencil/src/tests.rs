// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// `inspect_vortex_trees` prints encoding ids for diagnosis.
#![allow(clippy::use_debug)]

use crate::encode::encode_a;
use crate::encode::encode_b;
use crate::encode::encode_c;
use crate::encode::gen_f64;
use crate::encode::gen_u32;
use crate::patched;
use crate::strategies::aot;
use crate::strategies::fused;
use crate::strategies::materialized;

const N: usize = 8 * 1024;

#[test]
fn roundtrip_a() {
    let values = gen_u32(N, 1);
    let enc = encode_a(&values);
    assert_eq!(materialized::decode_a(&enc), values);
    assert_eq!(fused::decode_a(&enc), values);
    assert_eq!(aot::decode_a(&enc), values);
}

#[test]
fn roundtrip_b() {
    let values = gen_f64(N, 2, 7);
    let enc = encode_b(&values, 2);
    assert_eq!(materialized::decode_b(&enc), values);
    assert_eq!(fused::decode_b(&enc), values);
    assert_eq!(aot::decode_b(&enc), values);
    assert_eq!(patched::decode_b(&enc), values);
}

#[test]
fn roundtrip_c() {
    let enc = encode_c(4096, 2, 11);
    assert_eq!(materialized::decode_c(&enc), enc.values);
    assert_eq!(fused::decode_c(&enc), enc.values);
    assert_eq!(aot::decode_c(&enc), enc.values);
    assert_eq!(patched::decode_c(&enc), enc.values);
}

#[test]
fn vortex_baseline_decodes_same_data() {
    use crate::vortex_baseline;

    let values_u32 = gen_u32(N, 3);
    let arr_a = vortex_baseline::build_a(&values_u32);
    assert_eq!(vortex_baseline::decode_a(&arr_a), values_u32);

    let values_f64 = gen_f64(N, 2, 5);
    let arr_b = vortex_baseline::build_b(&values_f64);
    assert_eq!(vortex_baseline::decode_b(&arr_b), values_f64);
}

#[cfg(all(target_arch = "x86_64", unix))]
#[test]
fn stitched_affine_matches_aot() {
    use crate::stitched::StitchedAffine;
    use crate::stitched::affine_aot;

    // A multi-op affine pipeline with awkward constants.
    let ops = [(1.5f64, -3.25), (0.5, 100.0), (2.0, 0.125), (-1.0, 7.0)];
    let src: Vec<f64> = (0..1024).map(|i| i as f64 * 0.31 - 12.0).collect();

    let mut expected = vec![0f64; 1024];
    affine_aot(&ops, &src, &mut expected);

    let pipe = StitchedAffine::build(&ops);
    let mut got = vec![0f64; 1024];
    // SAFETY: both buffers hold 1024 (a multiple of 32) f64s.
    unsafe { pipe.run(src.as_ptr(), got.as_mut_ptr(), 1024) };

    // The stitched FMA loop matches the inlined `mul_add` reference bit-for-bit.
    assert_eq!(got, expected);
}

#[test]
fn vortex_same_stack_roundtrips() {
    use crate::vortex_baseline;

    // Stack A: genuine delta(bitpacking) decodes back to the input.
    let values_u32 = gen_u32(N, 3);
    let arr_a = vortex_baseline::build_a_same_stack(&values_u32);
    assert_eq!(
        vortex_baseline::decode(&arr_a).as_slice::<u32>(),
        values_u32
    );

    // Stack B integer core: genuine delta(ffor(bitpacking)) over the digits.
    let enc = encode_b(&gen_f64(N, 2, 5), 2);
    let arr_b = vortex_baseline::build_b_core_same_stack(&enc.digits);
    assert_eq!(
        vortex_baseline::decode(&arr_b).as_slice::<i64>(),
        enc.digits
    );

    // Stack B integer core, regular Vortex (shallow Delta).
    let arr_b_shallow = vortex_baseline::build_b_core_shallow(&enc.digits);
    assert_eq!(
        vortex_baseline::decode(&arr_b_shallow).as_slice::<i64>(),
        enc.digits
    );

    // Stack B full: genuine alp(delta(ffor(bitpacking))) decodes to the values.
    let values_f64 = gen_f64(N, 2, 5);
    let arr_b_full = vortex_baseline::build_b_full_same_stack(&values_f64);
    assert_eq!(vortex_baseline::decode_b(&arr_b_full), values_f64);

    // Stack C, regular Vortex (RunEnd of the logical column).
    let enc_c = encode_c(4096, 2, 11);
    let arr_c = vortex_baseline::build_c_regular(&enc_c.values);
    assert_eq!(vortex_baseline::decode_b(&arr_c), enc_c.values);
}

#[test]
fn inspect_vortex_trees() {
    use vortex_array::ArrayRef;

    use crate::vortex_baseline;

    fn dump(arr: &ArrayRef, depth: usize) {
        let pad = "  ".repeat(depth);
        println!(
            "{pad}{:?}  len={} nbytes={}",
            arr.encoding_id(),
            arr.len(),
            arr.nbytes()
        );
        let names = arr.children_names();
        for (i, child) in arr.children().iter().enumerate() {
            let name = names.get(i).map(String::as_str).unwrap_or("?");
            println!("{pad}  .{name}");
            dump(child, depth + 1);
        }
    }

    let n = 64 * 1024;
    println!("\n=== stack A: Vortex Delta of gen_u32 ===");
    dump(&vortex_baseline::build_a(&gen_u32(n, 3)), 0);
    println!("\n=== stack A: my encode_a (delta(bitpacking)) ===");
    let enc_a = encode_a(&gen_u32(n, 3));
    println!(
        "  tiles={} packed_u32_words={} widths(min/max)={}/{}",
        enc_a.n / 1024,
        enc_a.packed.len(),
        enc_a.width.iter().min().unwrap(),
        enc_a.width.iter().max().unwrap()
    );

    println!("\n=== stack B: Vortex ALP of gen_f64 ===");
    dump(&vortex_baseline::build_b(&gen_f64(n, 2, 5)), 0);
    println!("\n=== stack B: my encode_b (alp(delta(ffor(bitpacking)))) ===");
    let enc_b = encode_b(&gen_f64(n, 2, 5), 2);
    println!(
        "  tiles={} packed_u64_words={} widths(min/max)={}/{}",
        enc_b.n / 1024,
        enc_b.packed.len(),
        enc_b.width.iter().min().unwrap(),
        enc_b.width.iter().max().unwrap()
    );
}
