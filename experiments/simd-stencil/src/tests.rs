// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
