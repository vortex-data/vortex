// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures `VarBinScheme` against the same compressor with the scheme excluded, for both the
//! binary and the UTF-8 registration of the scheme.

#![allow(clippy::cast_possible_truncation, clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::binary::VarBinScheme;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const N: usize = 100_000;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn cases() -> Vec<(&'static str, ArrayRef)> {
    let mut s = 42u64;
    let random: Vec<Vec<u8>> = (0..N)
        .map(|_| (0..16).map(|_| (lcg(&mut s) >> 33) as u8).collect())
        .collect();
    let prefixed: Vec<Vec<u8>> = (0..N)
        .map(|i| format!("PREFIX_{i:09}").into_bytes())
        .collect();
    let mut s2 = 7u64;
    let wide: Vec<Vec<u8>> = (0..N)
        .map(|_| (0..256).map(|_| (lcg(&mut s2) >> 33) as u8).collect())
        .collect();

    let nullable = VarBinViewArray::from_iter(
        (0..N).map(|i| (i % 7 != 0).then(|| prefixed[i].as_slice())),
        DType::Binary(Nullability::Nullable),
    )
    .into_array();

    let mut s3 = 11u64;
    let strings: Vec<String> = (0..N)
        .map(|i| format!("row-{i:09}-{:x}", lcg(&mut s3) >> 33))
        .collect();
    let nullable_utf8 = VarBinViewArray::from_iter(
        (0..N).map(|i| (i % 7 != 0).then(|| strings[i].as_str())),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();

    vec![
        ("nulls every 7th", nullable),
        (
            "random 16B (hash)",
            VarBinViewArray::from_iter_bin(random.iter().map(|v| v.as_slice())).into_array(),
        ),
        (
            "shared prefix",
            VarBinViewArray::from_iter_bin(prefixed.iter().map(|v| v.as_slice())).into_array(),
        ),
        (
            "random 256B",
            VarBinViewArray::from_iter_bin(wide.iter().map(|v| v.as_slice())).into_array(),
        ),
        (
            "utf8 fixed width",
            VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array(),
        ),
        ("utf8 nulls every 7th", nullable_utf8),
    ]
}

#[test]
fn varbin_scheme_shrinks_output() -> VortexResult<()> {
    let with = BtrBlocksCompressorBuilder::default().build();
    let without = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([VarBinScheme::BINARY.id(), VarBinScheme::UTF8.id()])
        .build();

    println!(
        "{:<20}{:>12}{:>14}{:>14}{:>9}",
        "case", "input", "without", "with", "ratio"
    );
    for (name, array) in cases() {
        let a = {
            let mut ctx = SESSION.create_execution_ctx();
            without.compress(&array, &mut ctx)?.nbytes()
        };
        let b = {
            let mut ctx = SESSION.create_execution_ctx();
            with.compress(&array, &mut ctx)?.nbytes()
        };
        println!(
            "{:<20}{:>12}{:>14}{:>14}{:>9.2}",
            name,
            array.nbytes(),
            a,
            b,
            b as f64 / a as f64
        );

        assert!(
            b <= a,
            "{name}: enabling VarBinScheme grew the output, {a} -> {b}"
        );

        let mut ctx = SESSION.create_execution_ctx();
        let compressed = with.compress(&array, &mut ctx)?;
        let decoded = compressed
            .execute::<VarBinViewArray>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(&array, &decoded, &mut ctx);
    }
    Ok(())
}
