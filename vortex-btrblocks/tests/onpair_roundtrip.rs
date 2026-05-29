// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! End-to-end round-trip through the full Vortex compressor + decompressor
//! on string arrays. Lives in `vortex-btrblocks` (gated on `unstable_encodings`)
//! so it exercises the same code path the file writer uses, not just the
//! OnPair crate in isolation.

#![cfg(feature = "unstable_encodings")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::tests_outside_test_module,
    clippy::use_debug
)]

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Helper: synthetic short-string corpus that the cascading compressor should
/// route through OnPair.
fn corpus(n: usize) -> Vec<String> {
    let templates: &[&str] = &[
        "https://www.example.com/products/{id}",
        "https://cdn.example.com/img/{id}.webp",
        "https://api.example.com/v2/orders/{id}",
        "https://www.example.com/users/{id}/profile",
        "INFO  request_id={id} status=200 method=GET",
        "WARN  request_id={id} status=429 method=POST",
        "ERROR request_id={id} status=500 method=PUT",
    ];
    let mut out = Vec::with_capacity(n);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (state as usize) % templates.len();
        let id = state as u32;
        out.push(templates[pick].replace("{id}", &format!("{:08x}", id)));
    }
    out
}

#[test]
fn nonnullable_roundtrip_via_default_compressor() {
    let n = 4096;
    let strings = corpus(n);
    let array = VarBinViewArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();

    let compressed = BtrBlocksCompressor::default()
        .compress(&array, &mut SESSION.create_execution_ctx())
        .expect("compress");
    // Don't assert a specific scheme — both OnPair and FSST are registered and
    // the sample-based selector keeps whichever is smaller. What matters is the
    // round-trip.

    let decoded = compressed
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
        .expect("decompress");
    assert_eq!(decoded.len(), n);
    decoded
        .with_iterator(|iter| {
            for (i, got) in iter.enumerate() {
                assert_eq!(
                    got,
                    Some(strings[i].as_bytes()),
                    "mismatch at row {i}: got {:?}",
                    got.map(|b| String::from_utf8_lossy(b).into_owned()),
                );
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

#[test]
fn nullable_roundtrip_via_default_compressor() {
    let n = 2048;
    let strings: Vec<Option<String>> = corpus(n)
        .into_iter()
        .enumerate()
        .map(|(i, s)| (i % 7 != 0).then_some(s))
        .collect();

    let array = VarBinViewArray::from_iter(
        strings.iter().map(|s| s.as_deref()),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();

    let compressed = BtrBlocksCompressor::default()
        .compress(&array, &mut SESSION.create_execution_ctx())
        .expect("compress");
    // Don't assert OnPair specifically here — the sample-based selector may
    // pick a different scheme on tiny inputs. What matters is the round-trip.

    let decoded = compressed
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
        .expect("decompress");
    assert_eq!(decoded.len(), n);
    decoded
        .with_iterator(|iter| {
            for (i, got) in iter.enumerate() {
                let want = strings[i].as_deref().map(str::as_bytes);
                assert_eq!(got, want, "mismatch at row {i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

/// Larger corpus that exercises the offsets-narrowing / delta-encoding paths
/// the cascading compressor enables past 2048 entries. The decoder must
/// reconstruct absolute u32 offsets from whatever encoded shape the
/// compressor chose for each child.
#[test]
fn large_unique_short_strings_roundtrip() {
    let n = 1 << 13; // 8192 rows, all unique, short.
    let strings: Vec<String> = (0..n).map(|i| format!("k{i:05x}")).collect();
    let array = VarBinViewArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();

    let compressed = BtrBlocksCompressor::default()
        .compress(&array, &mut SESSION.create_execution_ctx())
        .expect("compress");

    let decoded = compressed
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
        .expect("decompress");
    assert_eq!(decoded.len(), n);
    decoded
        .with_iterator(|iter| {
            for (i, got) in iter.enumerate() {
                assert_eq!(got, Some(strings[i].as_bytes()), "row {i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

#[test]
fn empty_and_short_string_roundtrip() {
    // Edge cases: empty strings interleaved with short ones.
    let strings = vec!["", "a", "", "bb", "ccc", "", "dddd", "eeeee", ""];
    let array = VarBinViewArray::from_iter(
        strings.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();

    let compressed = BtrBlocksCompressor::default()
        .compress(&array, &mut SESSION.create_execution_ctx())
        .expect("compress");
    let decoded = compressed
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
        .expect("decompress");
    decoded
        .with_iterator(|iter| {
            let got: Vec<_> = iter.collect();
            for (i, want) in strings.iter().enumerate() {
                assert_eq!(got[i], Some(want.as_bytes()), "row {i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

/// Regression for the Euro2016 compress-bench panic
/// (`onpair::decompress`: "dictionary offsets must be nondecreasing").
///
/// A large, high-cardinality corpus fills the OnPair dictionary toward its
/// 4096-entry cap, so the cascading compressor narrows `dict_offsets` to `u16`
/// and Delta-encodes it across multiple FastLanes chunks (len > 1024). The old
/// decode path widened it via `arr.cast(u32).execute()`, but the `Delta` cast
/// kernel preserves the Delta wrapping and only widens the inner bases/deltas
/// in place — and the transposed bases layout is keyed on `T::LANES`, which
/// differs between `u16` and `u32`. Decoding the widened Delta against the
/// misaligned layout yields non-monotonic offsets and trips the upstream
/// assert. The fix canonicalises each child to a `PrimitiveArray` first, then
/// widens element-wise.
#[test]
fn delta_dict_offsets_roundtrip() {
    let n = 1usize << 16;
    // Hex-encoded index plus a hashed suffix: every row is unique with enough
    // shared structure to route through OnPair while filling the dictionary.
    let strings: Vec<String> = (0..n)
        .map(|i| format!("{i:016x}-{:08x}", i.wrapping_mul(2654435761)))
        .collect();
    let array = VarBinViewArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let compressed = BtrBlocksCompressor::default()
        .compress(&array, &mut SESSION.create_execution_ctx())
        .expect("compress");
    let decoded = compressed
        .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
        .expect("decompress");
    assert_eq!(decoded.len(), n);
    decoded
        .with_iterator(|iter| {
            for (i, got) in iter.enumerate() {
                assert_eq!(got, Some(strings[i].as_bytes()), "row {i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}
