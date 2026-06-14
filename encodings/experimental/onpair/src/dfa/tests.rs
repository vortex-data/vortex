// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// The randomised cross-check uses a tiny LCG whose `u64 -> usize` reductions are
// bounded well below `usize::MAX`; the truncation lint is noise here.
#![allow(clippy::cast_possible_truncation)]

use std::sync::LazyLock;
use std::sync::atomic::Ordering::Relaxed;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::OnPair;
use crate::OnPairArray;
use crate::compress::DEFAULT_DICT12_CONFIG;
use crate::compress::onpair_compress;
use crate::compute::like::PUSHDOWN_HITS;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn make_onpair(strings: &[Option<&str>], nullability: Nullability) -> OnPairArray {
    let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    onpair_compress(&varbin, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap()
}

fn run_like(array: OnPairArray, pattern: &str, opts: LikeOptions) -> VortexResult<BoolArray> {
    let len = array.len();
    let arr = array.into_array();
    let pattern = ConstantArray::new(pattern, len).into_array();
    let result = Like
        .try_new_array(len, opts, [arr, pattern])?
        .into_array()
        .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
    Ok(result.into_bool())
}

fn like(array: OnPairArray, pattern: &str) -> VortexResult<BoolArray> {
    run_like(array, pattern, LikeOptions::default())
}

#[test]
fn test_like_prefix() -> VortexResult<()> {
    let onpair = make_onpair(
        &[
            Some("http://example.com"),
            Some("http://test.org"),
            Some("ftp://files.net"),
            Some("http://vortex.dev"),
            Some("ssh://server.io"),
        ],
        Nullability::NonNullable,
    );
    let result = like(onpair, "http%")?;
    assert_arrays_eq!(
        &result,
        &BoolArray::from_iter([true, true, false, true, false])
    );
    Ok(())
}

#[test]
fn test_like_prefix_with_nulls() -> VortexResult<()> {
    let onpair = make_onpair(
        &[Some("hello"), None, Some("help"), None, Some("goodbye")],
        Nullability::Nullable,
    );
    let result = like(onpair, "hel%")?; // spellchecker:disable-line
    assert_arrays_eq!(
        &result,
        &BoolArray::from_iter([Some(true), None, Some(true), None, Some(false)])
    );
    Ok(())
}

#[test]
fn test_like_contains() -> VortexResult<()> {
    let onpair = make_onpair(
        &[
            Some("hello world"),
            Some("say hello"),
            Some("goodbye"),
            Some("hellooo"),
        ],
        Nullability::NonNullable,
    );
    let result = like(onpair, "%hello%")?;
    assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false, true]));
    Ok(())
}

#[test]
fn test_like_contains_cross_token() -> VortexResult<()> {
    let onpair = make_onpair(
        &[
            Some("the quick brown fox jumps over the lazy dog"),
            Some("a short string"),
            Some("the lazy dog sleeps"),
            Some("no match"),
        ],
        Nullability::NonNullable,
    );
    let result = like(onpair, "%lazy dog%")?;
    assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true, false]));
    Ok(())
}

#[test]
fn test_not_like_contains() -> VortexResult<()> {
    let onpair = make_onpair(
        &[Some("foobar_sdf"), Some("sdf_start"), Some("nothing")],
        Nullability::NonNullable,
    );
    let opts = LikeOptions {
        negated: true,
        case_insensitive: false,
    };
    let result = run_like(onpair, "%sdf%", opts)?;
    assert_arrays_eq!(&result, &BoolArray::from_iter([false, false, true]));
    Ok(())
}

#[test]
fn test_like_match_all() -> VortexResult<()> {
    let onpair = make_onpair(
        &[Some("abc"), Some(""), Some("xyz")],
        Nullability::NonNullable,
    );
    let result = like(onpair, "%")?;
    assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, true]));
    Ok(())
}

/// Call `LikeKernel::like` directly and verify it returns `Some(...)` (i.e. the
/// kernel handles it, rather than returning `None` = "fall back to decompress").
#[test]
fn test_like_prefix_kernel_handles() -> VortexResult<()> {
    let onpair = make_onpair(
        &[Some("http://a.com"), Some("ftp://b.com")],
        Nullability::NonNullable,
    );
    let pattern = ConstantArray::new("http%", onpair.len()).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let onpair = onpair.as_view();
    let result = <OnPair as LikeKernel>::like(onpair, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(result.is_some(), "OnPair LikeKernel should handle prefix%");
    assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
    Ok(())
}

#[test]
fn test_like_contains_kernel_handles() -> VortexResult<()> {
    let onpair = make_onpair(
        &[Some("hello world"), Some("goodbye")],
        Nullability::NonNullable,
    );
    let pattern = ConstantArray::new("%world%", onpair.len()).into_array();
    let mut ctx = SESSION.create_execution_ctx();

    let onpair = onpair.as_view();
    let result = <OnPair as LikeKernel>::like(onpair, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(result.is_some(), "OnPair LikeKernel should handle %needle%");
    assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
    Ok(())
}

/// Patterns we can't handle should return `None` (fall back).
#[test]
fn test_like_kernel_falls_back_for_complex_pattern() -> VortexResult<()> {
    let onpair = make_onpair(&[Some("abc"), Some("def")], Nullability::NonNullable);
    let mut ctx = SESSION.create_execution_ctx();

    // Underscore wildcard -- not handled.
    let pattern = ConstantArray::new("a_c", onpair.len()).into_array();
    let onpair_v = onpair.as_view();
    let result =
        <OnPair as LikeKernel>::like(onpair_v, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(result.is_none(), "underscore pattern should fall back");

    // Case-insensitive -- not handled.
    let pattern = ConstantArray::new("abc%", onpair.len()).into_array();
    let opts = LikeOptions {
        negated: false,
        case_insensitive: true,
    };
    let result = <OnPair as LikeKernel>::like(onpair_v, &pattern, opts, &mut ctx)?;
    assert!(result.is_none(), "ilike should fall back");

    // Suffix patterns are unsupported, even when the suffix is an escaped literal.
    let pattern = ConstantArray::new(r"%\%", onpair.len()).into_array();
    let result =
        <OnPair as LikeKernel>::like(onpair_v, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(result.is_none(), "escaped suffix pattern should fall back");

    Ok(())
}

#[test]
fn test_like_kernel_handles_escaped_prefix_and_contains() -> VortexResult<()> {
    let onpair = make_onpair(
        &[
            Some("%front"),
            Some("_front"),
            Some("\\front"),
            Some("middle%value"),
            Some("middle_value"),
            Some("middle\\value"),
            Some("front"),
        ],
        Nullability::NonNullable,
    );
    let onpair_v = onpair.as_view();
    let mut ctx = SESSION.create_execution_ctx();

    let pattern = ConstantArray::new(r"\%%", onpair.len()).into_array();
    let result =
        <OnPair as LikeKernel>::like(onpair_v, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(result.is_some(), "escaped percent prefix should use OnPair");
    assert_arrays_eq!(
        result.unwrap(),
        BoolArray::from_iter([true, false, false, false, false, false, false])
    );

    let pattern = ConstantArray::new(r"%\_%", onpair.len()).into_array();
    let result =
        <OnPair as LikeKernel>::like(onpair_v, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(
        result.is_some(),
        "escaped underscore contains should use OnPair"
    );
    assert_arrays_eq!(
        result.unwrap(),
        BoolArray::from_iter([false, true, false, false, true, false, false])
    );

    let pattern = ConstantArray::new(r"%\\%", onpair.len()).into_array();
    let result =
        <OnPair as LikeKernel>::like(onpair_v, &pattern, LikeOptions::default(), &mut ctx)?;
    assert!(
        result.is_some(),
        "escaped backslash contains should use OnPair"
    );
    assert_arrays_eq!(
        result.unwrap(),
        BoolArray::from_iter([false, false, true, false, false, true, false])
    );

    Ok(())
}

/// Longer multi-token needle that spans many dictionary tokens, validating the
/// per-code lift over a realistic dictionary.
#[test]
fn test_like_contains_long_needle() -> VortexResult<()> {
    let rows: Vec<Option<String>> = (0..256)
        .map(|i| Some(format!("https://www.example.com/path/{}/segment", i % 17)))
        .collect();
    let refs: Vec<Option<&str>> = rows.iter().map(|s| s.as_deref()).collect();
    let onpair = make_onpair(&refs, Nullability::NonNullable);

    let result = like(onpair, "%example.com/path%")?;
    assert_arrays_eq!(&result, &BoolArray::from_iter(vec![true; 256]));
    Ok(())
}

/// Randomised cross-check: the compressed-domain DFA must agree with
/// ground-truth `starts_with` / `contains` over the original strings, across
/// many rows and many needles. This is the soundness guard — a row's
/// concatenated token bytes are its decompressed bytes, so the DFA result must
/// match a byte-level match exactly.
#[test]
fn test_like_matches_ground_truth_fuzz() -> VortexResult<()> {
    // Small deterministic LCG so the test is reproducible without extra deps.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let alphabet = b"abcde /.:";
    let rng_string = |n: usize, r: &mut dyn FnMut() -> u64| -> String {
        (0..n)
            .map(|_| alphabet[(r() as usize) % alphabet.len()] as char)
            .collect()
    };

    let rows: Vec<String> = (0..600)
        .map(|_| {
            let len = 1 + (next() as usize) % 24;
            rng_string(len, &mut next)
        })
        .collect();
    let refs: Vec<Option<&str>> = rows.iter().map(|s| Some(s.as_str())).collect();
    let onpair = make_onpair(&refs, Nullability::NonNullable);

    // A mix of prefix and contains needles, including ones unlikely to match.
    let needles = [
        "a", "ab", "abc", "/", ".", ":", "a/", "/.", "de", "cde", "abcde", "zz", "a.b", " ",
    ];
    for needle in needles {
        for (kind, pattern) in [
            ("prefix", format!("{needle}%")),
            ("contains", format!("%{needle}%")),
        ] {
            let got = like(onpair.clone(), &pattern)?;
            let expected: Vec<bool> = rows
                .iter()
                .map(|s| {
                    if kind == "prefix" {
                        s.starts_with(needle)
                    } else {
                        s.contains(needle)
                    }
                })
                .collect();
            assert_arrays_eq!(&got, &BoolArray::from_iter(expected.clone()));
        }
    }
    Ok(())
}

/// Prove the pushdown *fires through the execution engine* — not just when the
/// kernel is called directly — for a bare OnPair array, a `Dict(OnPair)`, and a
/// `Dict(Shared(OnPair))` (the shape a dict-encoded column takes when read back
/// from a file, where the layout wraps shared dictionary values in `Shared`).
///
/// Correct results alone can't distinguish "pushdown ran" from "decompressed and
/// matched", so this asserts on the kernel's hit counter.
#[test]
fn test_pushdown_fires_through_dict_and_shared() -> VortexResult<()> {
    let values = make_onpair(
        &[
            Some("https://google.com"),
            Some("http://yandex.ru"),
            Some("https://google.com/maps"),
        ],
        Nullability::NonNullable,
    );
    // 5 rows referencing the 3 dictionary values; values.len() <= codes.len() so
    // Dict's LikeReduce pushes the predicate down to the values.
    let codes = buffer![0u8, 1, 2, 0, 1].into_array();
    // `%google%` over the 3 dictionary values, then over the 5 dict rows.
    let expected_values = BoolArray::from_iter([true, false, true]);
    let expected_rows = BoolArray::from_iter([true, false, true, true, false]);

    let run = |arr: ArrayRef| -> VortexResult<(BoolArray, usize)> {
        let before = PUSHDOWN_HITS.load(Relaxed);
        let len = arr.len();
        let pattern = ConstantArray::new("%google%", len).into_array();
        // Optimize first (as the engine does): this runs Dict's LIKE reduce,
        // which pushes the predicate down to the dictionary values.
        let result = Like
            .try_new_array(len, LikeOptions::default(), [arr, pattern])?
            .into_array()
            .optimize()?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_bool();
        Ok((result, PUSHDOWN_HITS.load(Relaxed) - before))
    };

    // (a) bare OnPair — the predicate dispatches straight to our kernel.
    let (ra, hits_a) = run(values.clone().into_array())?;
    assert_arrays_eq!(&ra, &expected_values);

    // (b) Dict(OnPair) — Dict::like pushes the predicate to the OnPair values.
    let dict = DictArray::try_new(codes.clone(), values.clone().into_array())?;
    let (rb, hits_b) = run(dict.into_array())?;
    assert_arrays_eq!(&rb, &expected_rows);

    // (c) Dict(Shared(OnPair)) — the read-back shape.
    let shared = SharedArray::new(values.into_array()).into_array();
    let dict = DictArray::try_new(codes, shared)?;
    let (rc, hits_c) = run(dict.into_array())?;
    assert_arrays_eq!(&rc, &expected_rows);

    eprintln!("pushdown hits: bare={hits_a} dict={hits_b} dict_shared={hits_c}");

    // Bare OnPair and Dict(OnPair) both route the predicate to our kernel.
    assert!(hits_a >= 1, "bare OnPair LIKE should fire the pushdown");
    assert!(hits_b >= 1, "Dict(OnPair) LIKE should fire the pushdown");

    // At the *array* level, `LIKE` over `Dict(Shared(OnPair))` still does NOT push
    // down: `Shared` has no parent-reduce forwarding, so the predicate
    // canonicalizes (decompresses) the source instead of reaching the OnPair
    // kernel. The fix is at the *layout* level: the dict reader applies predicates
    // to the bare (non-`Shared`) values so a column read back from a multi-chunk
    // file fires the pushdown (verified end-to-end) — see
    // `vortex-layout`'s `DictReader::values_array_uncanonical`. This guard pins the
    // array-level behavior that motivates that layout choice.
    assert_eq!(
        hits_c, 0,
        "Dict(Shared(OnPair)) unexpectedly fired the pushdown ({hits_c}); if Shared \
         now forwards LIKE to its source, update this characterization assertion"
    );

    Ok(())
}

/// A `%needle%` longer than the contains DFA's state space must fall back.
#[test]
fn test_like_long_contains_falls_back() -> VortexResult<()> {
    let needle = "a".repeat(255);
    let matching = format!("xx{needle}yy");
    let pattern = format!("%{needle}%");

    let onpair = make_onpair(&[Some(&matching)], Nullability::NonNullable);
    let onpair_v = onpair.as_view();
    let direct = <OnPair as LikeKernel>::like(
        onpair_v,
        &ConstantArray::new(pattern.as_str(), onpair.len()).into_array(),
        LikeOptions::default(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert!(
        direct.is_none(),
        "contains needles longer than 254 bytes exceed the DFA's u8 state space"
    );

    // ...but the generic fallback path still produces the right answer.
    let result = like(onpair, &pattern)?;
    assert_arrays_eq!(&result, &BoolArray::from_iter([true]));
    Ok(())
}
