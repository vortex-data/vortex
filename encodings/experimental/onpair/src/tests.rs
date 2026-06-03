// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_session::VortexSession;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::OnPairMetadata;
use crate::compress::DEFAULT_DICT12_CONFIG;
use crate::compress::onpair_compress;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn sample_input() -> VarBinArray {
    VarBinArray::from_iter(
        [
            Some("https://www.example.com/page"),
            Some("https://www.example.com/data"),
            Some("https://www.test.org/page"),
            Some("ftp://files.example.com/x"),
            Some("https://www.example.com/page"),
        ],
        DType::Utf8(Nullability::NonNullable),
    )
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_metadata_golden() {
    check_metadata(
        "onpair.metadata",
        &OnPairMetadata {
            uncompressed_lengths_ptype: PType::I32 as i32,
            bits: 12,
            dict_size: 4096,
            total_tokens: 128_000,
            dict_offsets_ptype: PType::U32 as i32,
            codes_ptype: PType::U16 as i32,
            codes_offsets_ptype: PType::U32 as i32,
        }
        .encode_to_vec(),
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_roundtrip() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();

    let compressed = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).expect("compress");
    assert!(compressed.clone().into_array().is::<OnPair>());

    let mut ctx = SESSION.create_execution_ctx();
    let decoded = compressed
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .expect("canonicalize");

    decoded
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got.len(), 5);
            assert_eq!(
                got[0].as_deref(),
                Some(b"https://www.example.com/page".as_ref())
            );
            assert_eq!(
                got[3].as_deref(),
                Some(b"ftp://files.example.com/x".as_ref())
            );
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_nullable_canonicalize() {
    let input = VarBinArray::from_iter(
        [Some("a"), None, Some("bbb"), None, Some("ccccc")],
        DType::Utf8(Nullability::Nullable),
    );
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = arr
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    canonical
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got[1], None);
            assert_eq!(got[3], None);
            assert_eq!(got[4].as_deref(), Some(b"ccccc".as_ref()));
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_scalar_at() {
    let input = sample_input();
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    let s = arr.into_array().execute_scalar(2, &mut ctx).unwrap();
    let v = s.as_utf8().value().unwrap();
    assert_eq!(v.as_bytes(), b"https://www.test.org/page");
}

/// `scalar_at` must decode only the requested row's code window — fetching
/// its two `codes_offsets` boundaries via point lookup, not by materialising
/// the whole `codes_offsets`/`codes` children. Verify correctness at several
/// indices (including the last row) on a full array, and on a *sliced* array
/// where `codes_offsets` is itself a narrowed view and the row index is
/// relative to the slice.
#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_scalar_at_window() -> vortex_error::VortexResult<()> {
    let n = 2_000usize;
    let strings: Vec<String> = (0..n)
        .map(|i| format!("https://www.example.com/items/{i:08}/page?q={i}"))
        .collect();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)?.into_array();

    let mut ctx = SESSION.create_execution_ctx();
    for &i in &[0usize, 1, 999, 1000, n - 1] {
        let got = arr.execute_scalar(i, &mut ctx)?;
        assert_eq!(
            got.as_utf8().value().unwrap().as_bytes(),
            strings[i].as_bytes(),
            "full array row {i}"
        );
    }

    // Sliced array: `codes_offsets` is narrowed (first boundary > 0), so the
    // point lookup must resolve indices relative to the slice.
    let (start, end) = (700usize, 1300usize);
    let sliced = arr.slice(start..end)?;
    assert!(sliced.is::<OnPair>(), "slice dropped OnPair encoding");
    for &j in &[0usize, 1, 300, end - start - 1] {
        let got = sliced.execute_scalar(j, &mut ctx)?;
        assert_eq!(
            got.as_utf8().value().unwrap().as_bytes(),
            strings[start + j].as_bytes(),
            "sliced row {j}"
        );
    }
    Ok(())
}

/// The hot decode loop is 4×-unrolled with a scalar tail. Anything that
/// lands in the tail (1-3 leftover tokens, or zero total tokens) must
/// produce the same bytes as the unrolled body. Hit every row-count
/// near the boundary.
#[cfg_attr(miri, ignore)]
#[rstest::rstest]
#[case::n_1(1)]
#[case::n_2(2)]
#[case::n_3(3)]
#[case::n_4(4)]
#[case::n_5(5)]
#[case::n_7(7)]
#[case::n_8(8)]
#[case::n_9(9)]
fn test_onpair_unroll_tail_boundaries(#[case] n: usize) {
    let words: &[&str] = &["a", "bb", "ccc", "https://www.example.com/x"];
    let strings: Vec<&str> = (0..n).map(|i| words[i % words.len()]).collect();
    let input = VarBinArray::from_iter(
        strings.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = arr
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    canonical
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got.len(), n);
            for (i, expected) in strings.iter().enumerate() {
                assert_eq!(got[i].as_deref(), Some(expected.as_bytes()), "n={n}, i={i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

/// Empty array — the unroll path must short-circuit cleanly.
#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_empty() {
    let input = VarBinArray::from_iter(
        std::iter::empty::<Option<&str>>(),
        DType::Utf8(Nullability::NonNullable),
    );
    let len = input.len();
    let dtype = input.dtype().clone();
    let arr = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG).unwrap();
    assert_eq!(arr.len(), 0);
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = arr
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    assert_eq!(canonical.len(), 0);
}

/// Filter must share the dictionary — never recompress (this is the
/// regression cause on TPC-H Q22 SF=10). Exercise both selectivities
/// and check that the result is bit-exact and still an OnPairArray.
#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_filter_shares_dict() {
    let n = 5_000usize;
    let strings: Vec<String> = (0..n)
        .map(|i| format!("https://www.example.com/items/{i:08}"))
        .collect();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
    let dict_bytes_before = arr.dict_bytes().clone();
    let dict_offsets_len_before = arr.dict_offsets().len();

    // Keep every 7th row.
    let keep: Vec<bool> = (0..n).map(|i| i % 7 == 0).collect();
    let mask = vortex_mask::Mask::from_iter(keep.iter().copied());
    let expected: Vec<&str> = strings
        .iter()
        .enumerate()
        .filter_map(|(i, s)| keep[i].then_some(s.as_str()))
        .collect();

    let mut filter_ctx = SESSION.create_execution_ctx();
    let filtered = <OnPair as FilterKernel>::filter(arr.as_view(), &mask, &mut filter_ctx)
        .unwrap()
        .expect("OnPair filter must return Some");
    assert!(
        filtered.is::<OnPair>(),
        "filter dropped OnPair encoding: got {}",
        filtered.encoding_id()
    );
    let typed = filtered.try_downcast::<OnPair>().expect("OnPair");
    // Dict must be byte-identical with the input — no retrain, no copy.
    assert_eq!(typed.dict_bytes().as_slice(), dict_bytes_before.as_slice());
    assert_eq!(typed.dict_offsets().len(), dict_offsets_len_before);
    assert_eq!(typed.len(), expected.len());

    let mut ctx = SESSION.create_execution_ctx();
    let canonical = typed
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    canonical
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got.len(), expected.len());
            for (i, want) in expected.iter().enumerate() {
                assert_eq!(got[i].as_deref(), Some(want.as_bytes()), "row {i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

/// Rebuild an OnPair array, swapping `codes_offsets` for a narrowed
/// (smaller-ptype) primitive copy. Used by the narrowed-child
/// regression tests below.
#[expect(clippy::cognitive_complexity)]
fn narrow_codes_offsets(arr: &crate::OnPairArray, target: PType) -> crate::OnPairArray {
    let view = arr.as_view();
    let mut ctx = SESSION.create_execution_ctx();
    let original = view
        .codes_offsets()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();

    let narrowed_array = match_each_integer_ptype!(original.ptype(), |SRC| {
        let src = original.as_slice::<SRC>();
        match_each_integer_ptype!(target, |DST| {
            let mut buf = BufferMut::<DST>::with_capacity(src.len());
            for &v in src {
                #[allow(
                    clippy::unnecessary_cast,
                    reason = "macro-generated SRC may already be u64"
                )]
                buf.push(DST::try_from(v as u64).expect("value must fit in target ptype"));
            }
            PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array()
        })
    });

    unsafe {
        OnPair::new_unchecked(
            view.dtype().clone(),
            view.dict_bytes_handle().clone(),
            view.dict_offsets().clone(),
            view.codes().clone(),
            narrowed_array,
            view.uncompressed_lengths().clone(),
            view.array_validity(),
            view.bits(),
        )
    }
}

/// Regression: the cascading compressor can narrow `codes_offsets`
/// from u32 → u16 when every row's token count is small. The previous
/// `filter` impl read the child as `as_slice::<u32>()` and panicked
/// with `Other error: Attempted to get slice of type u32 from array
/// of type u16`. The fix dispatches via `match_each_integer_ptype!`.
#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_filter_with_narrowed_codes_offsets_u16() {
    let n = 200usize;
    // Short rows so per-row token counts stay small and codes_offsets
    // values fit in u16. (We narrow manually below regardless — this
    // matches the shape the cascading compressor produces in the
    // wild.)
    let strings: Vec<String> = (0..n).map(|i| format!("r{:03}", i)).collect();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();

    // Force `codes_offsets` to u16 so the panicking pre-fix
    // `as_slice::<u32>()` would fire.
    let arr = narrow_codes_offsets(&arr, PType::U16);
    assert_eq!(
        arr.as_view().codes_offsets().dtype().as_ptype(),
        PType::U16,
        "codes_offsets must be u16 to exercise the regression path"
    );

    let keep: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let mask = vortex_mask::Mask::from_iter(keep.iter().copied());
    let expected: Vec<&str> = strings
        .iter()
        .enumerate()
        .filter_map(|(i, s)| keep[i].then_some(s.as_str()))
        .collect();

    let mut filter_ctx = SESSION.create_execution_ctx();
    // Pre-fix: this call panics with "Attempted to get slice of type
    // u32 from array of type u16". Post-fix: succeeds.
    let filtered = <OnPair as FilterKernel>::filter(arr.as_view(), &mask, &mut filter_ctx)
        .unwrap()
        .expect("OnPair filter must return Some");
    let typed = filtered.try_downcast::<OnPair>().expect("OnPair");
    assert_eq!(typed.len(), expected.len());

    let mut ctx = SESSION.create_execution_ctx();
    let canonical = typed
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    canonical
        .with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got.len(), expected.len());
            for (i, want) in expected.iter().enumerate() {
                assert_eq!(got[i].as_deref(), Some(want.as_bytes()), "row {i}");
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
}

/// Same regression, narrowed to u8 (smallest possible ptype) — extra
/// coverage that the macro dispatch handles every integer ptype the
/// cascading compressor might pick.
#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_filter_with_narrowed_codes_offsets_u8() {
    let n = 100usize;
    let strings: Vec<String> = (0..n).map(|i| format!("{i}")).collect();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG).unwrap();
    let arr = narrow_codes_offsets(&arr, PType::U8);
    assert_eq!(arr.as_view().codes_offsets().dtype().as_ptype(), PType::U8);

    let mask = vortex_mask::Mask::from_iter((0..n).map(|i| i % 2 == 0));

    let mut filter_ctx = SESSION.create_execution_ctx();
    let filtered = <OnPair as FilterKernel>::filter(arr.as_view(), &mask, &mut filter_ctx)
        .unwrap()
        .expect("OnPair filter must return Some");
    assert_eq!(filtered.len(), n / 2);
}

/// Regression: canonicalising a *sliced* OnPair array. `slice` keeps the full
/// `codes` child and only narrows `codes_offsets`, so a sliced array has a
/// non-contiguous code window (`code_start > 0` and/or `code_end <
/// codes.len()`). `onpair_decode_views` must decode exactly that window;
/// decoding the whole `codes` stream — as a boundary-agnostic whole-column
/// decoder would — yields the wrong bytes (and over-runs the output) for any
/// partial slice. `filter` never produces this shape (it rebuilds `codes`
/// contiguously), so the existing filter tests do not cover it.
#[cfg_attr(miri, ignore)]
#[test]
fn test_onpair_slice_canonicalize() -> vortex_error::VortexResult<()> {
    let n = 5_000usize;
    let strings: Vec<String> = (0..n)
        .map(|i| format!("https://www.example.com/items/{i:08}"))
        .collect();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let arr =
        onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)?.into_array();

    // interior (start>0, end<n), LIMIT-like (start=0, end<n), tail (start>0,
    // end=n), and a near-full window.
    for (start, end) in [(1234usize, 1240usize), (0, 7), (4993, n), (1, n - 1)] {
        let sliced = arr.clone().slice(start..end)?;
        assert_eq!(sliced.len(), end - start);
        assert!(
            sliced.is::<OnPair>(),
            "slice dropped OnPair encoding: got {}",
            sliced.encoding_id()
        );

        let mut ctx = SESSION.create_execution_ctx();
        let canonical = sliced.execute::<VarBinViewArray>(&mut ctx)?;
        canonical.with_iterator(|iter| {
            let got: Vec<Option<Vec<u8>>> = iter.map(|b| b.map(|s| s.to_vec())).collect();
            assert_eq!(got.len(), end - start, "window {start}..{end} length");
            for (i, want) in strings[start..end].iter().enumerate() {
                assert_eq!(
                    got[i].as_deref(),
                    Some(want.as_bytes()),
                    "window {start}..{end} row {i}"
                );
            }
            Ok::<_, vortex_error::VortexError>(())
        })?;
    }
    Ok(())
}
