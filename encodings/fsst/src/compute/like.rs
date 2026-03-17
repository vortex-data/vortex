// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArray;
use crate::dfa::FsstMatcher;
use crate::dfa::dfa_scan_to_bitbuf;

impl LikeKernel for FSST {
    #[allow(clippy::cast_possible_truncation)]
    fn like(
        array: &FSSTArray,
        pattern: &ArrayRef,
        options: LikeOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };

        if options.case_insensitive {
            return Ok(None);
        }

        let Some(pattern_str) = pattern_scalar.as_utf8().value() else {
            return Ok(None);
        };

        let symbols = array.symbols();
        let symbol_lengths = array.symbol_lengths();

        let Some(matcher) =
            FsstMatcher::try_new(symbols.as_slice(), symbol_lengths.as_slice(), pattern_str)?
        else {
            return Ok(None);
        };

        let negated = options.negated;
        let codes = array.codes();
        let offsets = codes.offsets().to_primitive();
        let all_bytes = codes.bytes();
        let all_bytes = all_bytes.as_slice();
        let n = codes.len();

        let result = match_each_integer_ptype!(offsets.ptype(), |T| {
            let off = offsets.as_slice::<T>();
            dfa_scan_to_bitbuf(n, off, all_bytes, negated, |codes| matcher.matches(codes))
        });

        // FSST delegates validity to its codes array, so we can read it
        // directly without cloning the entire FSSTArray into an ArrayRef.
        let validity = array
            .codes()
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FSST;
    use crate::FSSTArray;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_fsst(strings: &[Option<&str>], nullability: Nullability) -> FSSTArray {
        let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
        let compressor = fsst_train_compressor(&varbin);
        fsst_compress(varbin, &compressor)
    }

    fn run_like(array: FSSTArray, pattern: &str, opts: LikeOptions) -> VortexResult<BoolArray> {
        let len = array.len();
        let arr = array.into_array();
        let pattern = ConstantArray::new(pattern, len).into_array();
        let result = Like
            .try_new_array(len, opts, [arr, pattern])?
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
        Ok(result.into_bool())
    }

    fn like(array: FSSTArray, pattern: &str) -> VortexResult<BoolArray> {
        run_like(array, pattern, LikeOptions::default())
    }

    #[test]
    fn test_like_prefix() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("http://example.com"),
                Some("http://test.org"),
                Some("ftp://files.net"),
                Some("http://vortex.dev"),
                Some("ssh://server.io"),
            ],
            Nullability::NonNullable,
        );
        let result = like(fsst, "http%")?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([true, true, false, true, false])
        );
        Ok(())
    }

    #[test]
    fn test_like_prefix_with_nulls() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("hello"), None, Some("help"), None, Some("goodbye")],
            Nullability::Nullable,
        );
        let result = like(fsst, "hel%")?; // spellchecker:disable-line
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(true), None, Some(false)])
        );
        Ok(())
    }

    #[test]
    fn test_like_contains() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("hello world"),
                Some("say hello"),
                Some("goodbye"),
                Some("hellooo"),
            ],
            Nullability::NonNullable,
        );
        let result = like(fsst, "%hello%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_contains_cross_symbol() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("the quick brown fox jumps over the lazy dog"),
                Some("a short string"),
                Some("the lazy dog sleeps"),
                Some("no match"),
            ],
            Nullability::NonNullable,
        );
        let result = like(fsst, "%lazy dog%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true, false]));
        Ok(())
    }

    #[test]
    fn test_not_like_contains() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("foobar_sdf"), Some("sdf_start"), Some("nothing")],
            Nullability::NonNullable,
        );
        let opts = LikeOptions {
            negated: true,
            case_insensitive: false,
        };
        let result = run_like(fsst, "%sdf%", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([false, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_match_all() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("abc"), Some(""), Some("xyz")],
            Nullability::NonNullable,
        );
        let result = like(fsst, "%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, true]));
        Ok(())
    }

    /// Call `LikeKernel::like` directly on the FSSTArray and verify it
    /// returns `Some(...)` (i.e. the kernel handles it, rather than
    /// returning `None` which would mean "fall back to decompress").
    #[test]
    fn test_like_prefix_kernel_handles() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("http://a.com"), Some("ftp://b.com")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("http%", fsst.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_some(), "FSST LikeKernel should handle prefix%");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Same direct-call check for the contains pattern `%needle%`.
    #[test]
    fn test_like_contains_kernel_handles() -> VortexResult<()> {
        let fsst = make_fsst(
            &[Some("hello world"), Some("goodbye")],
            Nullability::NonNullable,
        );
        let pattern = ConstantArray::new("%world%", fsst.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_some(), "FSST LikeKernel should handle %needle%");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false]));
        Ok(())
    }

    /// Patterns we can't handle should return `None` (fall back).
    #[test]
    fn test_like_kernel_falls_back_for_complex_pattern() -> VortexResult<()> {
        let fsst = make_fsst(&[Some("abc"), Some("def")], Nullability::NonNullable);
        let mut ctx = SESSION.create_execution_ctx();

        // Underscore wildcard -- not handled.
        let pattern = ConstantArray::new("a_c", fsst.len()).into_array();
        let result = <FSST as LikeKernel>::like(&fsst, &pattern, LikeOptions::default(), &mut ctx)?;
        assert!(result.is_none(), "underscore pattern should fall back");

        // Case-insensitive -- not handled.
        let pattern = ConstantArray::new("abc%", fsst.len()).into_array();
        let opts = LikeOptions {
            negated: false,
            case_insensitive: true,
        };
        let result = <FSST as LikeKernel>::like(&fsst, &pattern, opts, &mut ctx)?;
        assert!(result.is_none(), "ilike should fall back");

        Ok(())
    }

    #[test]
    fn test_like_long_prefix_falls_back_but_still_matches() -> VortexResult<()> {
        let fsst = make_fsst(
            &[
                Some("abcdefghijklmn-tail"),
                Some("abcdefghijklmx-tail"),
                Some("abcdefghijklmn"),
            ],
            Nullability::NonNullable,
        );
        let pattern = "abcdefghijklmn%";

        let direct = <FSST as LikeKernel>::like(
            &fsst,
            &ConstantArray::new(pattern, fsst.len()).into_array(),
            LikeOptions::default(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(
            direct.is_none(),
            "14-byte prefixes exceed the packed prefix DFA and should fall back"
        );

        let result = like(fsst, pattern)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_long_contains_falls_back_but_still_matches() -> VortexResult<()> {
        let needle = "a".repeat(255);
        let matching = format!("xx{needle}yy");
        let non_matching = format!("xx{}byy", "a".repeat(254));
        let exact = needle.clone();
        let pattern = format!("%{needle}%");

        let fsst = make_fsst(
            &[Some(&matching), Some(&non_matching), Some(&exact)],
            Nullability::NonNullable,
        );

        let direct = <FSST as LikeKernel>::like(
            &fsst,
            &ConstantArray::new(pattern.as_str(), fsst.len()).into_array(),
            LikeOptions::default(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(
            direct.is_none(),
            "contains needles longer than 254 bytes exceed the DFA's u8 state space"
        );

        let result = like(fsst, &pattern)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_contains_len_254_kernel_handles() -> VortexResult<()> {
        let needle = "a".repeat(254);
        let matching = format!("xx{needle}yy");
        let non_matching = format!("xx{}byy", "a".repeat(253));
        let pattern = format!("%{needle}%");

        let fsst = make_fsst(
            &[Some(&matching), Some(&non_matching), Some(needle.as_str())],
            Nullability::NonNullable,
        );

        let direct = <FSST as LikeKernel>::like(
            &fsst,
            &ConstantArray::new(pattern.as_str(), fsst.len()).into_array(),
            LikeOptions::default(),
            &mut SESSION.create_execution_ctx(),
        )?;
        assert!(direct.is_some(), "254-byte contains needle should stay on the DFA path");
        assert_arrays_eq!(direct.unwrap(), BoolArray::from_iter([true, false, true]));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Fuzz tests: compare FSST kernel against naive string matching
    // -----------------------------------------------------------------------

    fn random_string(rng: &mut StdRng, max_len: usize) -> String {
        let len = rng.random_range(0..=max_len);
        // Use a small alphabet to increase substring hit rate.
        (0..len)
            .map(|_| (b'a' + rng.random_range(0..6u8)) as char)
            .collect()
    }

    fn fuzz_contains(seed: u64, needle_len: usize, n_strings: usize) -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(seed);

        let needle: String = (0..needle_len)
            .map(|_| (b'a' + rng.random_range(0..6u8)) as char)
            .collect();

        let owned: Vec<String> = (0..n_strings)
            .map(|_| random_string(&mut rng, 80))
            .collect();
        let strings: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();

        let expected: Vec<bool> = owned.iter().map(|s| s.contains(&needle)).collect();

        let fsst = make_fsst(&strings, Nullability::NonNullable);
        let pattern = format!("%{needle}%");
        let result = run_like(fsst, &pattern, LikeOptions::default())?;

        let got: Vec<bool> = (0..n_strings)
            .map(|i| result.to_bit_buffer().value(i))
            .collect();

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e, g,
                "mismatch at index {i}: string={:?}, needle={needle:?}, expected={e}, got={g}",
                &owned[i],
            );
        }
        Ok(())
    }

    fn fuzz_prefix(seed: u64, prefix_len: usize, n_strings: usize) -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(seed);

        let prefix: String = (0..prefix_len)
            .map(|_| (b'a' + rng.random_range(0..6u8)) as char)
            .collect();

        let owned: Vec<String> = (0..n_strings)
            .map(|_| random_string(&mut rng, 80))
            .collect();
        let strings: Vec<Option<&str>> = owned.iter().map(|s| Some(s.as_str())).collect();

        let expected: Vec<bool> = owned.iter().map(|s| s.starts_with(&prefix)).collect();

        let fsst = make_fsst(&strings, Nullability::NonNullable);
        let pattern = format!("{prefix}%");
        let result = run_like(fsst, &pattern, LikeOptions::default())?;

        let got: Vec<bool> = (0..n_strings)
            .map(|i| result.to_bit_buffer().value(i))
            .collect();

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                e, g,
                "mismatch at index {i}: string={:?}, prefix={prefix:?}, expected={e}, got={g}",
                &owned[i],
            );
        }
        Ok(())
    }

    /// Fuzz contains with short needles (1-7 chars) -> BranchlessShiftDfa
    #[test]
    fn fuzz_contains_short_needle() -> VortexResult<()> {
        for seed in 0..50 {
            for needle_len in 1..=7 {
                fuzz_contains(seed, needle_len, 200)?;
            }
        }
        Ok(())
    }

    /// Fuzz contains with medium needles (8-14 chars) -> FlatBranchlessDfa
    #[test]
    fn fuzz_contains_medium_needle() -> VortexResult<()> {
        for seed in 0..50 {
            for needle_len in [8, 10, 14] {
                fuzz_contains(seed, needle_len, 200)?;
            }
        }
        Ok(())
    }

    /// Fuzz contains with long needles (>14 chars) -> FsstContainsDfa
    #[test]
    fn fuzz_contains_long_needle() -> VortexResult<()> {
        for seed in 0..30 {
            for needle_len in [15, 20, 30] {
                fuzz_contains(seed, needle_len, 200)?;
            }
        }
        Ok(())
    }

    /// Fuzz prefix matching
    #[test]
    fn fuzz_prefix_matching() -> VortexResult<()> {
        for seed in 0..50 {
            for prefix_len in [1, 3, 5, 10, 13] {
                fuzz_prefix(seed, prefix_len, 200)?;
            }
        }
        Ok(())
    }
}
