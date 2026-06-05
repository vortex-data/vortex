// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! `LIKE` pushdown for OnPair-compressed strings.
//!
//! OnPair already ships compressed-domain search automata: the `onpair` crate
//! exposes a token-level prefix DFA and a KMP-style contains automaton via
//! [`SearchParts::search`]. Both evaluate a pattern over the row-concatenated
//! `codes` stream without ever decoding a row, so we can answer
//! `col LIKE 'prefix%'` and `col LIKE '%needle%'` straight off the encoded
//! children.
//!
//! Pushdown is conservative: anything other than a constant `prefix%` or
//! `%needle%` pattern (the `_` wildcard, interior `%`, bare suffixes, exact
//! matches, or `ILIKE`) returns `None` so the caller falls back to
//! decompression + the Arrow `LIKE` kernel.

use std::borrow::Cow;

use onpair::Pattern;
use onpair::SearchParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::decode::collect_widened;

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only constant, case-sensitive patterns can be pushed down.
        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };
        if options.case_insensitive {
            return Ok(None);
        }

        let pattern_bytes: &[u8] = if let Some(s) = pattern_scalar.as_utf8_opt() {
            let Some(v) = s.value() else {
                return Ok(None);
            };
            v.as_ref()
        } else if let Some(b) = pattern_scalar.as_binary_opt() {
            let Some(v) = b.value() else {
                return Ok(None);
            };
            v
        } else {
            return Ok(None);
        };

        let Some(kind) = LikeKind::parse(pattern_bytes) else {
            return Ok(None);
        };

        // onpair's contains automaton stores match-progress states in a `u8`,
        // so it caps the needle at 255 bytes (and panics beyond that). Fall
        // back to decompression for longer `%needle%` literals. The prefix
        // automaton is token-level and has no such bound.
        if let LikeKind::Contains(needle) = &kind
            && needle.len() > MAX_CONTAINS_NEEDLE_LEN
        {
            return Ok(None);
        }

        // Widen the encoded children to the native widths the `onpair` search
        // API consumes. The cascading compressor may have narrowed any of these
        // integer children on disk; `collect_widened` casts them back.
        let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;
        let codes = collect_widened::<u16>(array.codes(), ctx)?;
        // `codes_offsets` are absolute indices into the full `codes` child, so a
        // sliced array (non-zero first offset) is searched correctly as-is.
        let codes_offsets = collect_widened::<u32>(array.codes_offsets(), ctx)?;
        let n = codes_offsets.len().saturating_sub(1);

        let negated = options.negated;
        let (needle, is_contains) = match kind {
            // `prefix%` / `%needle%` with an empty literal matches every row;
            // the encoded children never need to be touched.
            LikeKind::MatchAll => {
                let result = BitBuffer::collect_bool(n, |_| !negated);
                let validity = array
                    .array_validity()
                    .union_nullability(pattern_scalar.dtype().nullability());
                return Ok(Some(BoolArray::new(result, validity).into_array()));
            }
            LikeKind::Prefix(needle) => (needle, false),
            LikeKind::Contains(needle) => (needle, true),
        };

        // The over-padded `dict_bytes` buffer is fine: search only reads
        // `dict_offsets`-bounded token ranges, never the trailing pad.
        let parts = SearchParts::<u32> {
            dict_bytes: array.dict_bytes().as_slice(),
            dict_offsets: dict_offsets.as_slice(),
            codes: codes.as_slice(),
            code_offsets: codes_offsets.as_slice(),
            first_codes: None,
        };
        let onpair_pattern = if is_contains {
            Pattern::Contains(needle.as_ref())
        } else {
            Pattern::Prefix(needle.as_ref())
        };
        let mask = parts.search(onpair_pattern);
        let words = mask.as_words();
        let result = BitBuffer::collect_bool(n, |i| {
            let set = (words[i >> 6] >> (i & 63)) & 1 == 1;
            set != negated
        });

        // A null input row stays null in the result; the outer validity child
        // carries that, unioned with the pattern's nullability.
        let validity = array
            .array_validity()
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}

/// Longest `%needle%` literal the onpair contains automaton accepts. Its match
/// states are `u8`-indexed, so a needle of more than 255 bytes overflows the
/// state space (and onpair asserts on it).
const MAX_CONTAINS_NEEDLE_LEN: usize = 255;

/// The subset of `LIKE` patterns OnPair can evaluate in the compressed domain.
enum LikeKind<'a> {
    /// `prefix%` / `%needle%` whose literal is empty — matches every row.
    MatchAll,
    /// `prefix%`
    Prefix(Cow<'a, [u8]>),
    /// `%needle%`
    Contains(Cow<'a, [u8]>),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a [u8]) -> Option<Self> {
        let kind = Self::parse_prefix(pattern)
            .map(LikeKind::Prefix)
            .or_else(|| Self::parse_contains(pattern).map(LikeKind::Contains))?;
        Some(match kind {
            LikeKind::Prefix(p) | LikeKind::Contains(p) if p.is_empty() => LikeKind::MatchAll,
            other => other,
        })
    }

    fn parse_prefix(pattern: &'a [u8]) -> Option<Cow<'a, [u8]>> {
        Self::parse_literal_until_final_percent(pattern, 0)
    }

    fn parse_contains(pattern: &'a [u8]) -> Option<Cow<'a, [u8]>> {
        if !pattern.starts_with(b"%") {
            return None;
        }
        Self::parse_literal_until_final_percent(pattern, 1)
    }

    /// Parse `pattern[literal_start..]` as a literal terminated by a single
    /// trailing `%`, honouring `\` escapes. Returns `None` if a `_` wildcard or
    /// a non-final `%` is encountered (those need the fallback path).
    fn parse_literal_until_final_percent(
        pattern: &'a [u8],
        literal_start: usize,
    ) -> Option<Cow<'a, [u8]>> {
        let mut literal: Option<Vec<u8>> = None;
        let mut idx = literal_start;
        while idx < pattern.len() {
            match pattern[idx] {
                b'\\' => {
                    // A trailing `\` is treated as a literal backslash.
                    let escaped = pattern.get(idx + 1).copied().unwrap_or(b'\\');
                    literal
                        .get_or_insert_with(|| pattern[literal_start..idx].to_vec())
                        .push(escaped);
                    idx = (idx + 2).min(pattern.len());
                }
                b'%' if idx + 1 == pattern.len() => {
                    return Some(match literal {
                        Some(buf) => Cow::Owned(buf),
                        None => Cow::Borrowed(&pattern[literal_start..idx]),
                    });
                }
                b'%' | b'_' => return None,
                byte => {
                    if let Some(literal) = &mut literal {
                        literal.push(byte);
                    }
                    idx += 1;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::OnPair;
    use crate::OnPairArray;
    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn make_onpair(strings: &[Option<&str>], nullability: Nullability) -> OnPairArray {
        let varbin = VarBinArray::from_iter(strings.iter().copied(), DType::Utf8(nullability));
        let len = varbin.len();
        let dtype = varbin.dtype().clone();
        onpair_compress(&varbin, len, &dtype, DEFAULT_DICT12_CONFIG).expect("compress")
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

    /// Direct kernel call so we can assert the DFA path actually handled the
    /// pattern (`Some`) rather than silently falling back (`None`).
    fn like_kernel(
        array: &OnPairArray,
        pattern: &str,
        opts: LikeOptions,
    ) -> VortexResult<Option<BoolArray>> {
        let pat = ConstantArray::new(pattern, array.len()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = <OnPair as LikeKernel>::like(array.as_view(), &pat, opts, &mut ctx)?;
        Ok(result.map(|r| {
            r.execute::<Canonical>(&mut SESSION.create_execution_ctx())
                .expect("canonicalize")
                .into_bool()
        }))
    }

    #[test]
    fn test_like_prefix() -> VortexResult<()> {
        let arr = make_onpair(
            &[
                Some("http://example.com"),
                Some("http://test.org"),
                Some("ftp://files.net"),
                Some("http://vortex.dev"),
                Some("ssh://server.io"),
            ],
            Nullability::NonNullable,
        );
        let result = like(arr, "http%")?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([true, true, false, true, false])
        );
        Ok(())
    }

    #[test]
    fn test_like_prefix_with_nulls() -> VortexResult<()> {
        let arr = make_onpair(
            &[Some("hello"), None, Some("help"), None, Some("goodbye")],
            Nullability::Nullable,
        );
        let result = like(arr, "hel%")?; // spellchecker:disable-line
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(true), None, Some(true), None, Some(false)])
        );
        Ok(())
    }

    #[test]
    fn test_like_contains() -> VortexResult<()> {
        let arr = make_onpair(
            &[
                Some("hello world"),
                Some("say hello"),
                Some("goodbye"),
                Some("hellooo"),
            ],
            Nullability::NonNullable,
        );
        let result = like(arr, "%hello%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_contains_cross_token() -> VortexResult<()> {
        let arr = make_onpair(
            &[
                Some("the quick brown fox jumps over the lazy dog"),
                Some("a short string"),
                Some("the lazy dog sleeps"),
                Some("no match"),
            ],
            Nullability::NonNullable,
        );
        let result = like(arr, "%lazy dog%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true, false]));
        Ok(())
    }

    #[test]
    fn test_not_like_contains() -> VortexResult<()> {
        let arr = make_onpair(
            &[Some("foobar_sdf"), Some("sdf_start"), Some("nothing")],
            Nullability::NonNullable,
        );
        let opts = LikeOptions {
            negated: true,
            case_insensitive: false,
        };
        let result = run_like(arr, "%sdf%", opts)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([false, false, true]));
        Ok(())
    }

    #[test]
    fn test_like_match_all() -> VortexResult<()> {
        let arr = make_onpair(
            &[Some("abc"), Some(""), Some("xyz")],
            Nullability::NonNullable,
        );
        let result = like(arr, "%")?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, true, true]));
        Ok(())
    }

    #[test]
    fn test_kernel_handles_prefix_and_contains() -> VortexResult<()> {
        let arr = make_onpair(
            &[Some("http://a.com"), Some("ftp://b.com")],
            Nullability::NonNullable,
        );

        let prefix = like_kernel(&arr, "http%", LikeOptions::default())?;
        assert!(prefix.is_some(), "OnPair should push down prefix%");
        assert_arrays_eq!(prefix.unwrap(), BoolArray::from_iter([true, false]));

        let contains = like_kernel(&arr, "%//b%", LikeOptions::default())?;
        assert!(contains.is_some(), "OnPair should push down %needle%");
        assert_arrays_eq!(contains.unwrap(), BoolArray::from_iter([false, true]));
        Ok(())
    }

    #[test]
    fn test_kernel_falls_back_for_unsupported_patterns() -> VortexResult<()> {
        let arr = make_onpair(&[Some("abc"), Some("def")], Nullability::NonNullable);

        // Underscore wildcard.
        assert!(
            like_kernel(&arr, "a_c", LikeOptions::default())?.is_none(),
            "underscore pattern should fall back"
        );
        // Bare suffix.
        assert!(
            like_kernel(&arr, "%bc", LikeOptions::default())?.is_none(),
            "suffix pattern should fall back"
        );
        // Exact match (no wildcard).
        assert!(
            like_kernel(&arr, "abc", LikeOptions::default())?.is_none(),
            "exact pattern should fall back"
        );
        // Case-insensitive.
        let ci = LikeOptions {
            negated: false,
            case_insensitive: true,
        };
        assert!(
            like_kernel(&arr, "abc%", ci)?.is_none(),
            "ilike should fall back"
        );
        Ok(())
    }

    #[test]
    fn test_kernel_handles_escaped_prefix_and_contains() -> VortexResult<()> {
        let arr = make_onpair(
            &[
                Some("%front"),
                Some("_front"),
                Some("middle%value"),
                Some("front"),
            ],
            Nullability::NonNullable,
        );

        let escaped_pct = like_kernel(&arr, r"\%%", LikeOptions::default())?;
        assert!(escaped_pct.is_some(), "escaped percent prefix should match");
        assert_arrays_eq!(
            escaped_pct.unwrap(),
            BoolArray::from_iter([true, false, false, false])
        );

        let escaped_contains = like_kernel(&arr, r"%\%%", LikeOptions::default())?;
        assert!(
            escaped_contains.is_some(),
            "escaped percent contains should match"
        );
        assert_arrays_eq!(
            escaped_contains.unwrap(),
            BoolArray::from_iter([true, false, true, false])
        );
        Ok(())
    }

    /// A long `prefix%` literal stays on the token-level prefix automaton,
    /// which (unlike the contains automaton) has no `u8` state-space bound.
    #[test]
    fn test_like_long_prefix_handled() -> VortexResult<()> {
        let prefix = "a".repeat(300);
        let matching = format!("{prefix}-tail");
        let non_matching = format!("{}b-tail", "a".repeat(299));

        let arr = make_onpair(
            &[Some(&matching), Some(&non_matching), Some(prefix.as_str())],
            Nullability::NonNullable,
        );
        let pattern = format!("{prefix}%");
        let result = like_kernel(&arr, &pattern, LikeOptions::default())?;
        assert!(result.is_some(), "OnPair handles arbitrarily long prefixes");
        assert_arrays_eq!(result.unwrap(), BoolArray::from_iter([true, false, true]));
        Ok(())
    }

    /// `%needle%` longer than 255 bytes exceeds onpair's contains state space,
    /// so the kernel falls back — but the full engine path (decompress + Arrow
    /// LIKE) must still return correct results.
    #[test]
    fn test_like_long_contains_falls_back_but_matches() -> VortexResult<()> {
        let needle = "a".repeat(300);
        let matching = format!("xx{needle}yy");
        let non_matching = format!("xx{}byy", "a".repeat(299));
        let pattern = format!("%{needle}%");

        let arr = make_onpair(
            &[Some(&matching), Some(&non_matching), Some(needle.as_str())],
            Nullability::NonNullable,
        );
        assert!(
            like_kernel(&arr, &pattern, LikeOptions::default())?.is_none(),
            "contains needles longer than 255 bytes must fall back"
        );

        let result = like(arr, &pattern)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true]));
        Ok(())
    }

    /// Slicing keeps `codes_offsets` absolute into the full `codes` child; the
    /// search path must still produce correct per-row results once the sliced
    /// array is reduced back to OnPair and the LIKE pushdown runs.
    #[test]
    fn test_like_after_slice() -> VortexResult<()> {
        let arr = make_onpair(
            &[
                Some("alpha"),
                Some("alpine"),
                Some("beta"),
                Some("alabama"),
                Some("gamma"),
            ],
            Nullability::NonNullable,
        );
        let sliced = arr.into_array().slice(1..4)?;
        let len = sliced.len();
        let pattern = ConstantArray::new("al%", len).into_array();
        let result = Like
            .try_new_array(len, LikeOptions::default(), [sliced, pattern])?
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_bool();
        assert_arrays_eq!(&result, &BoolArray::from_iter([true, false, true]));
        Ok(())
    }
}
