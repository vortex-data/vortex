// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::search;
use onpair::search::ContainsScan;
use onpair::search::PrefixQuery;
use onpair::search::index::TokenFrequencyIndexView;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::array::dict_view;
use crate::decode::collect_codes_window;
use crate::decode::collect_widened;

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };
        if options.case_insensitive {
            return Ok(None);
        }
        let Some(pattern) = pattern_scalar.as_utf8_opt().and_then(|s| s.value()) else {
            return Ok(None);
        };
        let Some(pattern) = Pattern::parse(pattern.as_bytes()) else {
            return Ok(None);
        };

        let len = array.len();
        let matches = match pattern {
            Pattern::Any => BitBuffer::new_set(len),
            Pattern::Equals(literal) => {
                let query = search::tokenize(&literal, dict_view(array, ctx)?);
                let window = collect_codes_window(array, ctx)?;
                BitBuffer::collect_bool(len, |i| search::equals(window.row(i), &query))
            }
            Pattern::StartsWith(literal) => {
                let query = PrefixQuery::new(&literal, dict_view(array, ctx)?);
                let window = collect_codes_window(array, ctx)?;
                BitBuffer::collect_bool(len, |i| search::starts_with(window.row(i), &query))
            }
            Pattern::Contains(literal) => {
                let Some(frequencies) = array.token_frequencies() else {
                    return Ok(None);
                };
                let frequencies = collect_widened::<u32>(frequencies, ctx)?;
                let frequencies = TokenFrequencyIndexView::new(&frequencies)
                    .map_err(|e| vortex_err!("Invalid OnPair token frequencies: {e}"))?;
                // A pattern longer than the scan supports, or a dictionary it cannot
                // tokenize the pattern with, falls back to decoding.
                let Ok(scan) = ContainsScan::new(&literal, dict_view(array, ctx)?, frequencies)
                else {
                    return Ok(None);
                };
                let window = collect_codes_window(array, ctx)?;
                let mut rows = Vec::new();
                scan.scan(window.codes(), window.row_offsets(), &mut rows);
                BitBuffer::from_indices(len, rows)
            }
        };
        let matches = if options.negated { !matches } else { matches };

        let validity = array
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());
        Ok(Some(BoolArray::new(matches, validity).into_array()))
    }
}

/// The LIKE patterns the kernel answers on the codes.
#[derive(Debug, PartialEq, Eq)]
enum Pattern {
    /// Only `%`: every value matches.
    Any,
    /// No wildcard: the value equals the literal.
    Equals(Vec<u8>),
    /// `literal%`.
    StartsWith(Vec<u8>),
    /// `%literal%`.
    Contains(Vec<u8>),
}

impl Pattern {
    /// Parses `pattern` as [`Like`](vortex_array::scalar_fn::fns::like::Like) does: `%`
    /// matches any sequence, `\` escapes the next character, and a trailing `\` is a literal
    /// backslash. Returns `None` for patterns with `_`, a suffix, or more than one literal.
    fn parse(pattern: &[u8]) -> Option<Self> {
        let mut literal = Vec::with_capacity(pattern.len());
        let mut leading = false;
        let mut trailing = false;
        let mut bytes = pattern.iter().copied();
        while let Some(byte) = bytes.next() {
            match byte {
                b'%' if literal.is_empty() => leading = true,
                b'%' => trailing = true,
                b'_' => return None,
                _ if trailing => return None,
                b'\\' => literal.push(bytes.next().unwrap_or(b'\\')),
                _ => literal.push(byte),
            }
        }
        match (leading, trailing) {
            (false, false) => Some(Self::Equals(literal)),
            (false, true) => Some(Self::StartsWith(literal)),
            (true, true) => Some(Self::Contains(literal)),
            (true, false) if literal.is_empty() => Some(Self::Any),
            (true, false) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::Pattern;
    use crate::DEFAULT_CONFIG;
    use crate::OnPair;
    use crate::OnPairArray;
    use crate::compress::onpair_compress;
    use crate::compress::onpair_token_frequencies;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const NEGATED: LikeOptions = LikeOptions {
        negated: true,
        case_insensitive: false,
    };

    fn values() -> VarBinArray {
        let rows = [
            Some("https://www.example.com/products/1"),
            None,
            Some("http://test.org/a%b_c\\d"),
            Some(""),
            Some("ftp://files.example.net"),
            Some("https://vortex.dev"),
            Some("https://www.example.com/users/7/profile"),
            Some("example"),
        ];
        VarBinArray::from_iter(
            rows.into_iter().cycle().take(64),
            DType::Utf8(Nullability::Nullable),
        )
    }

    fn compress(values: &VarBinArray, token_frequencies: bool) -> VortexResult<OnPairArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = onpair_compress(values.as_array(), DEFAULT_CONFIG, &mut ctx)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair, got {}", array.encoding_id()))?;
        if !token_frequencies {
            return Ok(array);
        }
        let token_frequencies = onpair_token_frequencies(array.as_view(), &mut ctx)?;
        OnPair::with_token_frequencies(array, token_frequencies)
    }

    fn kernel_like(
        array: &OnPairArray,
        pattern: &str,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let pattern = ConstantArray::new(pattern, array.len()).into_array();
        <OnPair as LikeKernel>::like(
            array.as_view(),
            &pattern,
            options,
            &mut SESSION.create_execution_ctx(),
        )
    }

    fn decoded_like(
        values: ArrayRef,
        pattern: &str,
        options: LikeOptions,
    ) -> VortexResult<ArrayRef> {
        let pattern = ConstantArray::new(pattern, values.len()).into_array();
        Ok(Like::try_new(values, pattern, options)?.into_array())
    }

    #[cfg_attr(miri, ignore)]
    #[rstest]
    fn like_matches_decoded(
        #[values(
            "https://www.example.com/products/1",
            "",
            "example",
            "missing",
            "https://%",
            "https://www.example.com/products/1%",
            "%example%",
            "%example.com/%",
            "%missing%",
            "%",
            "%%",
            "%%example%%",
            "%a\\%b\\_c\\\\d%",
            "http://test.org/a\\%b\\_c\\\\d",
            "http://test.org/a\\%b\\_c\\"
        )]
        pattern: &str,
        #[values(LikeOptions::default(), NEGATED)] options: LikeOptions,
    ) -> VortexResult<()> {
        let values = values();
        let array = compress(&values, true)?;
        let result = kernel_like(&array, pattern, options)?
            .ok_or_else(|| vortex_err!("the kernel declined {pattern:?}"))?;
        let expected = decoded_like(values.into_array(), pattern, options)?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case::suffix("%profile", LikeOptions::default())]
    #[case::underscore("http_%", LikeOptions::default())]
    #[case::two_literals("%example%users%", LikeOptions::default())]
    #[case::case_insensitive("example", LikeOptions { negated: false, case_insensitive: true })]
    fn like_declines(#[case] pattern: &str, #[case] options: LikeOptions) -> VortexResult<()> {
        let array = compress(&values(), true)?;
        assert!(kernel_like(&array, pattern, options)?.is_none());
        Ok(())
    }

    /// Without token frequencies, `%needle%` falls back to decoding.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn like_contains_needs_token_frequencies() -> VortexResult<()> {
        let array = compress(&values(), false)?;
        assert!(kernel_like(&array, "%example%", LikeOptions::default())?.is_none());
        assert!(kernel_like(&array, "https://%", LikeOptions::default())?.is_some());
        Ok(())
    }

    /// A slice keeps the whole `codes` child, so its rows start past the first code.
    #[cfg_attr(miri, ignore)]
    #[rstest]
    fn like_on_sliced_array(
        #[values("example", "https://%", "%example%", "%")] pattern: &str,
    ) -> VortexResult<()> {
        let values = values();
        let array = compress(&values, true)?
            .into_array()
            .slice(5..37)?
            .optimize_ctx(&SESSION)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair, got {}", array.encoding_id()))?;
        let result = kernel_like(&array, pattern, LikeOptions::default())?
            .ok_or_else(|| vortex_err!("the kernel declined {pattern:?}"))?;
        let expected = decoded_like(
            values.into_array().slice(5..37)?,
            pattern,
            LikeOptions::default(),
        )?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    #[case("", Some(Pattern::Equals(b"".to_vec())))]
    #[case("a\\%", Some(Pattern::Equals(b"a%".to_vec())))]
    #[case("\\%a%", Some(Pattern::StartsWith(b"%a".to_vec())))]
    #[case("%%a\\_%%", Some(Pattern::Contains(b"a_".to_vec())))]
    #[case("%a\\", None)]
    #[case("%a\\\\%", Some(Pattern::Contains(b"a\\".to_vec())))]
    #[case("a\\", Some(Pattern::Equals(b"a\\".to_vec())))]
    #[case("%%%", Some(Pattern::Any))]
    #[case("a%b", None)]
    #[case("a\\_%", Some(Pattern::StartsWith(b"a_".to_vec())))]
    fn parse_pattern(#[case] pattern: &str, #[case] expected: Option<Pattern>) {
        assert_eq!(Pattern::parse(pattern.as_bytes()), expected);
    }
}
