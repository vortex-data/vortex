// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

use onpair::search::ContainsTable;
use onpair::search::PrefixQuery;
use onpair::search::contains;
use onpair::search::starts_with;
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
use crate::array::dict_view;
use crate::decode::collect_codes_window;

/// The longest substring the token-level KMP table supports: its states are `u8`.
const MAX_CONTAINS_LEN: usize = u8::MAX as usize;

impl LikeKernel for OnPair {
    fn like(
        array: ArrayView<'_, Self>,
        pattern: &ArrayRef,
        options: LikeOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if options.case_insensitive {
            return Ok(None);
        }
        let Some(pattern_scalar) = pattern.as_constant() else {
            return Ok(None);
        };
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
        if matches!(&kind, LikeKind::Contains(needle) if needle.len() > MAX_CONTAINS_LEN) {
            return Ok(None);
        }

        let dict = dict_view(array, ctx)?;
        let window = collect_codes_window(array, ctx)?;
        let negated = options.negated;
        let result = match kind {
            LikeKind::Prefix(prefix) => {
                let query = PrefixQuery::new(&prefix, dict);
                BitBuffer::collect_bool(array.len(), |i| {
                    starts_with(window.row(i), &query) != negated
                })
            }
            LikeKind::Contains(needle) => {
                let table = ContainsTable::new(&needle, dict);
                BitBuffer::collect_bool(array.len(), |i| {
                    contains(window.row(i), &table) != negated
                })
            }
        };

        let validity = array
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());
        Ok(Some(BoolArray::new(result, validity).into_array()))
    }
}

/// The LIKE patterns matched on codes, parsed as the FSST kernel parses them: `\` escapes the
/// next byte, and a trailing `\` is a literal backslash.
enum LikeKind<'a> {
    /// `prefix%`
    Prefix(Cow<'a, [u8]>),
    /// `%needle%`
    Contains(Cow<'a, [u8]>),
}

impl<'a> LikeKind<'a> {
    fn parse(pattern: &'a [u8]) -> Option<Self> {
        literal_until_final_percent(pattern, 0)
            .map(LikeKind::Prefix)
            .or_else(|| {
                pattern
                    .starts_with(b"%")
                    .then(|| literal_until_final_percent(pattern, 1))
                    .flatten()
                    .map(LikeKind::Contains)
            })
    }
}

/// Parses `pattern[start..]` as a literal ended by a single trailing `%`, or `None` if it holds
/// `_`, a `%` before the end, or no trailing `%`.
fn literal_until_final_percent(pattern: &[u8], start: usize) -> Option<Cow<'_, [u8]>> {
    let mut literal: Option<Vec<u8>> = None;
    let mut idx = start;
    while idx < pattern.len() {
        match pattern[idx] {
            b'\\' => {
                let escaped = pattern.get(idx + 1).copied().unwrap_or(b'\\');
                literal
                    .get_or_insert_with(|| pattern[start..idx].to_vec())
                    .push(escaped);
                idx = (idx + 2).min(pattern.len());
            }
            b'%' if idx + 1 == pattern.len() => {
                return Some(match literal {
                    Some(literal) => Cow::Owned(literal),
                    None => Cow::Borrowed(&pattern[start..idx]),
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_array::scalar_fn::fns::like::LikeKernel;
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::DEFAULT_CONFIG;
    use crate::OnPair;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const URLS: [Option<&str>; 8] = [
        Some("http://www.google.com/search?q=vortex"),
        Some("https://maps.google.de/"),
        None,
        Some("http://example.com/googl"),
        Some("http://example.com/ogle"),
        Some(""),
        Some("goo%gle_100\\path"),
        Some("http://www.example.org/index.html"),
    ];

    fn like(array: ArrayRef, pattern: &str, options: LikeOptions) -> VortexResult<ArrayRef> {
        let len = array.len();
        Ok(Like::try_new(array, ConstantArray::new(pattern, len).into_array(), options)?
            .into_array())
    }

    /// The kernel must agree with LIKE over the decompressed strings, including escapes, nulls,
    /// negation and patterns it declines.
    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case::contains("%google%")]
    #[case::contains_across_tokens("%e.com/go%")]
    #[case::contains_missing("%vortexdata%")]
    #[case::contains_empty("%%")]
    #[case::prefix("http://www.%")]
    #[case::prefix_empty("%")]
    #[case::prefix_whole_value("http://example.com/ogle%")]
    #[case::escaped_percent("%goo\\%gle%")]
    #[case::escaped_underscore("%gle\\_1%")]
    #[case::trailing_backslash("%path\\")]
    #[case::declined_underscore("%go_gle%")]
    #[case::declined_suffix("%.html")]
    #[case::declined_exact("https://maps.google.de/")]
    fn matches_decompressed_like(
        #[case] pattern: &str,
        #[values(false, true)] negated: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = VarBinArray::from_iter(URLS, DType::Utf8(Nullability::Nullable)).into_array();
        let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?;
        assert!(encoded.is::<OnPair>());
        let options = LikeOptions {
            negated,
            case_insensitive: false,
        };
        let expected = like(input, pattern, options)?.execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(like(encoded, pattern, options)?, expected, &mut ctx);
        Ok(())
    }

    /// Prefix and substring patterns are matched on codes rather than declined; the rest fall
    /// back to decompression. Rows of a slice resolve against the retained codes.
    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case("%google%", Some([true, false, false]))]
    #[case("https://%", Some([true, false, false]))]
    #[case("%.de/", None)]
    #[case("%go_gle%", None)]
    fn kernel_on_sliced_array(
        #[case] pattern: &str,
        #[case] expected: Option<[bool; 3]>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = VarBinArray::from_iter(
            URLS.iter().flatten().map(|url| Some(*url)),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let sliced = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?
            .slice(1..4)?
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("sliced array was not OnPair"))?;
        let pattern = ConstantArray::new(pattern, sliced.len()).into_array();
        let result = <OnPair as LikeKernel>::like(
            sliced.as_view(),
            &pattern,
            LikeOptions::default(),
            &mut ctx,
        )?;
        match (result, expected) {
            (Some(result), Some(expected)) => {
                assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx)
            }
            (None, None) => {}
            (result, expected) => panic!(
                "kernel handled = {}, expected handled = {}",
                result.is_some(),
                expected.is_some()
            ),
        }
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn declines_case_insensitive_and_long_needles() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = VarBinArray::from_iter(URLS, DType::Utf8(Nullability::Nullable)).into_array();
        let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("expected OnPair array"))?;
        let ilike = LikeOptions {
            negated: false,
            case_insensitive: true,
        };
        let pattern = ConstantArray::new("%google%", encoded.len()).into_array();
        assert!(<OnPair as LikeKernel>::like(encoded.as_view(), &pattern, ilike, &mut ctx)?.is_none());

        let long = format!("%{}%", "a".repeat(256));
        let pattern = ConstantArray::new(long.as_str(), encoded.len()).into_array();
        assert!(
            <OnPair as LikeKernel>::like(
                encoded.as_view(),
                &pattern,
                LikeOptions::default(),
                &mut ctx
            )?
            .is_none()
        );
        Ok(())
    }
}
