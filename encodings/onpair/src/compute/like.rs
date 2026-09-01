// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::search;
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
use crate::index::token_frequency_index;

enum SearchPattern {
    Exact(Vec<u8>),
    Prefix(Vec<u8>),
    Contains(Vec<u8>),
}

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
        let Some(pattern) = pattern_scalar.as_utf8().value() else {
            return Ok(None);
        };
        let Some(search_pattern) = classify_like_pattern(pattern.as_bytes()) else {
            return Ok(None);
        };

        let dict = dict_view(array, ctx)?;
        let matches = match search_pattern {
            SearchPattern::Exact(needle) => {
                let window = collect_codes_window(array, ctx)?;
                let query = search::tokenize(&needle, dict);
                Some(BitBuffer::collect_bool(array.len(), |row| {
                    search::equals(window.row(row), &query)
                }))
            }
            SearchPattern::Prefix(prefix) => {
                let window = collect_codes_window(array, ctx)?;
                let query = search::PrefixQuery::new(&prefix, dict);
                Some(BitBuffer::collect_bool(array.len(), |row| {
                    search::starts_with(window.row(row), &query)
                }))
            }
            SearchPattern::Contains(needle) => contains(array, dict, &needle, ctx)?,
        };
        let Some(matches) = matches else {
            return Ok(None);
        };

        let matches = if options.negated { !matches } else { matches };
        let validity = array
            .array()
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());
        Ok(Some(BoolArray::new(matches, validity).into_array()))
    }
}

fn contains(
    array: ArrayView<'_, OnPair>,
    dict: onpair::CompactDictionaryView<'_>,
    needle: &[u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<BitBuffer>> {
    if needle.is_empty() {
        return Ok(Some(BitBuffer::new_set(array.len())));
    }

    if let Some(frequencies) = token_frequency_index(array, ctx)? {
        let analysis = search::analyze_prefilter(needle, dict, frequencies);
        if !search::prefilter_is_likely_profitable(&analysis, array.len()) {
            return Ok(None);
        }
        let window = collect_codes_window(array, ctx)?;
        let view = window.as_column_view(dict);
        let mut rows = Vec::new();
        if search::prefilter_candidates(view.codes, view.row_offsets, &analysis, &mut rows).is_err()
        {
            return Ok(None);
        }
        search::BytesVerifier::new(needle).retain(view, &mut rows);
        return Ok(Some(BitBuffer::from_indices(array.len(), rows)));
    }

    Ok(None)
}

fn classify_like_pattern(pattern: &[u8]) -> Option<SearchPattern> {
    let mut literal = Vec::with_capacity(pattern.len());
    let mut wildcards = Vec::new();
    let mut index = 0;
    while index < pattern.len() {
        match pattern[index] {
            b'\\' => {
                index += 1;
                if index < pattern.len() {
                    literal.push(pattern[index]);
                } else {
                    literal.push(b'\\');
                }
            }
            b'%' => wildcards.push(literal.len()),
            b'_' => return None,
            byte => literal.push(byte),
        }
        index += 1;
    }

    match wildcards.as_slice() {
        [] => Some(SearchPattern::Exact(literal)),
        [end] if *end == literal.len() => Some(SearchPattern::Prefix(literal)),
        [0, end] if *end == literal.len() => Some(SearchPattern::Contains(literal)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::like::Like;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::DEFAULT_CONFIG;
    use crate::OnPairArray;
    use crate::build_token_frequency_index;
    use crate::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn encode(values: &[Option<&str>], indexed: bool) -> VortexResult<OnPairArray> {
        let input =
            VarBinArray::from_iter(values.iter().copied(), DType::Utf8(Nullability::Nullable))
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair, got {}", array.encoding_id()))?;
        if indexed {
            build_token_frequency_index(encoded, &mut ctx)
        } else {
            Ok(encoded)
        }
    }

    fn execute_kernel(
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

    #[rstest]
    #[case("alpha", [Some(true), None, Some(false), Some(false), Some(false)])]
    #[case("alpha%", [Some(true), None, Some(true), Some(false), Some(false)])]
    fn supported_patterns(
        #[case] pattern: &str,
        #[case] expected: [Option<bool>; 5],
    ) -> VortexResult<()> {
        let array = encode(
            &[
                Some("alpha"),
                None,
                Some("alphabet"),
                Some("say hello"),
                Some(""),
            ],
            true,
        )?;
        let result = execute_kernel(&array, pattern, LikeOptions::default())?
            .ok_or_else(|| vortex_err!("OnPair should handle {pattern}"))?;
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(expected),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn unprofitable_contains_falls_back() -> VortexResult<()> {
        let array = encode(
            &[
                Some("alpha"),
                None,
                Some("alphabet"),
                Some("say hello"),
                Some(""),
            ],
            true,
        )?;
        assert!(
            execute_kernel(&array, "%ha%", LikeOptions::default())?.is_none(),
            "a high candidate fraction should use canonical LIKE"
        );

        let pattern = ConstantArray::new("%ha%", array.len()).into_array();
        let result = Like::try_new(array.into_array(), pattern, LikeOptions::default())?
            .into_array()
            .execute::<BoolArray>(&mut SESSION.create_execution_ctx())?
            .into_array();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), None, Some(true), Some(false), Some(false)]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn indexed_prefilter_supports_long_patterns() -> VortexResult<()> {
        let needle = "x".repeat(usize::from(u8::MAX) + 1);
        let matching = format!("before{needle}after");
        let pattern = format!("%{needle}%");

        let mut values = vec![Some("no match"); 4096];
        values[2048] = Some(&matching);
        let indexed = encode(&values, true)?;
        let result = execute_kernel(&indexed, &pattern, LikeOptions::default())?
            .ok_or_else(|| vortex_err!("indexed contains should use the byte verifier"))?;
        let mut expected = vec![Some(false); values.len()];
        expected[2048] = Some(true);
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(expected),
            &mut SESSION.create_execution_ctx()
        );

        let unindexed = encode(&values, false)?;
        assert!(
            execute_kernel(&unindexed, &pattern, LikeOptions::default())?.is_none(),
            "unindexed overlong contains should fall back to canonical LIKE"
        );
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn indexed_prefilter_handles_slices_and_negation() -> VortexResult<()> {
        let mut values = vec![Some("ordinary filler"); 1026];
        values[0] = Some("outside");
        values[100] = Some("rare hello needle");
        values[101] = None;
        values[1025] = Some("outside");
        let array = encode(&values, true)?
            .into_array()
            .slice(1..1025)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected sliced OnPair, got {}", array.encoding_id()))?;
        let result = execute_kernel(
            &array,
            "%hello%",
            LikeOptions {
                negated: true,
                case_insensitive: false,
            },
        )?
        .ok_or_else(|| vortex_err!("OnPair should handle indexed contains on a slice"))?;
        let expected = (1..1025).map(|index| values[index].map(|value| !value.contains("hello")));
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(expected),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[rstest]
    #[case("%suffix", LikeOptions::default())]
    #[case("a_b", LikeOptions::default())]
    #[case("a%b", LikeOptions::default())]
    #[case(
        "%alpha%",
        LikeOptions {
            negated: false,
            case_insensitive: true,
        }
    )]
    fn unsupported_patterns_fall_back(
        #[case] pattern: &str,
        #[case] options: LikeOptions,
    ) -> VortexResult<()> {
        let array = encode(&[Some("alpha")], true)?;
        assert!(execute_kernel(&array, pattern, options)?.is_none());
        Ok(())
    }
}
