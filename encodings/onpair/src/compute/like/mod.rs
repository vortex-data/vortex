// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::search;
use onpair::search::ContainsScan;
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
use crate::array::dict_view;
use crate::decode::CodesWindow;
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
                Some(match &window {
                    CodesWindow::U32(window) => BitBuffer::collect_bool(array.len(), |row| {
                        search::equals(window.row(row), &query)
                    }),
                    CodesWindow::U64(window) => BitBuffer::collect_bool(array.len(), |row| {
                        search::equals(window.row(row), &query)
                    }),
                })
            }
            SearchPattern::Prefix(prefix) => {
                let window = collect_codes_window(array, ctx)?;
                let query = search::PrefixQuery::new(&prefix, dict);
                Some(match &window {
                    CodesWindow::U32(window) => BitBuffer::collect_bool(array.len(), |row| {
                        search::starts_with(window.row(row), &query)
                    }),
                    CodesWindow::U64(window) => BitBuffer::collect_bool(array.len(), |row| {
                        search::starts_with(window.row(row), &query)
                    }),
                })
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

    if needle.len() > ContainsScan::MAX_PATTERN_LEN {
        return Ok(None);
    }

    let Some(frequencies) = token_frequency_index(array, ctx)? else {
        return Ok(None);
    };
    let scan = ContainsScan::new(needle, dict, frequencies)
        .map_err(|error| vortex_err!("OnPair substring scan preparation failed: {error}"))?;
    let window = collect_codes_window(array, ctx)?;
    let rows = match &window {
        CodesWindow::U32(window) => scan_contains(window.as_column_view(dict), &scan),
        CodesWindow::U64(window) => scan_contains(window.as_column_view(dict), &scan),
    };
    Ok(Some(BitBuffer::from_indices(array.len(), rows)))
}

fn scan_contains<O: onpair::Offset>(
    view: onpair::ColumnView<'_, O>,
    scan: &ContainsScan,
) -> Vec<usize> {
    let mut rows = Vec::new();
    scan.scan(view.codes, view.row_offsets, view.dict, &mut rows);
    rows
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
mod tests;
