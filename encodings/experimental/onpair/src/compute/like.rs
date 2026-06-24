// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use memchr::memmem::Finder;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::decode::code_boundary_at;
use crate::decode::collect_widened;

#[derive(Clone, Copy)]
enum SimpleLike<'a> {
    All,
    Exact(&'a [u8]),
    Prefix(&'a [u8]),
    Suffix(&'a [u8]),
    Contains(&'a [u8]),
}

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
        let Some(parsed) = parse_simple_like(pattern_bytes) else {
            return Ok(None);
        };

        let codes_offsets = array.codes_offsets();
        let code_start = code_boundary_at(codes_offsets, 0, ctx)?;
        let code_end = code_boundary_at(codes_offsets, array.len(), ctx)?;
        vortex_ensure!(
            code_start <= code_end,
            "OnPair codes_offsets must be nondecreasing"
        );
        vortex_ensure!(
            code_end <= array.codes().len(),
            "OnPair codes_offsets end {} exceeds codes len {}",
            code_end,
            array.codes().len()
        );

        let codes = collect_widened::<u16>(&array.codes().slice(code_start..code_end)?, ctx)?;
        let code_offsets = normalize_code_offsets(
            collect_widened::<u64>(codes_offsets, ctx)?.as_slice(),
            code_start,
            code_end,
        )?;
        let dict_offsets = collect_widened::<u32>(array.dict_offsets(), ctx)?;
        let dict_bytes = array.dict_bytes();
        let dict_bytes = dict_bytes.as_slice();
        let mut tail = Vec::new();
        let mut scratch = Vec::new();

        let bits = match parsed {
            SimpleLike::All => BitBuffer::collect_bool(array.len(), |_| true ^ options.negated),
            SimpleLike::Exact(needle) => BitBuffer::collect_bool(array.len(), |row| {
                row_matches_exact(
                    row_codes(&code_offsets, &codes, row),
                    dict_bytes,
                    dict_offsets.as_slice(),
                    needle,
                ) ^ options.negated
            }),
            SimpleLike::Prefix(needle) => BitBuffer::collect_bool(array.len(), |row| {
                row_matches_prefix(
                    row_codes(&code_offsets, &codes, row),
                    dict_bytes,
                    dict_offsets.as_slice(),
                    needle,
                ) ^ options.negated
            }),
            SimpleLike::Suffix(needle) => BitBuffer::collect_bool(array.len(), |row| {
                row_matches_suffix(
                    row_codes(&code_offsets, &codes, row),
                    dict_bytes,
                    dict_offsets.as_slice(),
                    needle,
                    &mut tail,
                ) ^ options.negated
            }),
            SimpleLike::Contains(needle) => {
                let finder = Finder::new(needle);
                BitBuffer::collect_bool(array.len(), |row| {
                    row_matches_contains(
                        row_codes(&code_offsets, &codes, row),
                        dict_bytes,
                        dict_offsets.as_slice(),
                        needle,
                        &finder,
                        &mut tail,
                        &mut scratch,
                    ) ^ options.negated
                })
            }
        };

        let validity = array
            .array_validity()
            .union_nullability(pattern_scalar.dtype().nullability());
        Ok(Some(BoolArray::new(bits, validity).into_array()))
    }
}

fn normalize_code_offsets(
    code_offsets: &[u64],
    code_start: usize,
    code_end: usize,
) -> VortexResult<Vec<usize>> {
    let offsets = code_offsets
        .iter()
        .map(|&offset| {
            usize::try_from(offset)
                .map_err(|_| vortex_err!("OnPair code offset {} exceeds usize", offset))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    for &offset in &offsets {
        vortex_ensure!(
            offset >= code_start && offset <= code_end,
            "OnPair codes offset {} outside row window {}..{}",
            offset,
            code_start,
            code_end
        );
    }
    for window in offsets.windows(2) {
        vortex_ensure!(
            window[0] <= window[1],
            "OnPair codes_offsets must be nondecreasing"
        );
    }

    Ok(offsets
        .into_iter()
        .map(|offset| offset - code_start)
        .collect())
}

fn parse_simple_like(pattern: &[u8]) -> Option<SimpleLike<'_>> {
    if pattern.is_empty() {
        return Some(SimpleLike::Exact(b""));
    }
    if pattern.iter().any(|&b| matches!(b, b'_' | b'\\')) {
        return None;
    }

    let Some(first_literal) = pattern.iter().position(|&b| b != b'%') else {
        return Some(SimpleLike::All);
    };
    let last_literal = pattern.iter().rposition(|&b| b != b'%')? + 1;
    let literal = &pattern[first_literal..last_literal];
    if literal.contains(&b'%') {
        return None;
    }

    match (first_literal == 0, last_literal == pattern.len()) {
        (true, true) => Some(SimpleLike::Exact(literal)),
        (true, false) => Some(SimpleLike::Prefix(literal)),
        (false, true) => Some(SimpleLike::Suffix(literal)),
        (false, false) => Some(SimpleLike::Contains(literal)),
    }
}

fn row_codes<'a>(code_offsets: &[usize], codes: &'a [u16], row: usize) -> &'a [u16] {
    let start = code_offsets[row];
    let end = code_offsets[row + 1];
    &codes[start..end]
}

fn token_bytes<'a>(dict_bytes: &'a [u8], dict_offsets: &[u32], code: u16) -> &'a [u8] {
    let code = usize::from(code);
    let start = dict_offsets[code] as usize;
    let end = dict_offsets[code + 1] as usize;
    &dict_bytes[start..end]
}

fn row_matches_exact(
    codes: &[u16],
    dict_bytes: &[u8],
    dict_offsets: &[u32],
    needle: &[u8],
) -> bool {
    let mut matched = 0;
    for &code in codes {
        let token = token_bytes(dict_bytes, dict_offsets, code);
        if matched + token.len() > needle.len() {
            return false;
        }
        if token != &needle[matched..matched + token.len()] {
            return false;
        }
        matched += token.len();
    }
    matched == needle.len()
}

fn row_matches_prefix(
    codes: &[u16],
    dict_bytes: &[u8],
    dict_offsets: &[u32],
    needle: &[u8],
) -> bool {
    if needle.is_empty() {
        return true;
    }

    let mut matched = 0;
    for &code in codes {
        let token = token_bytes(dict_bytes, dict_offsets, code);
        let take = (needle.len() - matched).min(token.len());
        if token[..take] != needle[matched..matched + take] {
            return false;
        }
        matched += take;
        if matched == needle.len() {
            return true;
        }
    }
    false
}

fn row_matches_suffix(
    codes: &[u16],
    dict_bytes: &[u8],
    dict_offsets: &[u32],
    needle: &[u8],
    tail: &mut Vec<u8>,
) -> bool {
    if needle.is_empty() {
        return true;
    }

    let mut total_len = 0;
    tail.clear();
    for &code in codes {
        let token = token_bytes(dict_bytes, dict_offsets, code);
        total_len += token.len();
        append_tail(tail, token, needle.len());
    }
    total_len >= needle.len() && tail.as_slice() == needle
}

fn row_matches_contains(
    codes: &[u16],
    dict_bytes: &[u8],
    dict_offsets: &[u32],
    needle: &[u8],
    finder: &Finder<'_>,
    tail: &mut Vec<u8>,
    scratch: &mut Vec<u8>,
) -> bool {
    if needle.is_empty() {
        return true;
    }

    tail.clear();
    for &code in codes {
        let token = token_bytes(dict_bytes, dict_offsets, code);
        if finder.find(token).is_some() {
            return true;
        }
        if !tail.is_empty() {
            scratch.clear();
            scratch.extend_from_slice(tail);
            scratch.extend_from_slice(token);
            if finder.find(scratch).is_some() {
                return true;
            }
        }
        append_tail(tail, token, needle.len() - 1);
    }
    false
}

fn append_tail(tail: &mut Vec<u8>, bytes: &[u8], max_len: usize) {
    if max_len == 0 {
        return;
    }
    if bytes.len() >= max_len {
        tail.clear();
        tail.extend_from_slice(&bytes[bytes.len() - max_len..]);
        return;
    }
    let overflow = tail.len() + bytes.len();
    if overflow > max_len {
        tail.drain(..overflow - max_len);
    }
    tail.extend_from_slice(bytes);
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
    use vortex_array::scalar_fn::fns::like::LikeOptions;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn run_like(
        values: &[Option<&str>],
        pattern: &str,
        options: LikeOptions,
    ) -> VortexResult<BoolArray> {
        let input =
            VarBinArray::from_iter(values.iter().copied(), DType::Utf8(Nullability::Nullable));
        let len = input.len();
        let dtype = input.dtype().clone();
        let array = onpair_compress(&input, len, &dtype, DEFAULT_DICT12_CONFIG)?.into_array();
        let pattern = ConstantArray::new(pattern, len).into_array();
        let result = Like
            .try_new_array(len, options, [array, pattern])?
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())?
            .into_bool();
        Ok(result)
    }

    #[test]
    fn like_contains() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = run_like(
            &[
                Some("https://google.example"),
                Some("no match"),
                Some("prefix Google suffix"),
                None,
            ],
            "%Google%",
            LikeOptions::default(),
        )?;
        assert_arrays_eq!(
            &result,
            &BoolArray::from_iter([Some(false), Some(false), Some(true), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn like_prefix_suffix_exact_and_negated() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = [
            Some("2020-10-01"),
            Some("2020-11-01"),
            Some("x-2020-10-01"),
            Some(""),
        ];
        assert_arrays_eq!(
            &run_like(&values, "2020-10-%", LikeOptions::default())?,
            &BoolArray::from_iter([Some(true), Some(false), Some(false), Some(false)]),
            &mut ctx
        );
        assert_arrays_eq!(
            &run_like(&values, "%-01", LikeOptions::default())?,
            &BoolArray::from_iter([Some(true), Some(true), Some(true), Some(false)]),
            &mut ctx
        );
        assert_arrays_eq!(
            &run_like(&values, "2020-10-01", LikeOptions::default())?,
            &BoolArray::from_iter([Some(true), Some(false), Some(false), Some(false)]),
            &mut ctx
        );
        assert_arrays_eq!(
            &run_like(
                &values,
                "%2020%",
                LikeOptions {
                    negated: true,
                    case_insensitive: false,
                },
            )?,
            &BoolArray::from_iter([Some(false), Some(false), Some(false), Some(true)]),
            &mut ctx
        );
        Ok(())
    }
}
