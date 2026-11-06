// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::cell::RefCell;

use onpair::CompactDictionaryView;
use onpair::DictionaryView;
use onpair::search;
use onpair::search::ContainsScan;
use onpair::search::index::TokenFrequencyIndexView;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::scalar_fn::fns::like::LikeKernel;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::array::dict_view;
use crate::decode::CodesWindow;
use crate::decode::collect_codes_window;

/// The number of codes that the token frequencies of a scan count at most. It also keeps the
/// counts within `u32`.
const FREQUENCY_SAMPLE_LEN: usize = 1 << 14;

/// The number of evenly spaced blocks that the frequency sample takes its codes from.
const FREQUENCY_SAMPLE_BLOCKS: usize = 16;

/// A `LIKE` pattern that the kernel can match on the codes without a decode.
#[derive(Debug, PartialEq, Eq)]
enum Shape<'a> {
    /// `'%needle%'`
    Contains(&'a [u8]),
    /// `'needle%'`
    Prefix(&'a [u8]),
}

/// The shape of `pattern`, or `None` if the kernel cannot match it on the codes.
fn shape(pattern: &[u8]) -> Option<Shape<'_>> {
    let body = pattern.strip_suffix(b"%")?;

    let (needle, contains) = match body.strip_prefix(b"%") {
        Some(needle) => (needle, true),
        None => (body, false),
    };

    // The needle must be a literal: no wildcards and no escapes.
    if needle
        .iter()
        .any(|byte| matches!(byte, b'%' | b'_' | b'\\'))
    {
        return None;
    }

    if contains {
        (needle.len() <= ContainsScan::MAX_PATTERN_LEN).then_some(Shape::Contains(needle))
    } else {
        Some(Shape::Prefix(needle))
    }
}

/// The rows of `window` that contain `needle`, or `None` if the search cannot be prepared.
///
/// The scan finds the rows from the codes of a few rare tokens. It is exact because the
/// encoder tokenizes each row greedily.
fn rows_containing(
    needle: &[u8],
    dict: CompactDictionaryView<'_>,
    window: &CodesWindow,
) -> Option<Vec<usize>> {
    CUMULATIVE.with_borrow_mut(|cumulative| {
        count_tokens(
            &frequency_sample(window.codes()),
            dict.num_tokens(),
            cumulative,
        );
        let frequencies = TokenFrequencyIndexView::new(cumulative).ok()?;
        let scan = ContainsScan::new(needle, dict, frequencies).ok()?;

        let mut rows = Vec::new();
        scan.scan(window.codes(), window.offsets(), &mut rows);
        Some(rows)
    })
}

thread_local! {
    /// The token counts of the last scan on this thread. A dictionary can have 2^16 tokens, so a
    /// new zeroed array for each scan costs more than the count.
    static CUMULATIVE: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Writes the prefix sums of the token counts of `codes` to `cumulative`, in the format of
/// [`TokenFrequencyIndexView`]: entry `t` counts the codes below token `t`, and the last entry
/// counts all of them. A code outside the dictionary is not counted.
fn count_tokens(codes: &[u16], num_tokens: usize, cumulative: &mut Vec<u32>) {
    cumulative.clear();
    cumulative.resize(num_tokens + 1, 0);

    for &code in codes {
        if let Some(count) = cumulative.get_mut(usize::from(code) + 1) {
            *count += 1;
        }
    }

    for token in 0..num_tokens {
        cumulative[token + 1] += cumulative[token];
    }
}

/// The codes from which a scan counts its token frequencies.
///
/// The frequencies only choose the tokens that the scan looks for, so inexact counts make a scan
/// slower but never wrong. Counting every code costs more than the scan, so a long stream gives
/// evenly spaced blocks of its codes.
fn frequency_sample(codes: &[u16]) -> Cow<'_, [u16]> {
    if codes.len() <= FREQUENCY_SAMPLE_LEN {
        return Cow::Borrowed(codes);
    }

    let block_len = FREQUENCY_SAMPLE_LEN / FREQUENCY_SAMPLE_BLOCKS;
    let stride = codes.len() / FREQUENCY_SAMPLE_BLOCKS;

    let mut sample = Vec::with_capacity(FREQUENCY_SAMPLE_LEN);
    for block in 0..FREQUENCY_SAMPLE_BLOCKS {
        let start = block * stride;
        sample.extend_from_slice(&codes[start..start + block_len]);
    }
    Cow::Owned(sample)
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

        let Some(shape) = shape(pattern_bytes) else {
            return Ok(None);
        };

        let dict = dict_view(array, ctx)?;
        let window = collect_codes_window(array, ctx)?;
        let negated = options.negated;

        let buffer = match shape {
            Shape::Contains(needle) => {
                let Some(rows) = rows_containing(needle, dict, &window) else {
                    return Ok(None);
                };

                let mut buffer = BitBufferMut::new_unset(array.len());
                for row in rows {
                    buffer.set(row);
                }

                if negated {
                    !buffer.freeze()
                } else {
                    buffer.freeze()
                }
            }
            Shape::Prefix(needle) => {
                let query = search::PrefixQuery::new(needle, dict);
                BitBuffer::collect_bool(array.len(), |i| {
                    search::starts_with(window.row(i), &query) != negated
                })
            }
        };

        let validity = array
            .validity()?
            .union_nullability(pattern_scalar.dtype().nullability());

        Ok(Some(BoolArray::new(buffer, validity).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
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

    use super::Shape;
    use super::shape;
    use crate::DEFAULT_CONFIG;
    use crate::OnPair;
    use crate::OnPairArraySlotsExt;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const VALUES: [Option<&str>; 9] = [
        Some("http://www.google.com/search"),
        Some("https://maps.google.de/"),
        None,
        Some("Google - the search engine"),
        Some("http://example.com/googl"),
        Some(""),
        Some("http://yandex.ru/?q=google"),
        Some("goo"),
        Some("http://www.google.com/search"),
    ];

    #[rstest]
    #[case(b"%google%", Some(Shape::Contains(b"google")))]
    #[case(b"%", Some(Shape::Prefix(b"")))]
    #[case(b"%%", Some(Shape::Contains(b"")))]
    #[case(b"http%", Some(Shape::Prefix(b"http")))]
    #[case(b"%google", None)]
    #[case(b"google", None)]
    #[case(b"%g_ogle%", None)]
    #[case(b"%a%b%", None)]
    #[case(b"%a\\%%", None)]
    fn pattern_shape(#[case] pattern: &[u8], #[case] expected: Option<Shape<'_>>) {
        assert_eq!(shape(pattern), expected);
    }

    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case("%google%", false)]
    #[case("%google%", true)]
    #[case("%.google.%", true)]
    #[case("%Google%", false)]
    #[case("%search%", false)]
    #[case("%", false)]
    #[case("%", true)]
    #[case("http%", false)]
    #[case("http://www.%", true)]
    #[case("goo%", false)]
    fn like_matches_decoded(#[case] pattern: &str, #[case] negated: bool) -> VortexResult<()> {
        let options = LikeOptions {
            negated,
            case_insensitive: false,
        };
        let mut ctx = SESSION.create_execution_ctx();

        let input = VarBinArray::from_iter(VALUES, DType::Utf8(Nullability::Nullable));
        let pattern_array = ConstantArray::new(pattern, VALUES.len()).into_array();
        let expected =
            Like::try_new(input.clone().into_array(), pattern_array, options)?.into_array();

        let compressed = onpair_compress(input.as_array(), DEFAULT_CONFIG, &mut ctx)?;
        let sliced = compressed.into_array().slice(1..VALUES.len())?;
        let expected = expected.slice(1..VALUES.len())?;
        let sliced = sliced
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair array, got {}", array.encoding_id()))?;

        let pattern_array = ConstantArray::new(pattern, sliced.len()).into_array();
        let actual =
            <OnPair as LikeKernel>::like(sliced.as_view(), &pattern_array, options, &mut ctx)?
                .ok_or_else(|| vortex_err!("OnPair LikeKernel declined {pattern}"))?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// A stream longer than the frequency sample plans from a sample of its codes.
    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case("%google%")]
    #[case("%.google.%")]
    #[case("%/7/%")]
    fn like_on_long_stream_matches_decoded(#[case] pattern: &str) -> VortexResult<()> {
        let values: Vec<String> = (0..40_000)
            .map(|row| match row % 5 {
                0 => format!("http://www.google.com/search?q={row}"),
                1 => format!("https://maps.google.de/{}/{row}", row % 13),
                _ => format!("http://site{}.example.org/page/{row}", row % 97),
            })
            .collect();
        let options = LikeOptions::default();
        let mut ctx = SESSION.create_execution_ctx();

        let input = VarBinArray::from_iter(
            values.iter().map(|value| Some(value.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );
        let pattern_array = ConstantArray::new(pattern, values.len()).into_array();
        let expected =
            Like::try_new(input.clone().into_array(), pattern_array.clone(), options)?.into_array();

        let compressed = onpair_compress(input.as_array(), DEFAULT_CONFIG, &mut ctx)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair array, got {}", array.encoding_id()))?;
        assert!(compressed.codes().len() > super::FREQUENCY_SAMPLE_LEN);
        let actual =
            <OnPair as LikeKernel>::like(compressed.as_view(), &pattern_array, options, &mut ctx)?
                .ok_or_else(|| vortex_err!("OnPair LikeKernel declined {pattern}"))?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
