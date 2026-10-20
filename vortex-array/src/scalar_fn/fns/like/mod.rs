// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;
mod pattern;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use memchr::memmem::Finder;
use pattern::LikePattern;
use prost::Message;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::ResolvedViews;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::display::ExprDisplay;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// Options for SQL LIKE function
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LikeOptions {
    pub negated: bool,
    pub case_insensitive: bool,
}

impl Display for LikeOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.negated {
            write!(f, "NOT ")?;
        }
        if self.case_insensitive {
            write!(f, "ILIKE")
        } else {
            write!(f, "LIKE")
        }
    }
}

/// Expression that performs SQL LIKE pattern matching.
#[derive(Clone)]
pub struct Like;

impl Like {
    /// Creates a lazy SQL `LIKE` operation over `input` and `pattern`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or are not UTF-8 arrays.
    pub fn try_new(
        input: ArrayRef,
        pattern: ArrayRef,
        options: LikeOptions,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Like.bind(options), [input, pattern])
    }
}

impl ScalarFnVTable for Like {
    type Options = LikeOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.like");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LikeOpts {
                negated: instance.negated,
                case_insensitive: instance.case_insensitive,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::LikeOpts::decode(_metadata)?;
        Ok(LikeOptions {
            negated: opts.negated,
            case_insensitive: opts.case_insensitive,
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("child"),
            1 => ChildName::from("pattern"),
            _ => unreachable!("Invalid child index {} for Like expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        Display::fmt(expr.display_child(0), f)?;
        if options.negated {
            write!(f, " not")?;
        }
        if options.case_insensitive {
            write!(f, " ilike ")?;
        } else {
            write!(f, " like ")?;
        }
        Display::fmt(expr.display_child(1), f)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        let pattern = &arg_dtypes[1];

        if !input.is_utf8() {
            vortex_bail!("LIKE expression requires UTF8 input dtype, got {}", input);
        }
        if !pattern.is_utf8() {
            vortex_bail!(
                "LIKE expression requires UTF8 pattern dtype, got {}",
                pattern
            );
        }

        Ok(DType::Bool(
            (input.is_nullable() || pattern.is_nullable()).into(),
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;
        let pattern = args.get(1)?;

        execute_like(&child, &pattern, *options, ctx)
    }

    fn is_strict(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Native implementation of LIKE over canonical [`VarBinViewArray`]s.
///
/// Patterns are compiled once per distinct pattern (see [`LikePattern`]) and evaluated per
/// value directly on the view bytes.
pub(crate) fn execute_like(
    array: &ArrayRef,
    pattern: &ArrayRef,
    options: LikeOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    assert_eq!(
        array.len(),
        pattern.len(),
        "LIKE: length mismatch for {}",
        array.encoding_id()
    );
    let len = array.len();
    let nullability =
        Nullability::from(array.dtype().is_nullable() || pattern.dtype().is_nullable());

    if len == 0 {
        return Ok(Canonical::empty(&DType::Bool(nullability)).into_array());
    }

    if let Some(pattern_const) = pattern.as_constant() {
        let Some(pattern_str) = pattern_const.as_utf8().value() else {
            // A null pattern makes every row null.
            return Ok(
                ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len)
                    .into_array(),
            );
        };
        let values = array.clone().execute::<VarBinViewArray>(ctx)?;
        let haystack = values.resolved_views();
        // The ASCII case-insensitive fast paths are only sound when the haystack is pure
        // ASCII; see `LikePattern::compile`.
        let ascii_haystack =
            options.case_insensitive && pattern_str.as_str().is_ascii() && is_ascii(&haystack);
        let compiled = LikePattern::compile(
            pattern_str.as_str(),
            options.case_insensitive,
            ascii_haystack,
        )?;
        let bits = eval_pattern(&haystack, &compiled, options.negated, ctx.allocator());
        let validity = values.validity()?.union_nullability(nullability);
        return Ok(BoolArray::new(bits, validity).into_array());
    }

    // Per-row patterns: compile each pattern individually.
    let values = array.clone().execute::<VarBinViewArray>(ctx)?;
    let patterns = pattern.clone().execute::<VarBinViewArray>(ctx)?;
    let haystack = values.resolved_views();
    let pattern_views = patterns.resolved_views();
    let ascii_haystack = options.case_insensitive && is_ascii(&haystack);

    // Reuse the previous row's compiled pattern while the pattern bytes repeat, so runs of
    // identical patterns (the common case for non-constant pattern children) compile once.
    let mut cached: Option<(&[u8], LikePattern)> = None;
    let mut bits = BitBufferMut::with_capacity_in(len, ctx.allocator().clone());
    for i in 0..len {
        let pattern_bytes = pattern_views.bytes(i);
        let compiled = match &cached {
            Some((bytes, compiled)) if *bytes == pattern_bytes => compiled,
            _ => {
                let pattern_str = std::str::from_utf8(pattern_bytes)
                    .map_err(|e| vortex_err!("LIKE pattern is not valid UTF-8: {e}"))?;
                let compiled = LikePattern::compile(
                    pattern_str,
                    options.case_insensitive,
                    ascii_haystack && pattern_str.is_ascii(),
                )?;
                &cached.insert((pattern_bytes, compiled)).1
            }
        };
        bits.append(compiled.matches(haystack.bytes(i)) != options.negated);
    }
    let validity = values
        .validity()?
        .and(patterns.validity()?)?
        .union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}

/// Evaluate `pattern` against every element of `haystack`.
///
/// The equality, prefix, and suffix patterns exploit the view layout: a view stores the value
/// length and its first four bytes inline, which settles most elements without touching the
/// data buffers (values of up to 12 bytes are stored entirely inline).
fn eval_pattern(
    haystack: &ResolvedViews<'_>,
    pattern: &LikePattern,
    negated: bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let len = haystack.len();
    match pattern {
        LikePattern::Eq(needle) if needle.len() <= BinaryView::MAX_INLINED_SIZE => {
            // The needle fits in a view, so equality is a single 16-byte comparison: a view
            // of a different length or prefix can never share the same bit pattern.
            let needle_view = BinaryView::new_inlined(needle).as_u128();
            BitBuffer::collect_bool_in(
                len,
                |i| (haystack.views()[i].as_u128() == needle_view) != negated,
                allocator.clone(),
            )
        }
        LikePattern::Eq(needle) => {
            // Compare the view head (length plus 4-byte prefix) first; only views that agree
            // on both dereference their data buffer for the remaining bytes.
            let needle_head = BinaryView::head_of(needle);
            BitBuffer::collect_bool_in(
                len,
                |i| {
                    let view = &haystack.views()[i];
                    let matched =
                        view.head() == needle_head && haystack.bytes(i)[4..] == needle[4..];
                    matched != negated
                },
                allocator.clone(),
            )
        }
        LikePattern::StartsWith(needle) => {
            // A branch-free masked comparison of the view's inline 4-byte prefix rejects
            // almost every element without touching the data buffers; only elements whose
            // prefix matches compare the remaining needle bytes.
            let needle_len = needle.len();
            let prefix_len = needle_len.min(4);
            // Raw prefix bytes, unswapped: this is a masked equality, not an ordering.
            let needle_prefix = u32::from_le_bytes(BinaryView::prefix_of(needle));
            let prefix_mask = if prefix_len == 4 {
                u32::MAX
            } else {
                (1u32 << (8 * prefix_len)) - 1
            };
            BitBuffer::collect_bool_in(
                len,
                |i| {
                    let view = &haystack.views()[i];
                    let matched = view.len() as usize >= needle_len
                        && (u32::from_le_bytes(view.prefix()) & prefix_mask) == needle_prefix
                        && (needle_len <= 4 || haystack.bytes(i)[4..needle_len] == needle[4..]);
                    matched != negated
                },
                allocator.clone(),
            )
        }
        LikePattern::EndsWith(needle) => {
            // Inlined values compare their suffix inside the view struct without touching the
            // data buffers; reference views slice exactly the suffix out of their buffer.
            let needle_len = needle.len();
            BitBuffer::collect_bool_in(
                len,
                |i| {
                    // SAFETY: `i` is below the array length, and the suffix length is only read
                    // once the view is known to be at least `needle_len` long.
                    let matched = unsafe {
                        let view = haystack.views().get_unchecked(i);
                        view.len() as usize >= needle_len
                            && bytes_eq(suffix_bytes_unchecked(haystack, view, needle_len), needle)
                    };
                    matched != negated
                },
                allocator.clone(),
            )
        }
        LikePattern::IEqAscii(needle) => BitBuffer::collect_bool_in(
            len,
            |i| {
                let view = &haystack.views()[i];
                let matched = view.len() as usize == needle.len()
                    && haystack.bytes(i).eq_ignore_ascii_case(needle);
                matched != negated
            },
            allocator.clone(),
        ),
        LikePattern::Contains(finder, needle_len) if searches_buffers(haystack) => {
            contains_in_buffers(haystack, finder, *needle_len, negated, allocator)
        }
        LikePattern::Contains(finder, needle_len) => BitBuffer::collect_bool_in(
            len,
            |i| {
                let view = &haystack.views()[i];
                let matched =
                    view.len() as usize >= *needle_len && finder.find(haystack.bytes(i)).is_some();
                matched != negated
            },
            allocator.clone(),
        ),
        _ => BitBuffer::collect_bool_in(
            len,
            |i| pattern.matches(haystack.bytes(i)) != negated,
            allocator.clone(),
        ),
    }
}

/// Whether to search for a substring through the data buffers of `haystack` rather than value
/// by value: the values outside the views must follow each other in their buffers in row order,
/// and fill at least half of the bytes between the first and the last of them.
fn searches_buffers(haystack: &ResolvedViews<'_>) -> bool {
    let mut value_bytes = 0usize;
    // The start of the first value and the end of the last value in each buffer.
    let mut spans = vec![(usize::MAX, 0usize); haystack.buffers().len()];
    for view in haystack.views() {
        if view.is_inlined() {
            continue;
        }
        let view = view.as_view();
        let range = view.as_range();
        let (start, end) = &mut spans[view.buffer_index as usize];
        if *start == usize::MAX {
            *start = range.start;
        } else if range.start < *end {
            return false;
        }
        *end = range.end;
        value_bytes += range.len();
    }
    let span_bytes: usize = spans
        .iter()
        .filter(|(start, _)| *start != usize::MAX)
        .map(|(start, end)| end - start)
        .sum();
    value_bytes > 0 && 2 * value_bytes >= span_bytes
}

/// Evaluate a `%needle%` pattern by searching the data buffers of `haystack` once.
///
/// Starting a substring search costs more than searching a short value, so searching each value
/// on its own spends most of its time starting searches. Values that follow each other in a
/// buffer, as decoding lays them out, share one search instead: the first match at or after a
/// value's start tells whether the value holds the needle, and serves every later value that
/// starts before it. See [`searches_buffers`] for when the values are laid out that way.
fn contains_in_buffers(
    haystack: &ResolvedViews<'_>,
    finder: &Finder<'static>,
    needle_len: usize,
    negated: bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    // For each buffer, the offset that its last search started at and the first match at or after
    // that offset, or `usize::MAX` without one.
    let mut searches = vec![(usize::MAX, usize::MAX); haystack.buffers().len()];
    BitBuffer::collect_bool_in(
        haystack.len(),
        |i| {
            let view = &haystack.views()[i];
            let len = view.len() as usize;
            let matched = if len < needle_len {
                false
            } else if view.is_inlined() {
                finder.find(view.as_inlined().value()).is_some()
            } else {
                let view = view.as_view();
                let buffer = haystack.buffers()[view.buffer_index as usize];
                let (search_start, next_match) = &mut searches[view.buffer_index as usize];
                let start = view.offset as usize;
                // A search from `search_start` finds the first match at or after `start` too while
                // `start` lies between the two.
                if start < *search_start || start > *next_match {
                    *search_start = start;
                    *next_match = finder
                        .find(&buffer[start..])
                        .map_or(usize::MAX, |position| start + position);
                }
                *next_match != usize::MAX && *next_match + needle_len <= start + len
            };
            matched != negated
        },
        allocator.clone(),
    )
}

/// Whether every value in `haystack` is ASCII. Validity is not consulted.
fn is_ascii(haystack: &ResolvedViews<'_>) -> bool {
    haystack
        .views()
        .iter()
        .all(|view| haystack.view_bytes(view).is_ascii())
}

/// The last `suffix_len` bytes of `view`, which must belong to `haystack`.
///
/// # Safety
///
/// `suffix_len` must be at most `view.len()`.
#[inline]
unsafe fn suffix_bytes_unchecked<'a>(
    haystack: &ResolvedViews<'a>,
    view: &'a BinaryView,
    suffix_len: usize,
) -> &'a [u8] {
    let len = view.len() as usize;
    if view.is_inlined() {
        // SAFETY: caller guarantees suffix_len <= len.
        unsafe { view.as_inlined().value().get_unchecked(len - suffix_len..) }
    } else {
        let view = view.as_view();
        let end = view.offset as usize + len;
        // SAFETY: the array validated this view's buffer index and range on construction.
        unsafe {
            haystack
                .buffers()
                .get_unchecked(view.buffer_index as usize)
                .get_unchecked(end - suffix_len..end)
        }
    }
}

/// Byte equality as an inlined loop with early exit.
///
/// Faster than slice `==` for the short, unpredictable-length comparisons in this module,
/// which otherwise lower to a `memcmp` libcall per element.
#[inline]
fn bytes_eq(lhs: &[u8], rhs: &[u8]) -> bool {
    lhs.len() == rhs.len() && std::iter::zip(lhs, rhs).all(|(l, r)| l == r)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::get_item;
    use crate::expr::like;
    use crate::expr::lit;
    use crate::expr::not_ilike;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::like::Like;
    use crate::scalar_fn::fns::like::LikeOptions;
    use crate::scalar_fn::fns::like::is_ascii;
    use crate::scalar_fn::fns::like::searches_buffers;
    use crate::scalar_fn::fns::like::suffix_bytes_unchecked;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    const LONG: &str = "a value far too long to live inside its own view";

    /// Values that a search through their buffer must tell apart: the needle at the start, at
    /// the end, inline, across two values, and not at all.
    const SEARCHED: [&str; 7] = [
        "http://www.google.com/search",
        "a long value that ends in goo",
        "gle starts this long value",
        "short google",
        "a long value that ends with google",
        "a long value without the needle",
        "google",
    ];

    fn search(values: crate::ArrayRef, negated: bool) -> crate::ArrayRef {
        let len = values.len();
        let options = LikeOptions {
            negated,
            ..Default::default()
        };
        run_like(
            values,
            ConstantArray::new("%google%", len).into_array(),
            options,
        )
    }

    #[rstest]
    fn contains_searches_buffers(#[values(false, true)] negated: bool) {
        let mut ctx = SESSION.create_execution_ctx();
        let values = VarBinViewArray::from_iter_str(SEARCHED);
        assert!(searches_buffers(&values.resolved_views()));
        let expected = SEARCHED.map(|value| value.contains("google") != negated);
        assert_arrays_eq!(
            search(values.into_array(), negated),
            BoolArray::from_iter(expected),
            &mut ctx
        );
    }

    #[rstest]
    fn contains_searches_values_out_of_order(
        #[values(false, true)] negated: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let reversed = VarBinViewArray::from_iter_str(SEARCHED)
            .into_array()
            .take(buffer![6u32, 5, 4, 3, 2, 1, 0].into_array())?
            .execute::<VarBinViewArray>(&mut ctx)?;
        assert!(!searches_buffers(&reversed.resolved_views()));
        let mut expected = SEARCHED.map(|value| value.contains("google") != negated);
        expected.reverse();
        assert_arrays_eq!(
            search(reversed.into_array(), negated),
            BoolArray::from_iter(expected),
            &mut ctx
        );
        Ok(())
    }

    fn run_like(
        array: crate::ArrayRef,
        pattern: crate::ArrayRef,
        options: LikeOptions,
    ) -> crate::ArrayRef {
        Like::try_new(array, pattern, options).unwrap().into_array()
    }

    #[rstest]
    // Exact, prefix, suffix, contains fast paths.
    #[case("hello", [true, false, false, false])]
    #[case("he%", [true, false, true, false])]
    #[case("%llo", [true, false, false, true])]
    #[case("%ell%", [true, false, false, true])]
    // Wildcards that require the regex path.
    #[case("h_llo", [true, false, false, false])]
    #[case("h%o", [true, false, false, false])]
    #[case("%", [true, true, true, true])]
    #[case("_____", [true, true, false, true])]
    fn test_like_patterns(#[case] pattern: &str, #[case] expected: [bool; 4]) {
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            VarBinViewArray::from_iter_str(["hello", "world", "help", "jello"]).into_array();
        let result = run_like(
            array,
            ConstantArray::new(pattern, 4).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
    }

    #[test]
    fn test_like_escapes() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["100%", "100x", "a_b", "axb"]).into_array();

        let result = run_like(
            array.clone(),
            ConstantArray::new(r"100\%", 4).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, false, false, false]),
            &mut ctx
        );

        let result = run_like(
            array,
            ConstantArray::new(r"a\_b", 4).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, true, false]),
            &mut ctx
        );
    }

    #[test]
    fn test_like_regex_meta_characters_are_literal() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["a.c", "abc", "a$c"]).into_array();
        let result = run_like(
            array,
            ConstantArray::new("a.%", 3).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);
    }

    #[test]
    fn test_like_unicode() {
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            VarBinViewArray::from_iter_str(["h\u{00a3}llo", "hxllo", "h\u{00a3}xllo"]).into_array();
        // `_` matches exactly one character of any byte width.
        let result = run_like(
            array,
            ConstantArray::new("h_llo", 3).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, false]), &mut ctx);
    }

    #[test]
    fn test_nlike() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        let result = run_like(
            array,
            ConstantArray::new("he%", 2).into_array(),
            LikeOptions {
                negated: true,
                case_insensitive: false,
            },
        );
        assert_arrays_eq!(result, BoolArray::from_iter([false, true]), &mut ctx);
    }

    #[test]
    fn test_ilike() {
        let mut ctx = SESSION.create_execution_ctx();
        let ilike = LikeOptions {
            negated: false,
            case_insensitive: true,
        };

        // Pure-ASCII data takes the ASCII fast paths.
        let array = VarBinViewArray::from_iter_str(["HELLO", "world", "Help"]).into_array();
        let result = run_like(array, ConstantArray::new("he%", 3).into_array(), ilike);
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true]), &mut ctx);

        // Non-ASCII data requires full case folding: `k` matches U+212A KELVIN SIGN.
        let array = VarBinViewArray::from_iter_str(["\u{212a}", "k", "x"]).into_array();
        let result = run_like(array, ConstantArray::new("k", 3).into_array(), ilike);
        assert_arrays_eq!(result, BoolArray::from_iter([true, true, false]), &mut ctx);
    }

    #[test]
    fn test_nilike() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["HELLO", "world"]).into_array();
        let result = run_like(
            array,
            ConstantArray::new("he%", 2).into_array(),
            LikeOptions {
                negated: true,
                case_insensitive: true,
            },
        );
        assert_arrays_eq!(result, BoolArray::from_iter([false, true]), &mut ctx);
    }

    #[test]
    fn test_like_nullable_input() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("help")])
            .into_array();
        let result = run_like(
            array,
            ConstantArray::new("he%", 3).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), None, Some(true)]),
            &mut ctx
        );
    }

    #[test]
    fn test_like_null_pattern() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        let result = run_like(
            array,
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), 2).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(result, BoolArray::from_iter([None, None]), &mut ctx);
    }

    #[test]
    fn test_like_per_row_patterns() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["hello", "hello", "hello"]).into_array();
        let patterns = VarBinViewArray::from_iter_str(["he%", "%world", "h_llo"]).into_array();
        let result = run_like(array, patterns, LikeOptions::default());
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true]), &mut ctx);
    }

    #[test]
    fn test_like_non_canonical_input() {
        let mut ctx = SESSION.create_execution_ctx();
        // VarBin (not the canonical VarBinView) is canonicalized before evaluation.
        let array = VarBinArray::from_iter(
            [Some("hello"), Some("world")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let result = run_like(
            array,
            ConstantArray::new("he%", 2).into_array(),
            LikeOptions::default(),
        );
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), Some(false)]),
            &mut ctx
        );
    }

    #[test]
    fn dtype() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let like_expr = like(root(), lit("%test%"));
        assert_eq!(
            like_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn signature() {
        let like_expr = like(root(), lit("%test%"));
        assert!(
            like_expr
                .as_scalar()
                .is_some_and(|f| f.signature().is_strict())
        );
        assert!(
            like_expr
                .as_scalar()
                .is_some_and(|f| f.signature().is_infallible())
        );
    }

    #[test]
    fn test_display() {
        let expr = like(get_item("name", root()), lit("%john%"));
        assert_eq!(expr.to_string(), "$.name like \"%john%\"");

        let expr2 = not_ilike(root(), lit("test*"));
        assert_eq!(expr2.to_string(), "$ not ilike \"test*\"");
    }

    #[test]
    fn suffix_bytes_are_value_suffixes() {
        let array = VarBinViewArray::from_iter_str(["short", LONG, "another long value here"]);
        let haystack = array.resolved_views();
        for (index, view) in haystack.views().iter().enumerate() {
            let value = haystack.bytes(index);
            for suffix_len in 0..=value.len() {
                // SAFETY: suffix_len <= value.len(), which is view.len().
                let suffix = unsafe { suffix_bytes_unchecked(&haystack, view, suffix_len) };
                assert_eq!(suffix, &value[value.len() - suffix_len..]);
            }
        }
    }

    #[rstest]
    #[case::inlined(["short", "tiny"], true)]
    #[case::referenced([LONG, "another long value here"], true)]
    #[case::inlined_non_ascii(["short", "é"], false)]
    #[case::referenced_non_ascii([LONG, "a long value ending in é"], false)]
    fn detects_ascii(#[case] values: [&str; 2], #[case] expected: bool) {
        let array = VarBinViewArray::from_iter_str(values);
        assert_eq!(is_ascii(&array.resolved_views()), expected);
    }
}
