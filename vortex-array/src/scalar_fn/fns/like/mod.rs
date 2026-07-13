// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;
mod pattern;

use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use pattern::LikePattern;
use prost::Message;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::and;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

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
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        if options.negated {
            write!(f, " not")?;
        }
        if options.case_insensitive {
            write!(f, " ilike ")?;
        } else {
            write!(f, " like ")?;
        }
        expr.child(1).fmt_sql(f)
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

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        tracing::warn!("Computing validity for LIKE expression");
        let child_validity = expression.child(0).validity()?;
        let pattern_validity = expression.child(1).validity()?;
        Ok(Some(and(child_validity, pattern_validity)))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
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
        let haystack = ResolvedViews::new(&values);
        // The ASCII case-insensitive fast paths are only sound when the haystack is pure
        // ASCII; see `LikePattern::compile`.
        let ascii_haystack =
            options.case_insensitive && pattern_str.as_str().is_ascii() && haystack.is_ascii();
        let compiled = LikePattern::compile(
            pattern_str.as_str(),
            options.case_insensitive,
            ascii_haystack,
        )?;
        let bits = eval_pattern(&haystack, &compiled, options.negated);
        let validity = values.validity()?.union_nullability(nullability);
        return Ok(BoolArray::new(bits, validity).into_array());
    }

    // Per-row patterns: compile each pattern individually.
    let values = array.clone().execute::<VarBinViewArray>(ctx)?;
    let patterns = pattern.clone().execute::<VarBinViewArray>(ctx)?;
    let haystack = ResolvedViews::new(&values);
    let pattern_views = ResolvedViews::new(&patterns);
    let ascii_haystack = options.case_insensitive && haystack.is_ascii();

    let mut bits = Vec::with_capacity(len);
    // Reuse the previous row's compiled pattern while the pattern bytes repeat, so runs of
    // identical patterns (the common case for non-constant pattern children) compile once.
    let mut cached: Option<(&[u8], LikePattern)> = None;
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
        bits.push(compiled.matches(haystack.bytes(i)) != options.negated);
    }
    let validity = values
        .validity()?
        .and(patterns.validity()?)?
        .union_nullability(nullability);
    Ok(BoolArray::new(BitBuffer::from_iter(bits), validity).into_array())
}

/// Resolved views over a canonical [`VarBinViewArray`]: the view structs plus borrowed slices
/// of every data buffer, supporting cheap per-element byte access.
struct ResolvedViews<'a> {
    views: &'a [BinaryView],
    buffers: Vec<&'a [u8]>,
}

impl<'a> ResolvedViews<'a> {
    fn new(array: &'a VarBinViewArray) -> Self {
        Self {
            views: array.views(),
            buffers: (0..array.data_buffers().len())
                .map(|idx| array.buffer(idx).as_slice())
                .collect(),
        }
    }

    #[inline]
    fn bytes(&self, index: usize) -> &'a [u8] {
        let view = &self.views[index];
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let view = view.as_view();
            &self.buffers[view.buffer_index as usize][view.as_range()]
        }
    }

    /// Whether every value (including values under a null) is pure ASCII.
    fn is_ascii(&self) -> bool {
        (0..self.views.len()).all(|i| self.bytes(i).is_ascii())
    }

    /// The last `suffix_len` bytes of `view`, which must belong to this array.
    ///
    /// # Safety
    ///
    /// `suffix_len` must be at most `view.len()`. Buffer bounds are guaranteed by
    /// [`VarBinViewArray::validate`], which checks every view against its data buffer at
    /// construction.
    #[inline]
    unsafe fn suffix_bytes_unchecked(&self, view: &'a BinaryView, suffix_len: usize) -> &'a [u8] {
        let len = view.len() as usize;
        if view.is_inlined() {
            // SAFETY: inlined values hold `len <= 12` value bytes, and the caller
            // guarantees `suffix_len <= len`.
            unsafe { view.as_inlined().value().get_unchecked(len - suffix_len..) }
        } else {
            let view = view.as_view();
            let end = view.offset as usize + len;
            // SAFETY: validated views reference `buffer_index < buffers.len()` and bytes
            // `offset..offset + len` within that buffer.
            unsafe {
                self.buffers
                    .get_unchecked(view.buffer_index as usize)
                    .get_unchecked(end - suffix_len..end)
            }
        }
    }
}

/// Evaluate `pattern` against every element of `haystack`.
///
/// The equality, prefix, and suffix patterns exploit the view layout: a view stores the value
/// length and its first four bytes inline, which settles most elements without touching the
/// data buffers (values of up to 12 bytes are stored entirely inline).
fn eval_pattern(haystack: &ResolvedViews<'_>, pattern: &LikePattern, negated: bool) -> BitBuffer {
    let len = haystack.views.len();
    match pattern {
        LikePattern::Eq(needle) if needle.len() <= BinaryView::MAX_INLINED_SIZE => {
            // The needle fits in a view, so equality is a single 16-byte comparison: a view
            // of a different length or prefix can never share the same bit pattern.
            let needle_view = BinaryView::new_inlined(needle).as_u128();
            BitBuffer::collect_bool(len, |i| {
                (haystack.views[i].as_u128() == needle_view) != negated
            })
        }
        LikePattern::Eq(needle) => {
            // Compare the view head (length plus 4-byte prefix) first; only views that agree
            // on both dereference their data buffer for the remaining bytes.
            let needle_head = needle_head(needle);
            BitBuffer::collect_bool(len, |i| {
                let view = &haystack.views[i];
                let matched =
                    view_head(view) == needle_head && haystack.bytes(i)[4..] == needle[4..];
                matched != negated
            })
        }
        LikePattern::StartsWith(needle) => {
            // A branch-free masked comparison of the view's inline 4-byte prefix rejects
            // almost every element without touching the data buffers; only elements whose
            // prefix matches compare the remaining needle bytes.
            let needle_len = needle.len();
            let prefix_len = needle_len.min(4);
            let needle_prefix = u32::from_le_bytes({
                let mut padded = [0u8; 4];
                padded[..prefix_len].copy_from_slice(&needle[..prefix_len]);
                padded
            });
            let prefix_mask = if prefix_len == 4 {
                u32::MAX
            } else {
                (1u32 << (8 * prefix_len)) - 1
            };
            BitBuffer::collect_bool(len, |i| {
                let view = &haystack.views[i];
                let matched = view.len() as usize >= needle_len
                    && (view_prefix(view) & prefix_mask) == needle_prefix
                    && (needle_len <= 4 || haystack.bytes(i)[4..needle_len] == needle[4..]);
                matched != negated
            })
        }
        LikePattern::EndsWith(needle) => {
            // Inlined values compare their suffix inside the view struct without touching the
            // data buffers; reference views slice exactly the suffix out of their buffer.
            let needle_len = needle.len();
            BitBuffer::collect_bool(len, |i| {
                // SAFETY: `i` is below the array length, and the suffix length is only read
                // once the view is known to be at least `needle_len` long.
                let matched = unsafe {
                    let view = haystack.views.get_unchecked(i);
                    view.len() as usize >= needle_len
                        && bytes_eq(haystack.suffix_bytes_unchecked(view, needle_len), needle)
                };
                matched != negated
            })
        }
        LikePattern::IEqAscii(needle) => BitBuffer::collect_bool(len, |i| {
            let view = &haystack.views[i];
            let matched = view.len() as usize == needle.len()
                && haystack.bytes(i).eq_ignore_ascii_case(needle);
            matched != negated
        }),
        LikePattern::Contains(finder, needle_len) => BitBuffer::collect_bool(len, |i| {
            let view = &haystack.views[i];
            let matched =
                view.len() as usize >= *needle_len && finder.find(haystack.bytes(i)).is_some();
            matched != negated
        }),
        _ => BitBuffer::collect_bool(len, |i| pattern.matches(haystack.bytes(i)) != negated),
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

/// The leading 8 bytes of a view: the `u32` length plus the first 4 bytes of the value
/// (zero-padded for values shorter than 4 bytes).
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
fn view_head(view: &BinaryView) -> u64 {
    view.as_u128() as u64
}

/// The view head a needle of more than 4 bytes would have: its length plus first 4 bytes.
fn needle_head(needle: &[u8]) -> u64 {
    let prefix: [u8; 4] = [needle[0], needle[1], needle[2], needle[3]];
    (needle.len() as u64) | (u64::from(u32::from_le_bytes(prefix)) << 32)
}

/// The first 4 value bytes stored inline in any view (zero-padded for values shorter than
/// 4 bytes), as a raw little-endian `u32` in memory order.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
fn view_prefix(view: &BinaryView) -> u32 {
    (view.as_u128() >> 32) as u32
}

/// Variants of the LIKE filter that we know how to turn into a stats pruning predicate.
#[derive(Debug, PartialEq)]
pub(crate) enum LikeVariant<'a> {
    Exact(Cow<'a, str>),
    Prefix(Cow<'a, str>),
}

impl<'a> LikeVariant<'a> {
    /// Parse a LIKE pattern string into its relevant variant
    pub(crate) fn from_str(string: &'a str) -> Option<LikeVariant<'a>> {
        let mut literal = None;
        let mut chars = string.char_indices();

        while let Some((idx, c)) = chars.next() {
            match c {
                '\\' => {
                    let literal = literal.get_or_insert_with(|| string[..idx].to_string());
                    match chars.next() {
                        Some((_, escaped)) => literal.push(escaped),
                        None => literal.push('\\'),
                    }
                }
                '%' | '_' => {
                    return match literal {
                        Some(literal) => (!literal.is_empty())
                            .then_some(LikeVariant::Prefix(Cow::Owned(literal))),
                        None => {
                            (idx != 0).then_some(LikeVariant::Prefix(Cow::Borrowed(&string[..idx])))
                        }
                    };
                }
                c => {
                    if let Some(literal) = &mut literal {
                        literal.push(c);
                    }
                }
            }
        }

        Some(match literal {
            Some(literal) => LikeVariant::Exact(Cow::Owned(literal)),
            None => LikeVariant::Exact(Cow::Borrowed(string)),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rstest::rstest;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::get_item;
    use crate::expr::like;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::not_ilike;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::like::Like;
    use crate::scalar_fn::fns::like::LikeOptions;
    use crate::scalar_fn::fns::like::LikeVariant;

    fn run_like(
        array: crate::ArrayRef,
        pattern: crate::ArrayRef,
        options: LikeOptions,
    ) -> crate::ArrayRef {
        let len = array.len();
        Like.try_new_array(len, options, [array, pattern]).unwrap()
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["hello", "hello", "hello"]).into_array();
        let patterns = VarBinViewArray::from_iter_str(["he%", "%world", "h_llo"]).into_array();
        let result = run_like(array, patterns, LikeOptions::default());
        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true]), &mut ctx);
    }

    #[test]
    fn test_like_non_canonical_input() {
        let mut ctx = array_session().create_execution_ctx();
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
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            bools.into_array().apply(&not_expr).unwrap(),
            BoolArray::from_iter([true, false, true, true, false, false]),
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
        assert!(!like_expr.signature().is_null_sensitive());
        assert!(!like_expr.signature().is_fallible());
    }

    #[test]
    fn test_display() {
        let expr = like(get_item("name", root()), lit("%john%"));
        assert_eq!(expr.to_string(), "$.name like \"%john%\"");

        let expr2 = not_ilike(root(), lit("test*"));
        assert_eq!(expr2.to_string(), "$ not ilike \"test*\"");
    }

    fn assert_borrowed_exact(pattern: &str, expected: &str) {
        let Some(LikeVariant::Exact(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected borrowed exact pattern");
        };
        assert!(matches!(actual, Cow::Borrowed(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    fn assert_owned_exact(pattern: &str, expected: &str) {
        let Some(LikeVariant::Exact(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected owned exact pattern");
        };
        assert!(matches!(actual, Cow::Owned(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    fn assert_borrowed_prefix(pattern: &str, expected: &str) {
        let Some(LikeVariant::Prefix(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected borrowed prefix pattern");
        };
        assert!(matches!(actual, Cow::Borrowed(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    fn assert_owned_prefix(pattern: &str, expected: &str) {
        let Some(LikeVariant::Prefix(actual)) = LikeVariant::from_str(pattern) else {
            panic!("expected owned prefix pattern");
        };
        assert!(matches!(actual, Cow::Owned(_)));
        assert_eq!(actual.as_ref(), expected);
    }

    #[test]
    fn test_like_variant_borrowed_patterns() {
        assert_borrowed_exact("simple", "simple");
        assert_borrowed_prefix("prefix%", "prefix");
        assert_borrowed_prefix("first%rest_stuff", "first");
    }

    #[test]
    fn test_like_variant_escaped_patterns() {
        assert_owned_prefix(r"\%%", "%");
        assert_owned_prefix(r"\_%", "_");
        assert_owned_prefix(r"\\%", "\\");
        assert_owned_exact(r"\%", "%");
        assert_owned_exact("trailing\\", "trailing\\");
    }

    #[test]
    fn test_like_variant_unsupported_patterns() {
        assert_eq!(LikeVariant::from_str("%suffix"), None);
        assert_eq!(LikeVariant::from_str(r"%\%%"), None);
        assert_eq!(LikeVariant::from_str("_pattern"), None);
    }
}
