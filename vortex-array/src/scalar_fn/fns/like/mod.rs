// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrow::Datum;
use crate::arrow::from_arrow_columnar;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::and;
use crate::expr::gt;
use crate::expr::gt_eq;
use crate::expr::lit;
use crate::expr::lt;
use crate::expr::or;
use crate::scalar::StringLike;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::literal::Literal;

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

        arrow_like(&child, &pattern, *options, ctx)
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

    fn stat_falsification(
        &self,
        like_opts: &LikeOptions,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // Attempt to do min/max pruning for LIKE 'exact' or LIKE 'prefix%'

        // Don't attempt to handle ilike or negated like
        if like_opts.negated || like_opts.case_insensitive {
            return None;
        }

        // Extract the pattern out
        let pat = expr.child(1).as_::<Literal>();

        // LIKE NULL is nonsensical, don't try to handle it
        let pat_str = pat.as_utf8().value()?;

        let src = expr.child(0).clone();
        let src_min = src.stat_min(catalog)?;
        let src_max = src.stat_max(catalog)?;

        match LikeVariant::from_str(pat_str)? {
            LikeVariant::Exact(text) => {
                // col LIKE 'exact' ==>  col.min > 'exact' || col.max < 'exact'
                Some(or(
                    gt(src_min, lit(text.as_ref())),
                    lt(src_max, lit(text.as_ref())),
                ))
            }
            LikeVariant::Prefix(prefix) => {
                // col LIKE 'prefix%' ==> col.max < 'prefix' || col.min >= 'prefiy'
                let succ = prefix.to_string().increment().ok()?;

                Some(or(
                    gt_eq(src_min, lit(succ)),
                    lt(src_max, lit(prefix.as_ref())),
                ))
            }
        }
    }
}

/// Implementation of LIKE using the Arrow crate.
pub(crate) fn arrow_like(
    array: &ArrayRef,
    pattern: &ArrayRef,
    options: LikeOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = array.dtype().is_nullable() | pattern.dtype().is_nullable();
    let len = array.len();
    assert_eq!(
        array.len(),
        pattern.len(),
        "Arrow Like: length mismatch for {}",
        array.encoding_id()
    );

    // convert the pattern to the preferred array datatype
    let lhs = Datum::try_new(array, ctx)?;
    let rhs = Datum::try_new_with_target_datatype(pattern, lhs.data_type(), ctx)?;

    let result = match (options.negated, options.case_insensitive) {
        (false, false) => arrow_string::like::like(&lhs, &rhs)?,
        (true, false) => arrow_string::like::nlike(&lhs, &rhs)?,
        (false, true) => arrow_string::like::ilike(&lhs, &rhs)?,
        (true, true) => arrow_string::like::nilike(&lhs, &rhs)?,
    };

    from_arrow_columnar(&result, len, nullable, ctx)
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

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::ilike;
    use crate::expr::like;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::not_ilike;
    use crate::expr::not_like;
    use crate::expr::pruning::pruning_expr::TrackingStatsCatalog;
    use crate::expr::root;
    use crate::scalar_fn::fns::like::LikeVariant;

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_arrays_eq!(
            bools.into_array().apply(&not_expr).unwrap(),
            BoolArray::from_iter([true, false, true, true, false, false])
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

    #[test]
    fn test_like_pushdown() {
        // Test that LIKE prefix and exactness filters can be pushed down into stats filtering
        // at scan time.
        let catalog = TrackingStatsCatalog::default();

        let pruning_expr = like(col("a"), lit("prefix%"))
            .stat_falsification(&catalog)
            .expect("LIKE stat falsification");

        insta::assert_snapshot!(pruning_expr, @r#"(($.a_min >= "prefiy") or ($.a_max < "prefix"))"#);

        let pruning_expr = like(col("a"), lit(r"\%%"))
            .stat_falsification(&catalog)
            .expect("LIKE stat falsification");
        insta::assert_snapshot!(pruning_expr, @r#"(($.a_min >= "&") or ($.a_max < "%"))"#);

        // Multiple wildcards
        let pruning_expr = like(col("a"), lit("pref%ix%"))
            .stat_falsification(&catalog)
            .expect("LIKE stat falsification");
        insta::assert_snapshot!(pruning_expr, @r#"(($.a_min >= "preg") or ($.a_max < "pref"))"#);

        let pruning_expr = like(col("a"), lit("pref_ix_"))
            .stat_falsification(&catalog)
            .expect("LIKE stat falsification");
        insta::assert_snapshot!(pruning_expr, @r#"(($.a_min >= "preg") or ($.a_max < "pref"))"#);

        // Exact match
        let pruning_expr = like(col("a"), lit("exactly"))
            .stat_falsification(&catalog)
            .expect("LIKE stat falsification");
        insta::assert_snapshot!(pruning_expr, @r#"(($.a_min > "exactly") or ($.a_max < "exactly"))"#);

        // Suffix search skips pushdown
        let pruning_expr = like(col("a"), lit("%suffix")).stat_falsification(&catalog);
        assert_eq!(pruning_expr, None);

        // NOT LIKE, ILIKE not supported currently
        assert_eq!(
            None,
            not_like(col("a"), lit("a")).stat_falsification(&catalog)
        );
        assert_eq!(None, ilike(col("a"), lit("a")).stat_falsification(&catalog));
    }
}
