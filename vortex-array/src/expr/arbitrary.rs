// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-directed synthesis of arbitrary, well-typed [`Expression`] trees.
//!
//! Generation runs "backwards" from a target [`DType`]: for every target dtype the
//! [`Synthesizer`] collects the [`Rule`]s that can produce it (type synthesis), picks one, and
//! the rule recursively synthesizes children for the argument dtypes it demands. Every
//! synthesized tree is then verified with [`Expression::return_dtype`] (type checking), so a
//! generator bug fails loudly instead of producing an ill-typed expression.
//!
//! The rule set is a flat registry: each scalar function is one [`Rule`] entry in
//! [`builtin_rules`], and new productions are registered with a single
//! [`Synthesizer::with_rule`] call.
//!
//! The default production space covers the serializable, infallible scalar functions: literals,
//! field access (root/get_item), boolean connectives, comparisons, between, like, is_(not_)null,
//! case_when, zip, fill_null, mask, byte_length, list_contains, pack, select, and merge.
//! [`SynthesisOptions::fallible`] additionally enables functions that may legitimately error at
//! runtime on valid input (cast, arithmetic operators, data-driven like patterns); callers that
//! enable it must establish a baseline by evaluating the expression eagerly and discarding inputs
//! whose baseline evaluation fails. Always excluded are functions that cannot round trip through
//! a file scan (dynamic comparisons, stat references).

use arbitrary::Arbitrary;
use arbitrary::Result as AResult;
use arbitrary::Unstructured;
use itertools::Itertools;
use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::between;
use crate::expr::byte_length;
use crate::expr::cast;
use crate::expr::fill_null;
use crate::expr::get_item;
use crate::expr::ilike;
use crate::expr::is_not_null;
use crate::expr::is_null;
use crate::expr::like;
use crate::expr::list_contains;
use crate::expr::lit;
use crate::expr::mask;
use crate::expr::merge;
use crate::expr::nested_case_when;
use crate::expr::not;
use crate::expr::not_ilike;
use crate::expr::not_like;
use crate::expr::pack;
use crate::expr::root;
use crate::expr::select;
use crate::expr::zip_expr;
use crate::scalar::Scalar;
use crate::scalar::arbitrary::random_scalar;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

/// Maximum expression tree depth (compound nodes on any root-to-leaf path).
const MAX_DEPTH: usize = 4;

/// Maximum number of scope paths (root plus nested struct fields) considered as leaves.
const MAX_PATHS: usize = 64;

/// Options controlling the synthesized production space.
#[derive(Clone, Copy, Debug, Default)]
pub struct SynthesisOptions {
    /// Include functions that may legitimately error at runtime on valid input: cast, the
    /// arithmetic operators, and like with a data-driven pattern.
    ///
    /// Callers enabling this must evaluate the synthesized expression eagerly first and discard
    /// inputs whose baseline evaluation fails, since a runtime error no longer implies a bug.
    pub fallible: bool,
}

/// Generates an arbitrary projection expression returning a struct assembled from scope columns
/// and synthesized values, or `None` to scan with the identity projection.
pub fn projection_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<Expression>> {
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };
    if u.ratio(1, 8)? {
        return Ok(None);
    }

    let field_count = u.int_in_range::<usize>(0..=struct_dtype.nfields().max(4))?;
    let mut fields = Vec::with_capacity(field_count);
    for i in 0..field_count {
        // Either project an existing column's dtype or synthesize a fresh scalar dtype. The
        // index suffix keeps the packed field names unique.
        let (name, field_dtype) = if struct_dtype.nfields() > 0 && u.ratio(2, 3)? {
            let (name, field_dtype) =
                u.choose_iter(struct_dtype.names().iter().zip(struct_dtype.fields()))?;
            (FieldName::from(format!("{name}_{i}")), field_dtype)
        } else {
            (
                FieldName::from(format!("synth_{i}")),
                random_scalar_dtype(u)?,
            )
        };
        fields.push((name, field_dtype));
    }

    let (names, dtypes): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
    let target = DType::Struct(
        StructFields::new(names.into(), dtypes),
        Nullability::arbitrary(u)?,
    );
    synthesize_expr_with(u, dtype, &target, SynthesisOptions { fallible: true }).map(Some)
}

/// Generates an arbitrary well-typed boolean filter expression over the scope, or `None` to scan
/// unfiltered.
///
/// Both this and [`projection_expr`] include fallible functions: the file fuzz harness evaluates
/// the expressions eagerly before writing the file and rejects inputs whose baseline fails.
pub fn filter_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<Expression>> {
    if dtype.as_struct_fields_opt().is_none() {
        return Ok(None);
    }
    if u.ratio(1, 8)? {
        return Ok(None);
    }
    let target = DType::Bool(Nullability::arbitrary(u)?);
    synthesize_expr_with(u, dtype, &target, SynthesisOptions { fallible: true }).map(Some)
}

/// Synthesizes an arbitrary infallible expression that evaluates to exactly `target` when applied
/// to data of dtype `scope`.
///
/// # Panics
///
/// Panics if the synthesized expression fails to type check to `target`, which indicates a bug in
/// the synthesizer rather than in the consumed bytes.
pub fn synthesize_expr(
    u: &mut Unstructured<'_>,
    scope: &DType,
    target: &DType,
) -> AResult<Expression> {
    synthesize_expr_with(u, scope, target, SynthesisOptions::default())
}

/// [`synthesize_expr`] with explicit [`SynthesisOptions`].
pub fn synthesize_expr_with(
    u: &mut Unstructured<'_>,
    scope: &DType,
    target: &DType,
    options: SynthesisOptions,
) -> AResult<Expression> {
    Synthesizer::new(scope, options).synthesize(u, target)
}

/// Whether a rule can synthesize an expression of the target dtype.
pub type RuleApplies = fn(&Synthesizer, &DType) -> bool;

/// Synthesizes an expression of exactly the target dtype. The depth is the remaining budget for
/// child expressions, to be passed through to [`Synthesizer::expr`].
pub type RuleSynth = fn(&Synthesizer, &mut Unstructured<'_>, &DType, usize) -> AResult<Expression>;

/// A synthesis rule: one production (typically one scalar function) in the expression space.
///
/// A rule pairs an applicability predicate over the target dtype with a synthesis function for
/// it, so registering a new expression is a single [`Synthesizer::with_rule`] call:
///
/// ```
/// use arbitrary::Unstructured;
/// use vortex_array::dtype::{DType, Nullability};
/// use vortex_array::expr::arbitrary::{Rule, Synthesizer, SynthesisOptions};
/// use vortex_array::expr::not;
///
/// let scope = DType::Bool(Nullability::NonNullable);
/// let synthesizer = Synthesizer::new(&scope, SynthesisOptions::default()).with_rule(Rule::new(
///     "double_not",
///     |_, target| matches!(target, DType::Bool(_)),
///     |g, u, target, depth| Ok(not(not(g.expr(u, target, depth)?))),
/// ));
/// let mut u = Unstructured::new(&[7u8; 64]);
/// let expr = synthesizer
///     .synthesize(&mut u, &DType::Bool(Nullability::NonNullable))
///     .unwrap();
/// ```
pub struct Rule {
    name: &'static str,
    /// Leaf rules are the only candidates once the depth budget is exhausted.
    leaf: bool,
    /// Fallible rules are only included when [`SynthesisOptions::fallible`] is set.
    fallible: bool,
    applies: RuleApplies,
    synth: RuleSynth,
}

impl Rule {
    /// An infallible compound rule.
    pub fn new(name: &'static str, applies: RuleApplies, synth: RuleSynth) -> Self {
        Self {
            name,
            leaf: false,
            fallible: false,
            applies,
            synth,
        }
    }

    /// A leaf rule: synthesizes without recursing, so it remains a candidate at depth zero.
    pub fn leaf(name: &'static str, applies: RuleApplies, synth: RuleSynth) -> Self {
        Self {
            leaf: true,
            ..Self::new(name, applies, synth)
        }
    }

    /// A rule that may legitimately error at runtime on valid input. Only included when
    /// [`SynthesisOptions::fallible`] is set.
    pub fn fallible(name: &'static str, applies: RuleApplies, synth: RuleSynth) -> Self {
        Self {
            fallible: true,
            ..Self::new(name, applies, synth)
        }
    }

    /// The rule's name, for debugging and coverage reporting.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// A type-directed expression synthesizer over a fixed scope dtype and rule set.
pub struct Synthesizer {
    /// Path expressions into the scope (root plus nested struct fields) and their dtypes.
    paths: Vec<(Expression, DType)>,
    options: SynthesisOptions,
    rules: Vec<Rule>,
}

impl Synthesizer {
    /// Creates a synthesizer over the scope dtype with the builtin rules for `options`.
    pub fn new(scope: &DType, options: SynthesisOptions) -> Self {
        let mut paths = Vec::new();
        collect_paths(root(), scope, &mut paths);
        let rules = builtin_rules()
            .into_iter()
            .filter(|rule| options.fallible || !rule.fallible)
            .collect();
        Self {
            paths,
            options,
            rules,
        }
    }

    /// Registers an additional rule.
    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// The options this synthesizer was created with.
    pub fn options(&self) -> SynthesisOptions {
        self.options
    }

    /// Path expressions into the scope (root plus nested struct fields) and their dtypes.
    pub fn paths(&self) -> &[(Expression, DType)] {
        &self.paths
    }

    /// Synthesizes an expression of exactly `target` with an arbitrary depth budget, verifying
    /// that the result type checks.
    ///
    /// # Panics
    ///
    /// Panics if the synthesized expression fails to type check to `target`, which indicates a
    /// bug in a rule rather than in the consumed bytes.
    pub fn synthesize(&self, u: &mut Unstructured<'_>, target: &DType) -> AResult<Expression> {
        let depth = u.int_in_range(0..=MAX_DEPTH)?;
        let expr = self.expr(u, target, depth)?;

        let scope = &self.paths[0].1;
        let actual = expr
            .return_dtype(scope)
            .vortex_expect("synthesized expression must type check against the scope");
        assert_eq!(
            &actual, target,
            "synthesized expression {expr} returned {actual}, expected {target}"
        );
        Ok(expr)
    }

    /// Synthesizes an expression of exactly `target` by dispatching to an applicable rule.
    /// Rules call this to synthesize their children.
    pub fn expr(
        &self,
        u: &mut Unstructured<'_>,
        target: &DType,
        depth: usize,
    ) -> AResult<Expression> {
        let candidates = self
            .rules
            .iter()
            .filter(|rule| (depth > 0 || rule.leaf) && (rule.applies)(self, target))
            .collect_vec();
        let rule = u.choose_iter(candidates)?;
        (rule.synth)(self, u, target, depth.saturating_sub(1))
    }

    /// Picks a non-nullable comparable dtype, biased towards dtypes present in the scope so that
    /// comparisons usually reference real columns.
    pub fn comparable_dtype(&self, u: &mut Unstructured<'_>) -> AResult<DType> {
        let scope_dtypes = self
            .paths
            .iter()
            .map(|(_, dtype)| dtype)
            .filter(|dtype| is_comparable_dtype(dtype))
            .collect_vec();
        if !scope_dtypes.is_empty() && u.ratio(2, 3)? {
            return Ok(u
                .choose_iter(scope_dtypes)?
                .with_nullability(Nullability::NonNullable));
        }
        random_comparable_dtype(u)
    }

    /// Picks any dtype for children whose dtype is unconstrained (e.g. `is_null`).
    pub fn any_dtype(&self, u: &mut Unstructured<'_>) -> AResult<DType> {
        if !self.paths.is_empty() && u.ratio(1, 2)? {
            return Ok(u.choose_iter(self.paths.iter())?.1.clone());
        }
        random_scalar_dtype(u)
    }
}

/// The builtin rule set: one [`Rule`] per scalar function the synthesizer can produce.
pub fn builtin_rules() -> Vec<Rule> {
    vec![
        Rule::leaf(
            "literal",
            |_, _| true,
            |_, u, t, _| Ok(lit(random_scalar(u, t)?)),
        ),
        Rule::leaf(
            "column",
            |g, t| g.paths.iter().any(|(_, dt)| dt == t),
            column,
        ),
        Rule::new("not", bool_target, |g, u, t, d| Ok(not(g.expr(u, t, d)?))),
        Rule::new("and_or", bool_target, and_or),
        Rule::new("compare", bool_target, compare),
        Rule::new("between", bool_target, between_),
        Rule::new("like", bool_target, like_),
        Rule::new("is_null", non_nullable_bool_target, is_null_),
        Rule::new("is_not_null", non_nullable_bool_target, is_not_null_),
        Rule::new("list_contains", list_contains_applies, list_contains_),
        Rule::new(
            "byte_length",
            |_, t| matches!(t, DType::Primitive(PType::U64, _)),
            byte_length_,
        ),
        Rule::new("pack", struct_target, pack_),
        Rule::new("select", unique_struct_target, select_),
        Rule::new(
            "merge",
            |g, t| unique_struct_target(g, t) && t.nullability() == Nullability::NonNullable,
            merge_,
        ),
        Rule::new("get_item", supported_target, get_item_),
        Rule::new("case_when", supported_target, case_when),
        Rule::new("zip", supported_target, zip),
        Rule::new("fill_null", supported_target, fill_null_),
        Rule::new(
            "mask",
            |g, t| supported_target(g, t) && t.nullability() == Nullability::Nullable,
            mask_,
        ),
        Rule::fallible(
            "arithmetic",
            |_, t| matches!(t, DType::Primitive(..)),
            arithmetic,
        ),
        Rule::fallible("cast", |_, t| is_comparable_dtype(t), cast_),
    ]
}

// ---- Rule applicability predicates ----

fn bool_target(_: &Synthesizer, target: &DType) -> bool {
    matches!(target, DType::Bool(_))
}

fn non_nullable_bool_target(_: &Synthesizer, target: &DType) -> bool {
    *target == DType::Bool(Nullability::NonNullable)
}

fn struct_target(_: &Synthesizer, target: &DType) -> bool {
    matches!(target, DType::Struct(..))
}

fn unique_struct_target(_: &Synthesizer, target: &DType) -> bool {
    target
        .as_struct_fields_opt()
        .is_some_and(|fields| fields.names().iter().all_unique())
}

/// Targets that every dtype-generic rule (get_item, case_when, zip, fill_null, mask) supports.
fn supported_target(_: &Synthesizer, target: &DType) -> bool {
    !matches!(
        target,
        DType::Null | DType::Extension(..) | DType::Union(..) | DType::Variant(..)
    )
}

fn list_contains_applies(g: &Synthesizer, target: &DType) -> bool {
    matches!(target, DType::Bool(_))
        && g.paths
            .iter()
            .any(|(_, dtype)| is_searchable_list(dtype, target.nullability()))
}

/// Whether `list_contains` over a list of this dtype can return a boolean of the given
/// nullability: the result is nullable iff the list or the needle is.
fn is_searchable_list(dtype: &DType, nullability: Nullability) -> bool {
    matches!(
        dtype,
        DType::List(element, list_nullability) if is_comparable_dtype(element)
            && (*list_nullability == Nullability::NonNullable
                || nullability == Nullability::Nullable)
    )
}

// ---- Rule synthesis functions ----

fn column(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    _: usize,
) -> AResult<Expression> {
    let matching = g
        .paths
        .iter()
        .filter(|(_, dtype)| dtype == target)
        .collect_vec();
    Ok(u.choose_iter(matching)?.0.clone())
}

fn and_or(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let op = if u.arbitrary()? {
        Operator::And
    } else {
        Operator::Or
    };
    let (lhs_n, rhs_n) = split_nullability2(u, target.nullability())?;
    Ok(Binary.new_expr(
        op,
        [
            g.expr(u, &DType::Bool(lhs_n), depth)?,
            g.expr(u, &DType::Bool(rhs_n), depth)?,
        ],
    ))
}

fn compare(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let op = comparison_operator(u)?;
    let base = g.comparable_dtype(u)?;
    let (lhs_n, rhs_n) = split_nullability2(u, target.nullability())?;
    Ok(Binary.new_expr(
        op,
        [
            g.expr(u, &base.with_nullability(lhs_n), depth)?,
            g.expr(u, &base.with_nullability(rhs_n), depth)?,
        ],
    ))
}

fn between_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let base = g.comparable_dtype(u)?;
    let (arr_n, lower_n, upper_n) = split_nullability3(u, target.nullability())?;
    Ok(between(
        g.expr(u, &base.with_nullability(arr_n), depth)?,
        g.expr(u, &base.with_nullability(lower_n), depth)?,
        g.expr(u, &base.with_nullability(upper_n), depth)?,
        BetweenOptions {
            lower_strict: strictness(u)?,
            upper_strict: strictness(u)?,
        },
    ))
}

fn like_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let (child_n, pattern_n) = split_nullability2(u, target.nullability())?;
    let child = g.expr(u, &DType::Utf8(child_n), depth)?;
    // Data-driven patterns can fail to compile (e.g. a trailing escape), so without fallible
    // mode patterns are escape-free literals.
    let pattern = if g.options.fallible {
        g.expr(u, &DType::Utf8(pattern_n), depth)?
    } else {
        like_pattern(u, pattern_n)?
    };
    Ok(match u.int_in_range(0..=3)? {
        0 => like(child, pattern),
        1 => ilike(child, pattern),
        2 => not_like(child, pattern),
        _ => not_ilike(child, pattern),
    })
}

fn is_null_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    _: &DType,
    depth: usize,
) -> AResult<Expression> {
    let child_dtype = g.any_dtype(u)?;
    Ok(is_null(g.expr(u, &child_dtype, depth)?))
}

fn is_not_null_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    _: &DType,
    depth: usize,
) -> AResult<Expression> {
    let child_dtype = g.any_dtype(u)?;
    Ok(is_not_null(g.expr(u, &child_dtype, depth)?))
}

fn list_contains_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let nullability = target.nullability();
    let lists = g
        .paths
        .iter()
        .filter(|(_, dtype)| is_searchable_list(dtype, nullability))
        .collect_vec();
    let (list_expr, list_dtype) = u.choose_iter(lists)?;
    let DType::List(element, list_nullability) = list_dtype else {
        unreachable!("list_contains rule requires a list path")
    };
    let needle_nullability = match list_nullability {
        // The result is nullable regardless of the needle.
        Nullability::Nullable => Nullability::arbitrary(u)?,
        Nullability::NonNullable => nullability,
    };
    let needle = g.expr(u, &element.with_nullability(needle_nullability), depth)?;
    Ok(list_contains(list_expr.clone(), needle))
}

fn byte_length_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let child_dtype = if u.arbitrary()? {
        DType::Utf8(target.nullability())
    } else {
        DType::Binary(target.nullability())
    };
    Ok(byte_length(g.expr(u, &child_dtype, depth)?))
}

fn pack_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let fields = target
        .as_struct_fields_opt()
        .vortex_expect("pack rule requires a struct target");
    let children = fields
        .names()
        .iter()
        .zip(fields.fields())
        .map(|(name, field_dtype)| Ok((name.clone(), g.expr(u, &field_dtype, depth)?)))
        .collect::<AResult<Vec<_>>>()?;
    Ok(pack(children, target.nullability()))
}

fn select_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let fields = target
        .as_struct_fields_opt()
        .vortex_expect("select rule requires a struct target");
    // Pack a superset of the target fields, then select the target fields back out.
    let mut names = fields.names().iter().cloned().collect_vec();
    let mut dtypes = fields.fields().collect::<Vec<_>>();
    for i in 0..u.int_in_range(0..=2usize)? {
        let name = FieldName::from(format!("__select_extra_{i}"));
        // The extra must not collide with a target field name, or the superset would contain
        // duplicates and the selection would no longer return the target fields.
        if fields.field(&name).is_some() {
            continue;
        }
        names.push(name);
        dtypes.push(random_scalar_dtype(u)?);
    }
    let superset = DType::Struct(
        StructFields::new(names.into(), dtypes),
        target.nullability(),
    );
    Ok(select(fields.names().clone(), g.expr(u, &superset, depth)?))
}

fn merge_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let fields = target
        .as_struct_fields_opt()
        .vortex_expect("merge rule requires a struct target");
    // Split the fields into contiguous chunks, one non-nullable struct child each.
    let mut children = Vec::new();
    let mut remaining = fields.names().iter().zip(fields.fields()).collect_vec();
    while !remaining.is_empty() {
        let take = u.int_in_range(1..=remaining.len())?;
        let chunk = remaining.drain(..take).collect_vec();
        let (names, dtypes): (Vec<_>, Vec<_>) =
            chunk.into_iter().map(|(n, d)| (n.clone(), d)).unzip();
        let chunk_dtype = DType::Struct(
            StructFields::new(names.into(), dtypes),
            Nullability::NonNullable,
        );
        children.push(g.expr(u, &chunk_dtype, depth)?);
    }
    Ok(merge(children))
}

fn get_item_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    // Wrap the target as a field of a synthesized non-nullable struct.
    let field = FieldName::from("__get_item_field");
    let mut names = vec![field.clone()];
    let mut dtypes = vec![target.clone()];
    for i in 0..u.int_in_range(0..=2usize)? {
        names.push(FieldName::from(format!("__get_item_extra_{i}")));
        dtypes.push(random_scalar_dtype(u)?);
    }
    let parent = DType::Struct(
        StructFields::new(names.into(), dtypes),
        Nullability::NonNullable,
    );
    Ok(get_item(field, g.expr(u, &parent, depth)?))
}

fn case_when(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let num_pairs = u.int_in_range(1..=2usize)?;
    let mut pairs = Vec::with_capacity(num_pairs);

    // The result nullability is the union of all THEN/ELSE branches, or forced nullable when
    // there is no ELSE.
    let (branch_nullabilities, else_branch) = match target.nullability() {
        Nullability::NonNullable => (
            vec![Nullability::NonNullable; num_pairs],
            Some(Nullability::NonNullable),
        ),
        Nullability::Nullable => {
            if u.arbitrary()? {
                let mut branches = Vec::with_capacity(num_pairs);
                for _ in 0..num_pairs {
                    branches.push(Nullability::arbitrary(u)?);
                }
                (branches, None)
            } else {
                // Force the first branch nullable so the union is nullable.
                let mut branches = vec![Nullability::Nullable];
                for _ in 1..num_pairs {
                    branches.push(Nullability::arbitrary(u)?);
                }
                (branches, Some(Nullability::arbitrary(u)?))
            }
        }
    };

    for branch_nullability in branch_nullabilities {
        let condition_dtype = DType::Bool(Nullability::arbitrary(u)?);
        let condition = g.expr(u, &condition_dtype, depth)?;
        let then_value = g.expr(u, &target.with_nullability(branch_nullability), depth)?;
        pairs.push((condition, then_value));
    }
    let else_value = else_branch
        .map(|n| g.expr(u, &target.with_nullability(n), depth))
        .transpose()?;

    Ok(nested_case_when(pairs, else_value))
}

fn zip(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    // Same dtype on both branches so the zip supertype is exactly the target.
    let if_true = g.expr(u, target, depth)?;
    let if_false = g.expr(u, target, depth)?;
    let condition_dtype = DType::Bool(Nullability::arbitrary(u)?);
    let condition = g.expr(u, &condition_dtype, depth)?;
    Ok(zip_expr(condition, if_true, if_false))
}

fn fill_null_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let child = g.expr(u, &target.with_nullability(Nullability::Nullable), depth)?;
    // The result takes the nullability of the fill value.
    let fill = g.expr(u, target, depth)?;
    Ok(fill_null(child, fill))
}

fn mask_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let child_dtype = target.with_nullability(Nullability::arbitrary(u)?);
    let child = g.expr(u, &child_dtype, depth)?;
    let mask_child = g.expr(u, &DType::Bool(Nullability::NonNullable), depth)?;
    Ok(mask(child, mask_child))
}

fn arithmetic(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let op = arithmetic_operator(u)?;
    let (lhs_n, rhs_n) = split_nullability2(u, target.nullability())?;
    Ok(Binary.new_expr(
        op,
        [
            g.expr(u, &target.with_nullability(lhs_n), depth)?,
            g.expr(u, &target.with_nullability(rhs_n), depth)?,
        ],
    ))
}

fn cast_(
    g: &Synthesizer,
    u: &mut Unstructured<'_>,
    target: &DType,
    depth: usize,
) -> AResult<Expression> {
    let child_nullability = Nullability::arbitrary(u)?;
    let child_dtype = g.comparable_dtype(u)?.with_nullability(child_nullability);
    Ok(cast(g.expr(u, &child_dtype, depth)?, target.clone()))
}

// ---- Shared helpers ----

/// Collects the root path and all nested struct field paths, applying the `get_item` nullability
/// rule along the way.
fn collect_paths(expr: Expression, dtype: &DType, paths: &mut Vec<(Expression, DType)>) {
    if paths.len() >= MAX_PATHS {
        return;
    }
    paths.push((expr.clone(), dtype.clone()));
    if let Some(fields) = dtype.as_struct_fields_opt() {
        for (name, field_dtype) in fields.names().iter().zip(fields.fields()) {
            // get_item on a nullable struct makes the field nullable.
            let field_dtype = if dtype.is_nullable() {
                field_dtype.as_nullable()
            } else {
                field_dtype
            };
            collect_paths(get_item(name.clone(), expr.clone()), &field_dtype, paths);
        }
    }
}

/// Whether values of this dtype can be ordered/compared without runtime errors.
fn is_comparable_dtype(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
    )
}

/// A random comparable dtype with arbitrary nullability.
fn random_scalar_dtype(u: &mut Unstructured<'_>) -> AResult<DType> {
    let nullability = Nullability::arbitrary(u)?;
    Ok(random_comparable_dtype(u)?.with_nullability(nullability))
}

fn random_comparable_dtype(u: &mut Unstructured<'_>) -> AResult<DType> {
    Ok(match u.int_in_range(0..=4)? {
        0 => DType::Bool(Nullability::NonNullable),
        1 => DType::Primitive(PType::arbitrary(u)?, Nullability::NonNullable),
        2 => DType::Decimal(DecimalDType::arbitrary(u)?, Nullability::NonNullable),
        3 => DType::Utf8(Nullability::NonNullable),
        _ => DType::Binary(Nullability::NonNullable),
    })
}

fn comparison_operator(u: &mut Unstructured<'_>) -> AResult<Operator> {
    Ok(match u.int_in_range(0..=5)? {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Gt,
        3 => Operator::Gte,
        4 => Operator::Lt,
        _ => Operator::Lte,
    })
}

fn arithmetic_operator(u: &mut Unstructured<'_>) -> AResult<Operator> {
    Ok(match u.int_in_range(0..=3)? {
        0 => Operator::Add,
        1 => Operator::Sub,
        2 => Operator::Mul,
        _ => Operator::Div,
    })
}

fn strictness(u: &mut Unstructured<'_>) -> AResult<StrictComparison> {
    Ok(if u.arbitrary()? {
        StrictComparison::Strict
    } else {
        StrictComparison::NonStrict
    })
}

/// A literal LIKE pattern. Patterns are literal-only and escape-free so that pattern compilation
/// cannot fail at runtime.
fn like_pattern(u: &mut Unstructured<'_>, nullability: Nullability) -> AResult<Expression> {
    if nullability == Nullability::Nullable && u.ratio(1, 4)? {
        return Ok(lit(Scalar::null(DType::Utf8(nullability))));
    }
    let pattern: String = u
        .arbitrary::<String>()?
        .chars()
        .filter(|c| *c != '\\')
        .collect();
    Ok(lit(Scalar::utf8(pattern, nullability)))
}

/// Splits a target nullability across two children such that their union equals the target.
fn split_nullability2(
    u: &mut Unstructured<'_>,
    nullability: Nullability,
) -> AResult<(Nullability, Nullability)> {
    Ok(match nullability {
        Nullability::NonNullable => (Nullability::NonNullable, Nullability::NonNullable),
        Nullability::Nullable => match u.int_in_range(0..=2)? {
            0 => (Nullability::Nullable, Nullability::Nullable),
            1 => (Nullability::Nullable, Nullability::NonNullable),
            _ => (Nullability::NonNullable, Nullability::Nullable),
        },
    })
}

/// Splits a target nullability across three children such that their union equals the target.
fn split_nullability3(
    u: &mut Unstructured<'_>,
    nullability: Nullability,
) -> AResult<(Nullability, Nullability, Nullability)> {
    Ok(match nullability {
        Nullability::NonNullable => (
            Nullability::NonNullable,
            Nullability::NonNullable,
            Nullability::NonNullable,
        ),
        Nullability::Nullable => {
            let forced = u.int_in_range(0..=2usize)?;
            let mut split = [Nullability::NonNullable; 3];
            for (i, n) in split.iter_mut().enumerate() {
                if i == forced {
                    *n = Nullability::Nullable;
                } else {
                    *n = Nullability::arbitrary(u)?;
                }
            }
            (split[0], split[1], split[2])
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::expr::eq;
    use crate::expr::select_exclude;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::fns::select::FieldSelection;
    use crate::scalar_fn::fns::select::Select;

    fn ar<T>(result: AResult<T>) -> VortexResult<T> {
        result.map_err(|e| vortex_err!("arbitrary: {e}"))
    }

    fn rich_scope() -> DType {
        DType::Struct(
            StructFields::new(
                vec![
                    FieldName::from("bool_n"),
                    FieldName::from("i32"),
                    FieldName::from("u64"),
                    FieldName::from("str_n"),
                    FieldName::from("bin"),
                    FieldName::from("dec"),
                    FieldName::from("list_i32"),
                    FieldName::from("nested"),
                ]
                .into(),
                vec![
                    DType::Bool(Nullability::Nullable),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::U64, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                    DType::Binary(Nullability::NonNullable),
                    DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable),
                    DType::List(
                        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                        Nullability::NonNullable,
                    ),
                    DType::Struct(
                        StructFields::new(
                            vec![FieldName::from("f64"), FieldName::from("str")].into(),
                            vec![
                                DType::Primitive(PType::F64, Nullability::NonNullable),
                                DType::Utf8(Nullability::Nullable),
                            ],
                        ),
                        Nullability::NonNullable,
                    ),
                ],
            ),
            Nullability::NonNullable,
        )
    }

    fn entropy(rng: &mut StdRng) -> Vec<u8> {
        let mut bytes = vec![0u8; 16 * 1024];
        rng.fill_bytes(&mut bytes);
        bytes
    }

    fn collect_fn_ids(expr: &Expression, ids: &mut BTreeSet<String>) {
        let id = expr.scalar_fn().id().to_string();
        // Distinguish the operator classes that share the binary vtable.
        if let Some(op) = expr.scalar_fn().as_opt::<Binary>() {
            if op.is_arithmetic() {
                ids.insert(format!("{id}#arithmetic"));
            } else if op.is_comparison() {
                ids.insert(format!("{id}#comparison"));
            } else {
                ids.insert(format!("{id}#boolean"));
            }
        }
        ids.insert(id);
        for child in expr.children().iter() {
            collect_fn_ids(child, ids);
        }
    }

    /// Synthesized expressions must type check (verified inside `synthesize`) for arbitrary
    /// scopes and targets, not just struct scopes.
    #[test]
    fn synthesized_expressions_typecheck() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..256 {
            let bytes = entropy(&mut rng);
            let mut u = Unstructured::new(&bytes);
            let scope = ar(DType::arbitrary(&mut u))?;
            let target = ar(DType::arbitrary(&mut u))?;
            let options = SynthesisOptions {
                fallible: ar(bool::arbitrary(&mut u))?,
            };
            // synthesize_expr_with panics if the expression does not type check to the target.
            let expr = ar(synthesize_expr_with(&mut u, &scope, &target, options))?;
            assert_eq!(expr.return_dtype(&scope)?, target);
        }
        Ok(())
    }

    /// The synthesizer must exercise every production (scalar function) in its space.
    #[test]
    fn synthesis_exhausts_expression_space() -> VortexResult<()> {
        let scope = rich_scope();
        let expected: Vec<(&str, ScalarFnId)> = vec![
            ("literal", lit(0i32).scalar_fn().id()),
            ("root", root().scalar_fn().id()),
            ("get_item", get_item("a", root()).scalar_fn().id()),
            ("binary", eq(lit(0i32), lit(0i32)).scalar_fn().id()),
            (
                "cast",
                cast(
                    lit(0i32),
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                )
                .scalar_fn()
                .id(),
            ),
            ("not", not(lit(true)).scalar_fn().id()),
            ("is_null", is_null(lit(0i32)).scalar_fn().id()),
            ("is_not_null", is_not_null(lit(0i32)).scalar_fn().id()),
            (
                "between",
                between(
                    lit(0i32),
                    lit(0i32),
                    lit(1i32),
                    BetweenOptions {
                        lower_strict: StrictComparison::NonStrict,
                        upper_strict: StrictComparison::NonStrict,
                    },
                )
                .scalar_fn()
                .id(),
            ),
            ("like", like(lit("a"), lit("a")).scalar_fn().id()),
            (
                "case_when",
                nested_case_when(vec![(lit(true), lit(0i32))], None)
                    .scalar_fn()
                    .id(),
            ),
            (
                "zip",
                zip_expr(lit(true), lit(0i32), lit(1i32)).scalar_fn().id(),
            ),
            (
                "fill_null",
                fill_null(lit(0i32), lit(1i32)).scalar_fn().id(),
            ),
            ("mask", mask(lit(0i32), lit(true)).scalar_fn().id()),
            ("byte_length", byte_length(lit("a")).scalar_fn().id()),
            (
                "list_contains",
                list_contains(root(), lit(0i32)).scalar_fn().id(),
            ),
            (
                "pack",
                pack(
                    [(FieldName::from("a"), lit(0i32))],
                    Nullability::NonNullable,
                )
                .scalar_fn()
                .id(),
            ),
            ("select", select(["a"], root()).scalar_fn().id()),
            ("merge", merge([root()]).scalar_fn().id()),
        ];

        let mut rng = StdRng::seed_from_u64(0);
        let mut seen = BTreeSet::new();
        let mut samples = Vec::new();
        for i in 0..2048 {
            let bytes = entropy(&mut rng);
            let mut u = Unstructured::new(&bytes);
            let exprs = [
                ar(filter_expr(&mut u, &scope))?,
                ar(projection_expr(&mut u, &scope))?,
            ];
            for expr in exprs.into_iter().flatten() {
                if i < 8 {
                    samples.push(expr.to_string());
                }
                collect_fn_ids(&expr, &mut seen);
            }
        }

        for sample in &samples {
            eprintln!("sample: {sample}");
        }

        let binary_id = eq(lit(0i32), lit(0i32)).scalar_fn().id();
        let mut missing = expected
            .iter()
            .filter(|(_, id)| !seen.contains(&id.to_string()))
            .map(|(name, _)| (*name).to_string())
            .collect_vec();
        missing.extend(
            ["arithmetic", "comparison", "boolean"]
                .iter()
                .filter(|class| !seen.contains(&format!("{binary_id}#{class}")))
                .map(|class| format!("binary#{class}")),
        );
        assert!(
            missing.is_empty(),
            "synthesis did not cover the full expression space, missing: {missing:?}, saw: {seen:?}"
        );
        Ok(())
    }

    /// Registering a new production is a single `with_rule` call.
    #[test]
    fn custom_rule_registration() -> VortexResult<()> {
        fn select_exclude_rule(
            g: &Synthesizer,
            u: &mut Unstructured<'_>,
            target: &DType,
            depth: usize,
        ) -> AResult<Expression> {
            // Pack an extra field alongside the target fields, then exclude it.
            let fields = target
                .as_struct_fields_opt()
                .vortex_expect("select_exclude rule requires a struct target");
            let extra = FieldName::from("__excluded");
            let names = fields.names().iter().cloned().chain([extra.clone()]);
            let dtypes = fields.fields().chain([random_scalar_dtype(u)?]);
            let superset = DType::Struct(
                StructFields::new(names.collect(), dtypes.collect()),
                target.nullability(),
            );
            Ok(select_exclude([extra], g.expr(u, &superset, depth)?))
        }

        fn applies(g: &Synthesizer, target: &DType) -> bool {
            // The excluded field must not collide with a target field name.
            unique_struct_target(g, target)
                && target
                    .as_struct_fields_opt()
                    .is_some_and(|fields| fields.field("__excluded").is_none())
        }

        let scope = rich_scope();
        let synthesizer = Synthesizer::new(&scope, SynthesisOptions::default())
            .with_rule(Rule::new("select_exclude", applies, select_exclude_rule));

        let target = DType::Struct(
            StructFields::new(
                vec![FieldName::from("a"), FieldName::from("b")].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                ],
            ),
            Nullability::NonNullable,
        );

        fn contains_exclude(expr: &Expression) -> bool {
            expr.scalar_fn()
                .as_opt::<Select>()
                .is_some_and(|selection| matches!(selection, FieldSelection::Exclude(_)))
                || expr.children().iter().any(contains_exclude)
        }

        let mut rng = StdRng::seed_from_u64(0);
        let mut excluded = 0;
        for _ in 0..512 {
            let bytes = entropy(&mut rng);
            let mut u = Unstructured::new(&bytes);
            // synthesize panics if the custom rule produces an ill-typed expression.
            let expr = ar(synthesizer.synthesize(&mut u, &target))?;
            if contains_exclude(&expr) {
                excluded += 1;
            }
        }
        assert!(excluded > 0, "custom rule was never chosen");
        Ok(())
    }
}
