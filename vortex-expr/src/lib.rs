// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! User-authored expressions and their opt-in binding rules.
//!
//! An [`Expression`] stores a function ID, opaque options, and child expressions. It never stores
//! an execution [`ScalarFnRef`](vortex_array::scalar_fn::ScalarFnRef). The [`ExpressionRegistry`]
//! decides which authored functions exist and how they bind against a schema. Engine integrations
//! can construct [`BoundExpression`]s directly and need not use this registry.

#![deny(missing_docs)]

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::scalar_fn::fns::case_when::CaseWhen;
use vortex_array::scalar_fn::fns::case_when::CaseWhenOptions;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::ext_storage::ExtStorage;
use vortex_array::scalar_fn::fns::fill_null::FillNull;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_array::scalar_fn::fns::list_sum::ListSum;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::merge::DuplicateHandling;
use vortex_array::scalar_fn::fns::merge::Merge;
use vortex_array::scalar_fn::fns::not::Not;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::pack::PackOptions;
use vortex_array::scalar_fn::fns::select::FieldSelection;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantGetOptions;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::zip::Zip;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

mod proto;

/// A schema-independent expression node in the authored language.
#[derive(Clone)]
pub enum Expression {
    /// The input scope.
    Root,
    /// A named function call with opaque options.
    Call {
        /// Authored function ID, resolved only at binding time.
        id: Arc<str>,
        /// Function-specific options, which cannot contain a bound scalar function.
        options: Arc<dyn Any + Send + Sync>,
        /// Authored arguments in order.
        children: Arc<Vec<Expression>>,
    },
}

impl fmt::Debug for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => f.write_str("Root"),
            Self::Call {
                id,
                options,
                children,
            } => f
                .debug_struct("Call")
                .field("id", id)
                .field("options", &DebugOptions { id, options })
                .field("children", children)
                .finish(),
        }
    }
}

struct DebugOptions<'a> {
    id: &'a str,
    options: &'a Arc<dyn Any + Send + Sync>,
}

fn debug_option<T: Any + fmt::Debug>(options: &(dyn Any + Send + Sync)) -> Option<&dyn fmt::Debug> {
    options
        .downcast_ref::<T>()
        .map(|value| value as &dyn fmt::Debug)
}

impl fmt::Debug for DebugOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self.id {
            "column" => debug_option::<FieldName>(self.options.as_ref()),
            "literal" => debug_option::<Scalar>(self.options.as_ref()),
            "binary" => debug_option::<Operator>(self.options.as_ref()),
            "cast" => debug_option::<DType>(self.options.as_ref()),
            "select" => debug_option::<FieldSelection>(self.options.as_ref()),
            "pack" => debug_option::<PackOptions>(self.options.as_ref()),
            "like" => debug_option::<LikeOptions>(self.options.as_ref()),
            "between" => debug_option::<BetweenOptions>(self.options.as_ref()),
            "merge" => debug_option::<DuplicateHandling>(self.options.as_ref()),
            "list_sum" => debug_option::<NumericalAggregateOpts>(self.options.as_ref()),
            "case_when" => debug_option::<CaseWhenOptions>(self.options.as_ref()),
            "variant_get" => debug_option::<VariantGetOptions>(self.options.as_ref()),
            _ => Some(&"<opaque>" as &dyn fmt::Debug),
        }
        .unwrap_or(&"<invalid options>" as &dyn fmt::Debug);
        value.fmt(f)
    }
}

impl Expression {
    /// Construct a named authored call. The function is checked when the tree is bound.
    pub fn call(
        id: impl Into<Arc<str>>,
        options: impl Any + Send + Sync,
        children: impl IntoIterator<Item = Expression>,
    ) -> Self {
        Self::Call {
            id: id.into(),
            options: Arc::new(options),
            children: Arc::new(Vec::from_iter(children)),
        }
    }

    /// Bind using the built-in authored function registry.
    pub fn bind(&self, dtype: &DType) -> VortexResult<BoundExpression> {
        self.bind_with(dtype, &ExpressionRegistry::with_builtins()?)
    }

    /// Bind with an explicit authored function registry.
    pub fn bind_with(
        &self,
        dtype: &DType,
        registry: &ExpressionRegistry,
    ) -> VortexResult<BoundExpression> {
        match self {
            Self::Root => Ok(BoundExpression::new_root(dtype.clone())),
            Self::Call {
                id,
                options,
                children,
            } => {
                let bound_children = children
                    .iter()
                    .map(|child| child.bind_with(dtype, registry))
                    .collect::<VortexResult<Vec<_>>>()?;
                let rule = registry
                    .get(id)
                    .ok_or_else(|| vortex_err!("authored function {id} is not registered"))?;
                rule.bind(options.as_ref(), &bound_children)
            }
        }
    }
}

impl Drop for Expression {
    fn drop(&mut self) {
        // Drain uniquely owned children iteratively so deep authored trees do not overflow on drop.
        let Self::Call { children, .. } = self else {
            return;
        };
        let Some(children) = Arc::get_mut(children) else {
            return;
        };
        let mut to_drop = std::mem::take(children);
        while let Some(mut child) = to_drop.pop() {
            if let Self::Call { children, .. } = &mut child
                && let Some(grandchildren) = Arc::get_mut(children)
            {
                to_drop.append(grandchildren);
            }
        }
    }
}

/// An opt-in binding rule for one authored function ID.
///
/// A rule may inspect argument dtypes, select an execution overload, insert casts, and construct
/// the final bound scalar function. Registering a `ScalarFnVTable` alone does not register a rule.
pub trait ExpressionFn: Send + Sync {
    /// ID referenced by authored calls.
    fn id(&self) -> &'static str;

    /// Type-check and lower a call whose children have already been bound.
    fn bind(
        &self,
        options: &(dyn Any + Send + Sync),
        children: &[BoundExpression],
    ) -> VortexResult<BoundExpression>;
}

/// Binding rules available to a user-authored expression language.
#[derive(Default)]
pub struct ExpressionRegistry {
    functions: BTreeMap<&'static str, Arc<dyn ExpressionFn>>,
}

impl ExpressionRegistry {
    /// An empty registry for a caller-defined language.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the built-in authored functions.
    pub fn with_builtins() -> VortexResult<Self> {
        let mut registry = Self::new();
        for function in [
            Builtin::Column,
            Builtin::Literal,
            Builtin::Binary,
            Builtin::Cast,
            Builtin::Not,
            Builtin::IsNull,
            Builtin::IsNotNull,
            Builtin::Select,
            Builtin::Pack,
            Builtin::Between,
            Builtin::Like,
            Builtin::FillNull,
            Builtin::ByteLength,
            Builtin::Merge,
            Builtin::ListContains,
            Builtin::ListLength,
            Builtin::ListSum,
            Builtin::CaseWhen,
            Builtin::Zip,
            Builtin::Mask,
            Builtin::ExtStorage,
            Builtin::VariantGet,
        ] {
            registry.register(Arc::new(function))?;
        }
        Ok(registry)
    }

    /// Register a binding rule. Duplicate authored IDs are rejected.
    pub fn register(&mut self, function: Arc<dyn ExpressionFn>) -> VortexResult<()> {
        let id = function.id();
        vortex_ensure!(
            !self.functions.contains_key(id),
            "authored function {id} is already registered"
        );
        self.functions.insert(id, function);
        Ok(())
    }

    /// Look up a registered binding rule by authored ID.
    pub fn get(&self, id: &str) -> Option<&dyn ExpressionFn> {
        self.functions.get(id).map(|function| function.as_ref())
    }
}

#[derive(Clone, Copy)]
enum Builtin {
    Column,
    Literal,
    Binary,
    Cast,
    Not,
    IsNull,
    IsNotNull,
    Select,
    Pack,
    Between,
    Like,
    FillNull,
    ByteLength,
    Merge,
    ListContains,
    ListLength,
    ListSum,
    CaseWhen,
    Zip,
    Mask,
    ExtStorage,
    VariantGet,
}

fn options<'a, T: Any>(id: &str, value: &'a (dyn Any + Send + Sync)) -> VortexResult<&'a T> {
    value
        .downcast_ref::<T>()
        .ok_or_else(|| vortex_err!("invalid options for authored function {id}"))
}

fn coerce_literal(child: &BoundExpression, target: &DType) -> VortexResult<BoundExpression> {
    if child.dtype().eq_ignore_nullability(target) {
        return Ok(child.clone());
    }

    let Some(literal) = child.as_opt::<Literal>() else {
        return Ok(child.clone());
    };
    let target = if literal.is_null() {
        target.as_nullable()
    } else {
        target.clone()
    };
    Literal.try_new_bound_expr(literal.cast(&target)?, [])
}

impl ExpressionFn for Builtin {
    fn id(&self) -> &'static str {
        match self {
            Self::Column => "column",
            Self::Literal => "literal",
            Self::Binary => "binary",
            Self::Cast => "cast",
            Self::Not => "not",
            Self::IsNull => "is_null",
            Self::IsNotNull => "is_not_null",
            Self::Select => "select",
            Self::Pack => "pack",
            Self::Between => "between",
            Self::Like => "like",
            Self::FillNull => "fill_null",
            Self::ByteLength => "byte_length",
            Self::Merge => "merge",
            Self::ListContains => "list_contains",
            Self::ListLength => "list_length",
            Self::ListSum => "list_sum",
            Self::CaseWhen => "case_when",
            Self::Zip => "zip",
            Self::Mask => "mask",
            Self::ExtStorage => "ext_storage",
            Self::VariantGet => "variant_get",
        }
    }

    fn bind(
        &self,
        value: &(dyn Any + Send + Sync),
        children: &[BoundExpression],
    ) -> VortexResult<BoundExpression> {
        let id = self.id();
        match self {
            Self::Column => {
                let name = options::<FieldName>(id, value)?;
                vortex_ensure!(children.len() == 1, "column expects one child");
                GetItem.try_new_bound_expr(name.clone(), children.to_vec())
            }
            Self::Literal => {
                let scalar = options::<Scalar>(id, value)?;
                vortex_ensure!(children.is_empty(), "literal expects no children");
                Literal.try_new_bound_expr(scalar.clone(), [])
            }
            Self::Binary => {
                let operator = options::<Operator>(id, value)?;
                vortex_ensure!(children.len() == 2, "binary expects two children");
                let mut lhs = children[0].clone();
                let mut rhs = children[1].clone();
                if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
                    match (
                        lhs.as_opt::<Literal>().cloned(),
                        rhs.as_opt::<Literal>().cloned(),
                    ) {
                        (None, Some(literal)) => {
                            let target = if literal.is_null() {
                                lhs.dtype().as_nullable()
                            } else {
                                lhs.dtype().clone()
                            };
                            rhs = Literal.try_new_bound_expr(literal.cast(&target)?, [])?;
                        }
                        (Some(literal), None) => {
                            let target = if literal.is_null() {
                                rhs.dtype().as_nullable()
                            } else {
                                rhs.dtype().clone()
                            };
                            lhs = Literal.try_new_bound_expr(literal.cast(&target)?, [])?;
                        }
                        _ => {}
                    }
                }
                Binary.try_new_bound_expr(*operator, [lhs, rhs])
            }
            Self::Cast => {
                let target = options::<DType>(id, value)?;
                Cast.try_new_bound_expr(target.clone(), children.to_vec())
            }
            Self::Not => Not.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::IsNull => IsNull.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::IsNotNull => IsNotNull.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::Select => Select.try_new_bound_expr(
                options::<FieldSelection>(id, value)?.clone(),
                children.to_vec(),
            ),
            Self::Pack => Pack.try_new_bound_expr(
                options::<PackOptions>(id, value)?.clone(),
                children.to_vec(),
            ),
            Self::Between => {
                vortex_ensure!(children.len() == 3, "between expects three children");
                let dtype = children[0].dtype();
                Between.try_new_bound_expr(
                    options::<BetweenOptions>(id, value)?.clone(),
                    [
                        children[0].clone(),
                        coerce_literal(&children[1], dtype)?,
                        coerce_literal(&children[2], dtype)?,
                    ],
                )
            }
            Self::Like => {
                Like.try_new_bound_expr(*options::<LikeOptions>(id, value)?, children.to_vec())
            }
            Self::FillNull => {
                vortex_ensure!(children.len() == 2, "fill_null expects two children");
                FillNull.try_new_bound_expr(
                    EmptyOptions,
                    [
                        children[0].clone(),
                        coerce_literal(&children[1], children[0].dtype())?,
                    ],
                )
            }
            Self::ByteLength => ByteLength.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::Merge => Merge
                .try_new_bound_expr(*options::<DuplicateHandling>(id, value)?, children.to_vec()),
            Self::ListContains => {
                vortex_ensure!(children.len() == 2, "list_contains expects two children");
                let element_dtype = match children[0].dtype() {
                    DType::List(element, _) | DType::FixedSizeList(element, ..) => element,
                    _ => return ListContains.try_new_bound_expr(EmptyOptions, children.to_vec()),
                };
                ListContains.try_new_bound_expr(
                    EmptyOptions,
                    [
                        children[0].clone(),
                        coerce_literal(&children[1], element_dtype)?,
                    ],
                )
            }
            Self::ListLength => ListLength.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::ListSum => ListSum.try_new_bound_expr(
                *options::<NumericalAggregateOpts>(id, value)?,
                children.to_vec(),
            ),
            Self::CaseWhen => {
                let case_options = options::<CaseWhenOptions>(id, value)?;
                vortex_ensure!(
                    case_options.num_when_then_pairs > 0,
                    "case_when needs a WHEN/THEN pair"
                );
                vortex_ensure!(
                    children.len() == case_options.num_children(),
                    "case_when child count mismatch"
                );
                let result_dtype = children
                    .iter()
                    .enumerate()
                    .find(|(index, child)| {
                        (index % 2 == 1 || (case_options.has_else && *index == children.len() - 1))
                            && child.as_opt::<Literal>().is_none()
                    })
                    .map(|(_, child)| child.dtype())
                    .unwrap_or_else(|| children[1].dtype());
                let branches = children
                    .iter()
                    .enumerate()
                    .map(|(index, child)| {
                        if index % 2 == 1 || (case_options.has_else && index == children.len() - 1)
                        {
                            coerce_literal(child, result_dtype)
                        } else {
                            Ok(child.clone())
                        }
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                CaseWhen.try_new_bound_expr(*case_options, branches)
            }
            Self::Zip => {
                vortex_ensure!(children.len() == 3, "zip expects three children");
                let target = if children[0].as_opt::<Literal>().is_some() {
                    children[1].dtype()
                } else {
                    children[0].dtype()
                };
                Zip.try_new_bound_expr(
                    EmptyOptions,
                    [
                        coerce_literal(&children[0], target)?,
                        coerce_literal(&children[1], target)?,
                        children[2].clone(),
                    ],
                )
            }
            Self::Mask => Mask.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::ExtStorage => ExtStorage.try_new_bound_expr(EmptyOptions, children.to_vec()),
            Self::VariantGet => VariantGet.try_new_bound_expr(
                options::<VariantGetOptions>(id, value)?.clone(),
                children.to_vec(),
            ),
        }
    }
}

/// The input scope.
pub fn root() -> Expression {
    Expression::Root
}

/// Read a field from the input scope.
pub fn col(name: impl Into<FieldName>) -> Expression {
    Expression::call("column", name.into(), [root()])
}

/// Read a field from a struct expression.
pub fn get_item(name: impl Into<FieldName>, child: Expression) -> Expression {
    Expression::call("column", name.into(), [child])
}

/// A typed scalar literal.
pub fn lit(value: impl Into<Scalar>) -> Expression {
    Expression::call("literal", value.into(), [])
}

/// An authored binary operation. Its binding rule may coerce literal operands.
pub fn binary(operator: Operator, lhs: Expression, rhs: Expression) -> Expression {
    Expression::call("binary", operator, [lhs, rhs])
}

/// An explicit cast to a Vortex dtype.
pub fn cast(child: Expression, target: DType) -> Expression {
    Expression::call("cast", target, [child])
}

/// Boolean negation.
pub fn not(child: Expression) -> Expression {
    Expression::call("not", (), [child])
}

/// Test for a null value.
pub fn is_null(child: Expression) -> Expression {
    Expression::call("is_null", (), [child])
}

/// Test for a non-null value.
pub fn is_not_null(child: Expression) -> Expression {
    Expression::call("is_not_null", (), [child])
}

/// Select fields from a struct expression.
pub fn select(names: impl Into<FieldNames>, child: Expression) -> Expression {
    Expression::call("select", FieldSelection::Include(names.into()), [child])
}

/// Exclude fields from a struct expression.
pub fn select_exclude(names: impl Into<FieldNames>, child: Expression) -> Expression {
    Expression::call("select", FieldSelection::Exclude(names.into()), [child])
}

/// Pack named values into a struct.
pub fn pack(
    fields: impl IntoIterator<Item = (FieldName, Expression)>,
    nullability: Nullability,
) -> Expression {
    let (names, children): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
    Expression::call(
        "pack",
        PackOptions {
            names: names.into(),
            nullability,
        },
        children,
    )
}

/// Combine expressions with a binary operator.
pub fn and_collect(values: impl IntoIterator<Item = Expression>) -> Option<Expression> {
    collect_binary(values, Operator::And)
}

/// Combine expressions with OR.
pub fn or_collect(values: impl IntoIterator<Item = Expression>) -> Option<Expression> {
    collect_binary(values, Operator::Or)
}

fn collect_binary(
    values: impl IntoIterator<Item = Expression>,
    operator: Operator,
) -> Option<Expression> {
    let mut level: Vec<_> = values.into_iter().collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut pairs = level.into_iter();
        while let Some(lhs) = pairs.next() {
            next.push(match pairs.next() {
                Some(rhs) => binary(operator, lhs, rhs),
                None => lhs,
            });
        }
        level = next;
    }
    level.pop()
}

macro_rules! binary_constructor {
    ($name:ident, $operator:ident) => {
        #[doc = concat!("Create an authored ", stringify!($operator), " operation.")]
        pub fn $name(lhs: Expression, rhs: Expression) -> Expression {
            binary(Operator::$operator, lhs, rhs)
        }
    };
}

binary_constructor!(eq, Eq);
binary_constructor!(not_eq, NotEq);
binary_constructor!(gt, Gt);
binary_constructor!(gt_eq, Gte);
binary_constructor!(lt, Lt);
binary_constructor!(lt_eq, Lte);
binary_constructor!(checked_add, Add);
binary_constructor!(and, And);
binary_constructor!(or, Or);

/// An authored BETWEEN call.
pub fn between(
    value: Expression,
    lower: Expression,
    upper: Expression,
    options: BetweenOptions,
) -> Expression {
    Expression::call("between", options, [value, lower, upper])
}

/// An authored case-sensitive LIKE call.
pub fn like(value: Expression, pattern: Expression) -> Expression {
    Expression::call("like", LikeOptions::default(), [value, pattern])
}

/// An authored case-insensitive LIKE call.
pub fn ilike(value: Expression, pattern: Expression) -> Expression {
    Expression::call(
        "like",
        LikeOptions {
            case_insensitive: true,
            negated: false,
        },
        [value, pattern],
    )
}

/// An authored negated, case-sensitive LIKE call.
pub fn not_like(value: Expression, pattern: Expression) -> Expression {
    Expression::call(
        "like",
        LikeOptions {
            case_insensitive: false,
            negated: true,
        },
        [value, pattern],
    )
}

/// An authored negated, case-insensitive LIKE call.
pub fn not_ilike(value: Expression, pattern: Expression) -> Expression {
    Expression::call(
        "like",
        LikeOptions {
            case_insensitive: true,
            negated: true,
        },
        [value, pattern],
    )
}

/// Replace null values with another expression.
pub fn fill_null(value: Expression, replacement: Expression) -> Expression {
    Expression::call("fill_null", (), [value, replacement])
}

/// Count the bytes in a binary or UTF-8 value.
pub fn byte_length(value: Expression) -> Expression {
    Expression::call("byte_length", (), [value])
}

/// Merge struct-valued expressions using the given duplicate-field policy.
pub fn merge_opts(
    values: impl IntoIterator<Item = Expression>,
    handling: DuplicateHandling,
) -> Expression {
    Expression::call("merge", handling, values)
}

/// Test whether a list contains a value.
pub fn list_contains(list: Expression, value: Expression) -> Expression {
    Expression::call("list_contains", (), [list, value])
}

/// Count elements in a list.
pub fn list_length(list: Expression) -> Expression {
    Expression::call("list_length", (), [list])
}

/// Sum numeric elements in a list using the given aggregate options.
pub fn list_sum_opts(list: Expression, options: NumericalAggregateOpts) -> Expression {
    Expression::call("list_sum", options, [list])
}

/// Build a searched CASE expression from condition/result pairs and an optional ELSE value.
pub fn nested_case_when(
    pairs: Vec<(Expression, Expression)>,
    else_value: Option<Expression>,
) -> VortexResult<Expression> {
    let pair_count = u32::try_from(pairs.len())
        .map_err(|_| vortex_err!("too many CASE WHEN pairs: {}", pairs.len()))?;
    let has_else = else_value.is_some();
    let mut children = Vec::with_capacity(pairs.len() * 2 + usize::from(has_else));
    for (condition, value) in pairs {
        children.extend([condition, value]);
    }
    children.extend(else_value);
    Ok(Expression::call(
        "case_when",
        CaseWhenOptions {
            num_when_then_pairs: pair_count,
            has_else,
        },
        children,
    ))
}

/// Choose one of two values according to a boolean mask.
pub fn zip_expr(mask: Expression, if_true: Expression, if_false: Expression) -> Expression {
    Expression::call("zip", (), [if_true, if_false, mask])
}

/// Set values to null where a boolean mask is false.
pub fn mask(value: Expression, mask: Expression) -> Expression {
    Expression::call("mask", (), [value, mask])
}

/// Read the storage value of an extension expression.
pub fn ext_storage(value: Expression) -> Expression {
    Expression::call("ext_storage", (), [value])
}

/// Read a path from a variant value, optionally selecting the output dtype.
pub fn variant_get(
    value: Expression,
    path: impl Into<VariantPath>,
    dtype: Option<DType>,
) -> Expression {
    Expression::call(
        "variant_get",
        VariantGetOptions::new(path.into(), dtype),
        [value],
    )
}

#[cfg(test)]
mod tests {
    use std::thread;

    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::scalar_fn::fns::between::StrictComparison;

    use super::*;

    #[test]
    fn authored_literal_coerces_to_column_type() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "value",
                DType::Primitive(PType::I64, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let expr = binary(Operator::Eq, col("value"), lit(42i32));
        let bound = expr.bind(&dtype)?;
        assert_eq!(
            bound.child(1).dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn authored_between_coerces_literal_bounds() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "value",
                DType::Primitive(PType::I64, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        let bound = between(col("value"), lit(1i32), lit(10i32), options).bind(&dtype)?;
        assert_eq!(bound.child(1).dtype(), bound.child(0).dtype());
        assert_eq!(bound.child(2).dtype(), bound.child(0).dtype());
        Ok(())
    }

    #[test]
    fn exposed_like_and_merge_have_binding_rules() -> VortexResult<()> {
        let string = DType::Utf8(Nullability::NonNullable);
        let struct_dtype = DType::Struct(
            StructFields::from_iter([("name", string)]),
            Nullability::NonNullable,
        );
        let predicate = like(col("name"), lit("a%"));
        assert_eq!(
            predicate.bind(&struct_dtype)?.dtype(),
            &DType::Bool(Nullability::NonNullable)
        );

        let merged =
            merge_opts([root(), root()], DuplicateHandling::RightMost).bind(&struct_dtype)?;
        assert_eq!(merged.dtype(), &struct_dtype);
        Ok(())
    }

    #[test]
    fn scalar_registration_does_not_expose_authored_function() {
        let expr = Expression::call("vortex.unregistered", (), []);
        assert!(expr.bind(&DType::Null).is_err());
    }

    #[test]
    fn two_literals_with_different_types_need_an_explicit_cast() {
        let lhs = binary(Operator::Eq, lit(1u8), lit(2i32));
        let rhs = binary(Operator::Eq, lit(2i32), lit(1u8));
        assert!(lhs.bind(&DType::Null).is_err());
        assert!(rhs.bind(&DType::Null).is_err());
    }

    #[test]
    fn authored_tree_round_trips_without_an_execution_expression() -> VortexResult<()> {
        let expression = binary(Operator::Eq, col("value"), lit(42i32));
        let session = vortex_array::array_session();
        let encoded = expression.serialize_proto()?;
        let decoded = Expression::from_proto(&encoded, &session)?;
        assert_eq!(format!("{expression:?}"), format!("{decoded:?}"));
        Ok(())
    }

    #[test]
    fn serializable_authored_functions_keep_their_execution_wire_ids() -> VortexResult<()> {
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        let expressions = [
            ("vortex.between", between(col("x"), lit(1), lit(2), options)),
            ("vortex.like", like(col("name"), lit("A%"))),
            ("vortex.fill_null", fill_null(col("x"), lit(0))),
            ("vortex.byte_length", byte_length(col("name"))),
            (
                "vortex.merge",
                merge_opts([col("left"), col("right")], DuplicateHandling::RightMost),
            ),
            ("vortex.list.contains", list_contains(col("items"), lit(1))),
            ("vortex.list.length", list_length(col("items"))),
            (
                "vortex.list.sum",
                list_sum_opts(col("items"), NumericalAggregateOpts::default()),
            ),
            (
                "vortex.zip",
                zip_expr(col("mask"), col("left"), col("right")),
            ),
            ("vortex.mask", mask(col("x"), col("mask"))),
            ("vortex.ext.storage", ext_storage(col("x"))),
            (
                "vortex.variant_get",
                variant_get(col("payload"), VariantPath::field("user"), None),
            ),
        ];
        let session = vortex_array::array_session();
        for (wire_id, expression) in expressions {
            let encoded = expression.serialize_proto()?;
            assert_eq!(encoded.id, wire_id);
            let decoded = Expression::from_proto(&encoded, &session)?;
            assert_eq!(format!("{expression:?}"), format!("{decoded:?}"));
        }
        Ok(())
    }

    #[test]
    fn deep_authored_expression_drops_on_a_small_stack() -> VortexResult<()> {
        const DEPTH: usize = 100_000;
        const STACK_SIZE: usize = 256 * 1024;
        let dropper = thread::Builder::new().stack_size(STACK_SIZE).spawn(|| {
            let mut expression = lit(true);
            for _ in 0..DEPTH {
                expression = not(expression);
            }
            drop(expression);
        })?;
        assert!(dropper.join().is_ok());
        Ok(())
    }
}
