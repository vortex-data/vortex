// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lambdas, the function arguments of higher-order scalar functions such as `list_transform`.
//!
//! A [`Lambda`] is a list of parameter names and a body that reads them with
//! [`var`](crate::expr::var). A function that takes a lambda keeps it in its options. When the
//! function's node is bound, it binds the lambda to its parameters' dtypes, once, and keeps the
//! [`BoundLambda`] instead.
//!
//! Binding turns the lambda into an ordinary expression over a struct of its inputs, one field per
//! parameter: `@x` becomes `$.x`. To run it, the function puts the parameters' arrays in that
//! struct, such as one row per list element, and applies the bound body to it.
//!
//! A body may read values from outside the lambda, such as a column. Before a function keeps a
//! lambda, [`Lambda::extract_captures`] turns each of those values into a new parameter and returns
//! the expressions that compute them, which the function takes as extra children. A lambda in a
//! function's options therefore reads nothing but its parameters, and the function's children are
//! everything it reads.

#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::get_item;
use crate::expr::root;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::expr::var;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::internal::row_count::RowCount;
use crate::validity::Validity;

/// Parameter names and a body: the function argument of a higher-order scalar function, as
/// written.
///
/// The body is an ordinary [`Expression`] that reads its parameters with
/// [`var`](crate::expr::var). It may also read values from outside the lambda, which
/// [`Lambda::extract_captures`] turns into parameters.
///
/// ```rust
/// # use vortex_array::expr::{Lambda, checked_add, lit, var};
/// let lambda = Lambda::new(["x"], checked_add(var("x"), lit(1_i32))).unwrap();
/// assert_eq!(lambda.to_string(), "(x) -> (@x + 1i32)");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    params: FieldNames,
    body: Expression,
}

impl Lambda {
    /// Create a lambda with these parameters and body.
    ///
    /// The body may read from outside the lambda: `$` or a column, another lambda's parameter, or
    /// the scan's `row_idx()` or `row_count()`. See [`Lambda::extract_captures`].
    ///
    /// Returns an error when a parameter repeats.
    pub fn new(
        params: impl IntoIterator<Item = impl Into<FieldName>>,
        body: Expression,
    ) -> VortexResult<Self> {
        let params: FieldNames = params.into_iter().map(Into::into).collect();
        if let Some(param) = params.iter().duplicates().next() {
            vortex_bail!("lambda parameter {param} is declared more than once");
        }
        Ok(Self { params, body })
    }

    /// Turn every value the body reads from outside the lambda into a new parameter.
    ///
    /// Returns the lambda with those parameters after its own, and the expressions that compute
    /// their values, in the same order. A function that takes a lambda keeps the returned lambda
    /// and takes the expressions as extra children, so that a rewrite of the expression, such as
    /// `replace` or field pruning, sees them.
    ///
    /// A value is read as a whole: `$.s.a` becomes one parameter, `a`, not a read of `$`. A value
    /// read twice becomes one parameter. A new parameter is named after the field or variable it
    /// reads, with a number added when that name is taken.
    ///
    /// ```rust
    /// # use vortex_array::expr::{Lambda, checked_add, col, var};
    /// let lambda = Lambda::new(["x"], checked_add(var("x"), col("a"))).unwrap();
    /// let (lambda, captures) = lambda.extract_captures();
    /// assert_eq!(lambda.to_string(), "(x, a) -> (@x + @a)");
    /// assert_eq!(captures, [col("a")]);
    /// ```
    pub fn extract_captures(self) -> (Lambda, Vec<Expression>) {
        let written = self.params;
        let mut params = written.iter().cloned().collect_vec();
        let mut captures: Vec<Expression> = Vec::new();
        let body = self
            .body
            .transform_down(|node| {
                if !is_capture(&node, written.as_ref()) {
                    return Ok(Transformed::no(node));
                }
                let name = match captures.iter().position(|capture| capture == &node) {
                    Some(index) => params[written.len() + index].clone(),
                    None => {
                        let name = unused_name(capture_name(&node), &params);
                        params.push(name.clone());
                        captures.push(node);
                        name
                    }
                };
                Ok(Transformed {
                    value: var(name),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            })
            .vortex_expect("expression traversal cannot fail")
            .into_inner();
        let lambda = Lambda {
            params: params.into(),
            body,
        };
        (lambda, captures)
    }

    /// The parameter names, in order.
    pub fn params(&self) -> &[FieldName] {
        self.params.as_ref()
    }

    /// The body, as written.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// The dtype of the body, given the parameters' dtypes, without binding it.
    pub fn return_dtype(&self, param_dtypes: &[DType]) -> VortexResult<DType> {
        let inputs = inputs_dtype(self.param_fields(param_dtypes)?);
        self.body_over_inputs()?.return_dtype(&inputs)
    }

    /// Bind the body to these parameter dtypes: `@x` becomes `$.x`, a field of the struct of the
    /// lambda's inputs.
    ///
    /// A function that takes a lambda calls this from its `bind_options`, so the body is bound
    /// once, with the node that holds it, and never while the function runs.
    ///
    /// Returns an error when the body reads from outside the lambda: the function must pass those
    /// values in as parameters, with [`Lambda::extract_captures`].
    pub fn bind(&self, param_dtypes: &[DType]) -> VortexResult<BoundLambda> {
        let params = self.param_fields(param_dtypes)?;
        let body = self
            .body_over_inputs()?
            .bind(&inputs_dtype(params.clone()))?;
        Ok(BoundLambda { params, body })
    }

    /// Whether nothing in the body can fail, so that the body may run over values whose results
    /// are never read, such as dictionary values that no row uses.
    pub fn is_infallible(&self) -> bool {
        let mut infallible = true;
        pre_order_visit_down(&self.body, |node: &Expression| {
            infallible &= node
                .as_scalar()
                .is_none_or(|scalar_fn| scalar_fn.signature().is_infallible());
            Ok(if infallible {
                TraversalOrder::Continue
            } else {
                TraversalOrder::Stop
            })
        })
        .vortex_expect("expression traversal cannot fail");
        infallible
    }

    fn param_fields(&self, param_dtypes: &[DType]) -> VortexResult<StructFields> {
        vortex_ensure!(
            param_dtypes.len() == self.params.len(),
            "lambda {self} needs one dtype per parameter, got {}",
            param_dtypes.len()
        );
        Ok(StructFields::new(
            self.params.clone(),
            param_dtypes.to_vec(),
        ))
    }

    /// The body over the struct of the lambda's inputs, where `@x` reads the field `x`.
    ///
    /// Returns an error when the body reads from outside the lambda, which the struct can't hold.
    /// A lambda in the body is a function's options, which this leaves alone: its variables are its
    /// own parameters.
    fn body_over_inputs(&self) -> VortexResult<Expression> {
        Ok(self
            .body
            .clone()
            .transform_down(|node| {
                vortex_ensure!(
                    !reads_outside_lambda(&node, self.params.as_ref()),
                    "lambda ({}) can only read its parameters, not {node}; \
                     Lambda::extract_captures turns what it reads from outside into parameters",
                    self.params.iter().format(", ")
                );
                Ok(match &node {
                    Expression::Variable(name) => Transformed {
                        value: get_item(name.clone(), root()),
                        changed: true,
                        order: TraversalOrder::Skip,
                    },
                    _ => Transformed::no(node),
                })
            })?
            .into_inner())
    }
}

/// The struct of a lambda's inputs, one field per parameter.
fn inputs_dtype(params: StructFields) -> DType {
    DType::Struct(params, Nullability::NonNullable)
}

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.params.iter().format(", "), self.body)
    }
}

/// A [`Lambda`] bound to its parameters' dtypes, ready to run.
///
/// The body is a [`BoundExpression`] over the struct of the lambda's inputs, one field per
/// parameter: `@x` was bound as `$.x`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    params: StructFields,
    body: BoundExpression,
}

impl BoundLambda {
    /// The parameters' names and dtypes, in order.
    pub fn params(&self) -> &StructFields {
        &self.params
    }

    /// The bound body, over the struct of the lambda's inputs.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// The dtype of the body.
    pub fn dtype(&self) -> &DType {
        self.body.dtype()
    }

    /// Whether the lambda is bound to these parameter dtypes.
    pub fn is_bound_to(&self, param_dtypes: &[DType]) -> bool {
        self.params.fields().eq(param_dtypes.iter().cloned())
    }

    /// The same lambda bound to other parameter dtypes, such as after a rewrite of the function's
    /// node made a value it passes in non-nullable.
    ///
    /// Every node of the body is rebuilt over the new struct of inputs, so the body's dtype follows
    /// the parameters'. The result is the same as binding the written lambda to these dtypes.
    pub fn rebind(&self, param_dtypes: &[DType]) -> VortexResult<BoundLambda> {
        vortex_ensure_eq!(param_dtypes.len(), self.params.nfields(), "lambda {self}");
        let params = StructFields::new(self.params.names().clone(), param_dtypes.to_vec());
        let root = BoundExpression::new_root(inputs_dtype(params.clone()));
        let body = with_root(&self.body, &root)?;
        Ok(BoundLambda { params, body })
    }

    /// Whether nothing in the body can fail. See [`Lambda::is_infallible`].
    pub fn is_infallible(&self) -> bool {
        let mut infallible = true;
        pre_order_visit_down(&self.body, |node: &BoundExpression| {
            infallible &= node
                .as_scalar()
                .is_none_or(|scalar_fn| scalar_fn.signature().is_infallible());
            Ok(if infallible {
                TraversalOrder::Continue
            } else {
                TraversalOrder::Stop
            })
        })
        .vortex_expect("bound expression traversal cannot fail");
        infallible
    }

    /// Evaluate the body over `len` rows of `args`, one array per parameter, with the dtypes the
    /// lambda is bound to. They become the fields of the struct the body is applied to.
    ///
    /// The result is computed later, like any applied expression, so errors from running the body
    /// come from executing it.
    pub fn evaluate(&self, args: Vec<ArrayRef>, len: usize) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            args.len() == self.params.nfields(),
            "lambda {self} needs one array per parameter, got {}",
            args.len()
        );
        for ((name, dtype), arg) in self
            .params
            .names()
            .iter()
            .zip(self.params.fields())
            .zip(&args)
        {
            vortex_ensure!(
                arg.dtype() == &dtype,
                "lambda {self}: parameter {name} is bound to {dtype}, not {}",
                arg.dtype()
            );
        }
        let inputs = StructArray::try_new(
            self.params.names().clone(),
            args,
            len,
            Validity::NonNullable,
        )?
        .into_array();
        inputs.apply_bound(&self.body)
    }
}

impl Display for BoundLambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}) -> {}",
            self.params.names().iter().format(", "),
            self.body
        )
    }
}

/// `expr` rebuilt over another root, node by node, so that every node's dtype follows the root's.
fn with_root(expr: &BoundExpression, root: &BoundExpression) -> VortexResult<BoundExpression> {
    if expr.as_scalar().is_none() {
        return Ok(root.clone());
    }
    let children: Vec<_> = expr
        .children()
        .iter()
        .map(|child| with_root(child, root))
        .try_collect()?;
    expr.clone().with_children(children)
}

/// Whether `node`, in the body of a lambda with these parameters, reads a value from outside the
/// lambda: `$`, another lambda's parameter, or the scan's `row_idx()` or `row_count()`. The scan
/// replaces those two before anything runs, but never looks inside a lambda's body, so they too
/// must be passed in as parameters.
fn reads_outside_lambda(node: &Expression, params: &[FieldName]) -> bool {
    match node {
        Expression::Root => true,
        Expression::Variable(name) => !params.contains(name),
        // `row_idx` is defined in vortex-layout, so it is recognized by its id.
        Expression::Scalar { scalar_fn, .. } => {
            scalar_fn.is::<RowCount>() || scalar_fn.id().as_ref() == "vortex.row_idx"
        }
    }
}

/// Whether `node` is a value from outside a lambda with these parameters, read as a whole: a read
/// from outside, or a field of one, such as `$.s.a`.
fn is_capture(node: &Expression, params: &[FieldName]) -> bool {
    if node
        .as_scalar()
        .is_some_and(|scalar_fn| scalar_fn.is::<GetItem>())
    {
        return is_capture(&node.children()[0], params);
    }
    reads_outside_lambda(node, params)
}

/// The name of the parameter for a captured value: the field or variable it reads, such as `a`
/// for `$.s.a`.
fn capture_name(capture: &Expression) -> FieldName {
    match capture {
        Expression::Root => FieldName::from("root"),
        Expression::Variable(name) => name.clone(),
        Expression::Scalar { scalar_fn, .. } => match scalar_fn.as_opt::<GetItem>() {
            Some(field) => field.clone(),
            // `row_idx()` or `row_count()`, named after their ids.
            None => {
                let id = scalar_fn.id();
                let (_, name) = id
                    .as_ref()
                    .rsplit_once('.')
                    .unwrap_or_else(|| ("", id.as_ref()));
                FieldName::from(name)
            }
        },
    }
}

/// `name`, or `name_2`, `name_3`, ... if it is taken.
fn unused_name(name: FieldName, taken: &[FieldName]) -> FieldName {
    if !taken.contains(&name) {
        return name;
    }
    (2..)
        .map(|n| FieldName::from(format!("{name}_{n}")))
        .find(|candidate| !taken.contains(candidate))
        .vortex_expect("some numbered name is free")
}
