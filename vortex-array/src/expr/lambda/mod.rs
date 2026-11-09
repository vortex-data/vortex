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
//! struct, such as one row per list element, and applies the bound body to it. The body reads
//! nothing but its parameters, so the function's children are everything it reads.

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
use crate::scalar_fn::internal::row_count::RowCount;
use crate::validity::Validity;

/// Parameter names and a body: the function argument of a higher-order scalar function, as
/// written.
///
/// The body is an ordinary [`Expression`] that reads its parameters with
/// [`var`](crate::expr::var), and nothing else from outside the lambda.
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
    /// Returns an error when a parameter repeats, or when the body reads from outside the lambda:
    /// `$` or a column, another lambda's parameter, or the scan's `row_idx()` or `row_count()`.
    pub fn new(
        params: impl IntoIterator<Item = impl Into<FieldName>>,
        body: Expression,
    ) -> VortexResult<Self> {
        let params: FieldNames = params.into_iter().map(Into::into).collect();
        if let Some(param) = params.iter().duplicates().next() {
            vortex_bail!("lambda parameter {param} is declared more than once");
        }
        pre_order_visit_down(&body, |node: &Expression| {
            vortex_ensure!(
                !reads_outside_lambda(node, params.as_ref()),
                "lambda ({}) can only read its parameters, not {node}; reading anything else is \
                 not supported yet",
                params.iter().format(", ")
            );
            Ok(TraversalOrder::Continue)
        })?;

        Ok(Self { params, body })
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
        self.body_over_inputs().return_dtype(&inputs)
    }

    /// Bind the body to these parameter dtypes: `@x` becomes `$.x`, a field of the struct of the
    /// lambda's inputs.
    ///
    /// A function that takes a lambda calls this from its `bind_options`, so the body is bound
    /// once, with the node that holds it, and never while the function runs.
    pub fn bind(&self, param_dtypes: &[DType]) -> VortexResult<BoundLambda> {
        let params = self.param_fields(param_dtypes)?;
        let body = self
            .body_over_inputs()
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
    /// A lambda in the body is a function's options, which this leaves alone: its variables are its
    /// own parameters.
    fn body_over_inputs(&self) -> Expression {
        self.body
            .clone()
            .transform_down(|node| {
                Ok(match &node {
                    Expression::Variable(name) => Transformed {
                        value: get_item(name.clone(), root()),
                        changed: true,
                        order: TraversalOrder::Skip,
                    },
                    _ => Transformed::no(node),
                })
            })
            .vortex_expect("expression traversal cannot fail")
            .into_inner()
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

/// Whether `node`, in the body of a lambda with these parameters, reads a value from outside the
/// lambda, which lambdas can't do yet: `$`, another lambda's parameter, or the scan's `row_idx()`
/// or `row_count()`. The scan replaces those two before anything runs, but never looks inside a
/// lambda's body.
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
