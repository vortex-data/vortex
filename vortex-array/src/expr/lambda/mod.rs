// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lambdas, the function arguments of higher-order scalar functions such as `list_transform`.
//!
//! A [`Lambda`] is a list of parameter names and a body that reads them with
//! [`var`](crate::expr::var). A function that takes a lambda keeps it in its options. When the
//! function's node is bound, it binds the lambda to its parameters' dtypes, once, and keeps the
//! [`BoundLambda`] instead.
//!
//! Binding turns the lambda into an ordinary expression over a struct of its inputs: `@x` becomes
//! `$.x`. To run it, the function puts the inputs' arrays in that struct, such as one row per list
//! element, and applies the bound body to it.
//!
//! The body may also read values from outside the lambda, such as a column. The function takes
//! them as one more child, the lambda's [environment](Lambda::environment), so that a rewrite of
//! the expression sees everything it reads. The environment's fields are the lambda's other
//! inputs, named the way the body reads them, so the body itself stays as written.

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
use crate::expr::pack;
use crate::expr::root;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::internal::row_count::RowCount;
use crate::validity::Validity;

/// Parameter names and a body: the function argument of a higher-order scalar function, as
/// written.
///
/// The body is an ordinary [`Expression`] that reads its parameters with
/// [`var`](crate::expr::var). It may also read from outside the lambda: see
/// [`Lambda::environment`].
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
    /// Returns an error when a parameter repeats, or is named `$`, the name of the row in the
    /// lambda's environment.
    pub fn new(
        params: impl IntoIterator<Item = impl Into<FieldName>>,
        body: Expression,
    ) -> VortexResult<Self> {
        let params: FieldNames = params.into_iter().map(Into::into).collect();
        if let Some(param) = params.iter().duplicates().next() {
            vortex_bail!("lambda parameter {param} is declared more than once");
        }
        vortex_ensure!(
            params.iter().all(|param| param.as_ref() != ROW),
            "a lambda parameter can't be named {ROW}, the row the lambda reads from outside"
        );
        Ok(Self { params, body })
    }

    /// The values the body reads from outside the lambda, as one struct, or `None` when it reads
    /// nothing but its parameters.
    ///
    /// A function that takes the lambda takes this as one more child, so that a rewrite of the
    /// expression, such as `replace` or field pruning, sees everything the lambda reads. Each field
    /// is named the way the body reads it: `$` for the row, another lambda's parameter by its name,
    /// and the scan's `row_idx()` or `row_count()` by its id. A value is cut down to the fields the
    /// body reads below it.
    ///
    /// ```rust
    /// # use vortex_array::expr::{Lambda, checked_add, col, var};
    /// let lambda = Lambda::new(["x"], checked_add(var("x"), col("a"))).unwrap();
    /// assert_eq!(lambda.environment().unwrap().to_string(), "pack($: pack(a: $.a))");
    /// ```
    pub fn environment(&self) -> Option<Expression> {
        let reads = self.outside_reads();
        (!reads.is_empty()).then(|| {
            pack(
                reads
                    .into_iter()
                    .map(|read| (read.name, project(read.value, &read.paths))),
                Nullability::NonNullable,
            )
        })
    }

    /// The parameter names, in order.
    pub fn params(&self) -> &[FieldName] {
        self.params.as_ref()
    }

    /// The body, as written.
    pub fn body(&self) -> &Expression {
        &self.body
    }

    /// The dtype of the body, given the parameters' dtypes and the fields of the environment,
    /// without binding it.
    pub fn return_dtype(
        &self,
        param_dtypes: &[DType],
        environment: &StructFields,
    ) -> VortexResult<DType> {
        let inputs = inputs_dtype(self.inputs(param_dtypes, environment)?);
        self.body_over_inputs().return_dtype(&inputs)
    }

    /// Bind the body to these parameter dtypes and fields of the [environment](Self::environment)
    /// (empty without one): `@x` becomes `$.x`, a field of the struct of the lambda's inputs, which
    /// are its parameters followed by the environment's fields. `$` becomes `$.$`, the row in the
    /// environment.
    ///
    /// A function that takes a lambda calls this from its `bind_options`, so the body is bound
    /// once, with the node that holds it, and never while the function runs.
    pub fn bind(
        &self,
        param_dtypes: &[DType],
        environment: &StructFields,
    ) -> VortexResult<BoundLambda> {
        let params = self.inputs(param_dtypes, environment)?;
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

    /// The lambda's inputs: its parameters, with these dtypes, then the fields of its environment.
    fn inputs(
        &self,
        param_dtypes: &[DType],
        environment: &StructFields,
    ) -> VortexResult<StructFields> {
        vortex_ensure!(
            param_dtypes.len() == self.params.len(),
            "lambda {self} needs one dtype per parameter, got {}",
            param_dtypes.len()
        );
        for read in self.outside_reads() {
            vortex_ensure!(
                environment.find(&read.name).is_some(),
                "lambda {self} reads {} from outside, which its function must pass in as the \
                 lambda's environment",
                read.value
            );
        }
        let names: FieldNames = self
            .params
            .iter()
            .chain(environment.names().iter())
            .cloned()
            .collect();
        let dtypes = param_dtypes
            .iter()
            .cloned()
            .chain(environment.fields())
            .collect();
        Ok(StructFields::new(names, dtypes))
    }

    /// The values the body reads from outside the lambda, in the order it first reads them.
    fn outside_reads(&self) -> Vec<OutsideRead> {
        let mut reads: Vec<OutsideRead> = Vec::new();
        pre_order_visit_down(&self.body, |node: &Expression| {
            let Some((value, path)) = outside_read(node, self.params.as_ref()) else {
                return Ok(TraversalOrder::Continue);
            };
            let name = environment_name(&value);
            match reads.iter_mut().find(|read| read.name == name) {
                Some(read) => read.paths.push(path),
                None => reads.push(OutsideRead {
                    name,
                    value,
                    paths: vec![path],
                }),
            }
            Ok(TraversalOrder::Skip)
        })
        .vortex_expect("expression traversal cannot fail");
        reads
    }

    /// The body over the struct of the lambda's inputs, where `@x` reads the field `x` and a value
    /// from outside the lambda reads its field of the environment: `$` reads `$.$`.
    ///
    /// A lambda in the body is a function's options, which this leaves alone: it reads its own
    /// inputs.
    fn body_over_inputs(&self) -> Expression {
        let input = |name: FieldName| Transformed {
            value: get_item(name, root()),
            changed: true,
            order: TraversalOrder::Skip,
        };
        self.body
            .clone()
            .transform_down(|node| {
                Ok(match &node {
                    Expression::Variable(name) => input(name.clone()),
                    _ if reads_outside_lambda(&node, self.params.as_ref()) => {
                        input(environment_name(&node))
                    }
                    _ => Transformed::no(node),
                })
            })
            .vortex_expect("expression traversal cannot fail")
            .into_inner()
    }
}

/// The field of a lambda's environment that holds the row.
const ROW: &str = "$";

/// A value the body of a lambda reads from outside it.
struct OutsideRead {
    /// The field of the lambda's environment that holds the value.
    name: FieldName,
    value: Expression,
    /// The fields the body reads below the value, in the order it reads them. An empty path reads
    /// the whole value.
    paths: Vec<Vec<FieldName>>,
}

/// The value from outside a lambda with these parameters that `node` reads, and the fields it reads
/// below it: `$` and `[s, a]` for `$.s.a`.
fn outside_read(node: &Expression, params: &[FieldName]) -> Option<(Expression, Vec<FieldName>)> {
    let mut path = Vec::new();
    let mut value = node;
    while let Some(field) = value.as_opt::<GetItem>() {
        path.push(field.clone());
        value = &value.children()[0];
    }
    path.reverse();
    reads_outside_lambda(value, params).then(|| (value.clone(), path))
}

/// The field of a lambda's environment that holds this value from outside the lambda.
fn environment_name(value: &Expression) -> FieldName {
    match value {
        Expression::Root => FieldName::from(ROW),
        Expression::Variable(name) => name.clone(),
        Expression::Scalar { scalar_fn, .. } => FieldName::from(scalar_fn.id().as_ref()),
    }
}

/// `value` cut down to the fields read at `paths`, keeping their nesting: `pack(a: $.a, s: pack(i:
/// $.s.i))` for `$` read at `a` and `s.i`. An empty path reads the whole value.
fn project(value: Expression, paths: &[Vec<FieldName>]) -> Expression {
    if paths.iter().any(Vec::is_empty) {
        return value;
    }
    let fields = paths
        .iter()
        .map(|path| path[0].clone())
        .unique()
        .collect_vec();
    pack(
        fields.into_iter().map(|field| {
            let below = paths
                .iter()
                .filter(|path| path[0] == field)
                .map(|path| path[1..].to_vec())
                .collect_vec();
            let value = project(get_item(field.clone(), value.clone()), &below);
            (field, value)
        }),
        Nullability::NonNullable,
    )
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

/// A [`Lambda`] bound to its inputs' dtypes, ready to run.
///
/// The body is a [`BoundExpression`] over the struct of the lambda's inputs, its parameters then
/// its environment's fields: `@x` was bound as `$.x`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    params: StructFields,
    body: BoundExpression,
}

impl BoundLambda {
    /// The inputs' names and dtypes, in order: the parameters, then the environment's fields.
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

    /// Whether the lambda is bound to these input dtypes.
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

    /// Evaluate the body over `len` rows of `args`, one array per input, with the dtypes the lambda
    /// is bound to. They become the fields of the struct the body is applied to.
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
/// lambda: `$`, another lambda's parameter, or the scan's `row_idx()` or `row_count()`. The scan
/// replaces those two before anything runs, but never looks inside a lambda's body, so they too
/// come in through the environment.
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
