// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lambdas, the function arguments of higher-order scalar functions such as `list_transform`.
//!
//! A [`Lambda`] is an expression over its own row: one field per parameter, then one field per
//! *capture*, a value the body reads from outside the lambda. A function that takes a lambda keeps
//! it in its options and takes the captures as ordinary children after its value arguments, so
//! binding, traversal and rewrites of the enclosing expression see an ordinary node.

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
use crate::expr::Expression;
use crate::expr::get_item;
use crate::expr::root;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::var::Var;
use crate::validity::Validity;

/// Parameters and a body: the function argument of a higher-order scalar function.
///
/// The body is an ordinary [`Expression`] whose root `$` is the lambda's row: one field per
/// parameter, named after it, then one field per capture, named `c0`, `c1` and so on. A function
/// that takes a lambda keeps it in its options, takes the captures as children after its value
/// arguments, and fills the row with [`Lambda::evaluate`].
///
/// ```rust
/// # use vortex_array::expr::{Lambda, checked_add, col, var};
/// let (lambda, captures) = Lambda::new(["x"], checked_add(var("x"), col("a"))).unwrap();
/// assert_eq!(lambda.to_string(), "(x, c0) -> ($.x + $.c0)");
/// assert_eq!(captures, [col("a")]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lambda {
    /// The row's field names: the parameters, then the captures.
    names: FieldNames,
    /// How many of the row's fields are parameters.
    params: usize,
    /// The body, over the row.
    body: Expression,
    infallible: bool,
}

impl Lambda {
    /// Create a lambda taking `params` from `body`, and return it with its captures.
    ///
    /// The body reads a parameter with [`var`](crate::expr::var) and anything else as usual, for
    /// example with [`col`](crate::expr::col). It is rewritten to read the lambda's row: `var("x")`
    /// of a parameter `x` becomes `$.x`, and each distinct read from outside becomes a capture
    /// `$.cN`. Reads from outside are `$` and field paths into it, variables that are not
    /// parameters of this lambda and field paths into them, and functions without arguments other
    /// than literals, such as `row_idx()`.
    ///
    /// The returned captures are those reads, in order. The function taking the lambda takes them
    /// as children, so a lambda in the body of another is an ordinary node: the outer lambda
    /// rewrites its children, and leaves its body alone.
    ///
    /// Returns an error when a parameter name repeats.
    pub fn new(
        params: impl IntoIterator<Item = impl Into<FieldName>>,
        body: Expression,
    ) -> VortexResult<(Self, Vec<Expression>)> {
        let params: Vec<FieldName> = params.into_iter().map(Into::into).collect();
        if let Some(param) = params.iter().duplicates().next() {
            vortex_bail!("lambda parameter `{param}` is declared more than once");
        }

        let mut rewrite = Rewrite {
            params: &params,
            captures: Vec::new(),
            capture_names: Vec::new(),
            next_name: 0,
        };
        let body = body
            .transform_down(|node| Ok(rewrite.rewrite(node)))?
            .into_inner();
        let Rewrite {
            captures,
            capture_names,
            ..
        } = rewrite;

        let infallible = is_infallible(&body);
        let lambda = Self {
            names: params.iter().chain(&capture_names).cloned().collect(),
            params: params.len(),
            body,
            infallible,
        };
        Ok((lambda, captures))
    }

    /// The parameter names, in order.
    pub fn params(&self) -> &[FieldName] {
        &self.names.as_ref()[..self.params]
    }

    /// The names of the captures in the row, in the order of the captures.
    pub fn capture_names(&self) -> &[FieldName] {
        &self.names.as_ref()[self.params..]
    }

    /// The dtype of the row, given the dtypes of the parameters and of the captures.
    pub fn row_dtype(
        &self,
        param_dtypes: &[DType],
        capture_dtypes: &[DType],
    ) -> VortexResult<DType> {
        self.check_arguments(param_dtypes.len(), capture_dtypes.len())?;
        let dtypes = param_dtypes.iter().chain(capture_dtypes).cloned().collect();
        Ok(DType::Struct(
            StructFields::new(self.names.clone(), dtypes),
            Nullability::NonNullable,
        ))
    }

    /// The dtype of the body, given the dtypes of the parameters and of the captures.
    pub fn return_dtype(
        &self,
        param_dtypes: &[DType],
        capture_dtypes: &[DType],
    ) -> VortexResult<DType> {
        self.body
            .return_dtype(&self.row_dtype(param_dtypes, capture_dtypes)?)
    }

    /// Evaluate the body over `len` rows of the given parameters and captures.
    ///
    /// The body is bound here, against the dtypes of these arrays. They can differ from the dtypes
    /// the enclosing expression was bound against: a scan may, for example, hand over a capture
    /// with a different nullability.
    pub fn evaluate(
        &self,
        params: Vec<ArrayRef>,
        captures: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        self.check_arguments(params.len(), captures.len())?;
        let row = StructArray::try_new(
            self.names.clone(),
            params.into_iter().chain(captures),
            len,
            Validity::NonNullable,
        )?
        .into_array();
        let body = self.body.bind(row.dtype())?;
        row.apply_bound(&body)
    }

    /// Whether every function in the body is infallible, so that the body may run over values
    /// that no row reads, such as the elements of a null list.
    pub fn is_infallible(&self) -> bool {
        self.infallible
    }

    fn check_arguments(&self, params: usize, captures: usize) -> VortexResult<()> {
        vortex_ensure!(
            params == self.params().len(),
            "lambda {self} takes {} parameter(s), got {params}",
            self.params().len()
        );
        vortex_ensure!(
            captures == self.capture_names().len(),
            "lambda {self} has {} capture(s), got {captures}",
            self.capture_names().len()
        );
        Ok(())
    }
}

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}) -> {}", self.names.iter().format(", "), self.body)
    }
}

/// Rewrites a lambda body to read the lambda's row, collecting what it reads from outside.
struct Rewrite<'a> {
    params: &'a [FieldName],
    captures: Vec<Expression>,
    capture_names: Vec<FieldName>,
    next_name: usize,
}

impl Rewrite<'_> {
    fn rewrite(&mut self, node: Expression) -> Transformed<Expression> {
        let field = if let Some(name) = node.as_opt::<Var>()
            && self.params.contains(name)
        {
            name.clone()
        } else if self.reads_outside(&node) {
            self.capture(node)
        } else {
            return Transformed::no(node);
        };
        Transformed {
            value: get_item(field, root()),
            order: TraversalOrder::Skip,
            changed: true,
        }
    }

    /// Whether `node` only reads from outside the lambda: `$`, a variable that is not one of its
    /// parameters, a field path into either, or a function without arguments that is not a
    /// literal.
    fn reads_outside(&self, node: &Expression) -> bool {
        let Some(scalar_fn) = node.as_scalar() else {
            return true;
        };
        match node.children() {
            [] => match scalar_fn.as_opt::<Var>() {
                Some(name) => !self.params.contains(name),
                None => !scalar_fn.is::<Literal>(),
            },
            [child] => scalar_fn.is::<GetItem>() && self.reads_outside(child),
            _ => false,
        }
    }

    /// The row field for `node`, captured once however often the body reads it.
    fn capture(&mut self, node: Expression) -> FieldName {
        if let Some(index) = self.captures.iter().position(|capture| *capture == node) {
            return self.capture_names[index].clone();
        }
        // Capture names skip any name a parameter already has.
        let name = loop {
            let name = FieldName::from(format!("c{}", self.next_name));
            self.next_name += 1;
            if !self.params.contains(&name) {
                break name;
            }
        };
        self.captures.push(node);
        self.capture_names.push(name.clone());
        name
    }
}

fn is_infallible(body: &Expression) -> bool {
    let mut infallible = true;
    pre_order_visit_down(body, |node: &Expression| {
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
