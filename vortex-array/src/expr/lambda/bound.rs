// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::Variable;
use crate::expr::traversal::TraversalOrder;
use crate::expr::traversal::pre_order_visit_down;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

/// A lambda bound by its higher-order function: typed, with a body that reads only its inputs.
///
/// The inputs are the parameters followed by the captures, the outer references that binding moved
/// out of the body. The body reads input `i` through a variable at frame `0`, slot `i`. The
/// function holds this in its options, takes the captures as extra children, and evaluates the body
/// with [`BoundLambda::evaluate`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundLambda {
    params: Arc<[(Variable, DType)]>,
    captures: Arc<[DType]>,
    body: Arc<BoundExpression>,
}

impl BoundLambda {
    pub(crate) fn new(
        params: Vec<(Variable, DType)>,
        captures: Vec<DType>,
        body: BoundExpression,
    ) -> Self {
        Self {
            params: params.into(),
            captures: captures.into(),
            body: Arc::new(body),
        }
    }

    /// The parameters with their dtypes, in declaration order.
    pub fn params(&self) -> &[(Variable, DType)] {
        &self.params
    }

    /// The dtypes of the captures, the inputs after the parameters.
    pub fn capture_dtypes(&self) -> &[DType] {
        &self.captures
    }

    /// The body.
    pub fn body(&self) -> &BoundExpression {
        &self.body
    }

    /// The dtype the body evaluates to.
    pub fn body_dtype(&self) -> &DType {
        self.body.dtype()
    }

    /// Whether every function in the body is infallible, so the body may run over values that no
    /// row observes, such as the elements of a null list.
    pub fn is_infallible(&self) -> bool {
        let mut infallible = true;
        pre_order_visit_down(self.body.as_ref(), |node| {
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

    /// Check that inputs of `dtypes`, in order, fit this lambda.
    ///
    /// An input of a nullable dtype also accepts its non-nullable form, because rewrites after
    /// binding can narrow a child's nullability, such as a layout reader stepping into the fields of
    /// a nullable struct. [`BoundLambda::evaluate`] widens such inputs back.
    pub fn check_inputs<'a>(
        &self,
        dtypes: impl IntoIterator<Item = &'a DType>,
    ) -> VortexResult<()> {
        let dtypes = dtypes.into_iter().collect_vec();
        vortex_ensure!(
            dtypes.len() == self.params.len() + self.captures.len(),
            "lambda {self} takes {} input(s) but was given {}",
            self.params.len() + self.captures.len(),
            dtypes.len()
        );
        for (actual, expected) in dtypes.into_iter().zip(self.input_dtypes()) {
            vortex_ensure!(
                actual == expected || actual.as_nullable() == *expected,
                "lambda {self} input has dtype {actual}, expected {expected}"
            );
        }
        Ok(())
    }

    /// Evaluate the body over `inputs`, arrays of equal length, into a lazy array of that length.
    pub fn evaluate(&self, inputs: &[ArrayRef]) -> VortexResult<ArrayRef> {
        self.check_inputs(inputs.iter().map(ArrayRef::dtype))?;
        let Some(len) = inputs.first().map(ArrayRef::len) else {
            vortex_bail!("lambda {self} has no inputs to evaluate over");
        };
        let inputs: Vec<ArrayRef> = inputs
            .iter()
            .zip(self.input_dtypes())
            .map(|(input, dtype)| {
                vortex_ensure!(
                    input.len() == len,
                    "lambda {self} inputs have lengths {} and {len}",
                    input.len()
                );
                if input.dtype() == dtype {
                    Ok(input.clone())
                } else {
                    input.cast(dtype.clone())
                }
            })
            .try_collect()?;
        eval(&self.body, &inputs, len)
    }

    fn input_dtypes(&self) -> impl Iterator<Item = &DType> {
        self.params
            .iter()
            .map(|(_, dtype)| dtype)
            .chain(self.captures.iter())
    }
}

/// Evaluate a bound lambda body, whose variables read `inputs`, into a lazy array of length `len`.
fn eval(node: &BoundExpression, inputs: &[ArrayRef], len: usize) -> VortexResult<ArrayRef> {
    match node {
        BoundExpression::Variable(variable) => match inputs.get(variable.variable_ref().slot()) {
            Some(input) => Ok(input.clone()),
            None => vortex_bail!("lambda body reads input {variable}, which it was not given"),
        },
        BoundExpression::Root { .. } => vortex_bail!("a bound lambda body cannot read root()"),
        BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => {
            if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
                return Ok(ConstantArray::new(scalar.clone(), len).into_array());
            }
            let children: Vec<_> = children
                .iter()
                .map(|child| eval(child, inputs, len))
                .try_collect()?;
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?
                .into_array()
                .optimize()
        }
    }
}

impl Display for BoundLambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let params = self.params.iter().map(|(name, _)| name).join(", ");
        write!(f, "({params}) -> {}", self.body)
    }
}
