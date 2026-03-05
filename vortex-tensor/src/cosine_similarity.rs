// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity expression for [`FixedShapeTensor`] arrays.

use std::fmt::Formatter;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;

/// Cosine similarity between two [`FixedShapeTensor`] columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor. The
/// shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement.
///
/// Both inputs must be [`FixedShapeTensor`] extension arrays with the same dtype and a float
/// element type (`f32` or `f64`). The output is a primitive column of the same float type.
///
/// [`FixedShapeTensor`]: crate::FixedShapeTensor
#[derive(Clone)]
pub struct CosineSimilarity;

impl ScalarFnVTable for CosineSimilarity {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.cosine_similarity")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("CosineSimilarity has exactly two children"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "cosine_similarity(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        todo!("return_dtype")
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        todo!("execute")
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        // TODO(connor): Is this correct since we need to canonicalize?
        false
    }
}
