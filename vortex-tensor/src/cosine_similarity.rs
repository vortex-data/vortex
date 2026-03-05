// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity expression for [`FixedShapeTensor`] arrays.

use std::fmt::Formatter;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
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
            _ => unreachable!("CosineSimilarity must have exactly two children"),
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

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        debug_assert_eq!(arg_dtypes.len(), 2);

        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        // Both must have the same dtype (ignoring top-level nullability).
        vortex_ensure!(
            lhs.eq_ignore_nullability(rhs),
            "cosine_similarity requires both inputs to have the same dtype, got {lhs} and {rhs}"
        );

        // We don't need to look at rhs anymore since we know lhs and rhs are equal.

        // Both inputs must be extension types.
        let lhs_ext = lhs.as_extension_opt().ok_or_else(|| {
            vortex_err!("cosine_similarity lhs must be an extension type, got {lhs}")
        })?;

        // Extract the element dtype from the storage FixedSizeList.
        let element_dtype = lhs_ext
            .storage_dtype()
            .as_fixed_size_list_element_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "cosine_similarity storage dtype must be a FixedSizeList, got {}",
                    lhs_ext.storage_dtype()
                )
            })?;

        // Element dtype must be a non-nullable float primitive.
        vortex_ensure!(
            element_dtype.is_float(),
            "cosine_similarity element dtype must be a float primitive, got {element_dtype}"
        );
        vortex_ensure!(
            !element_dtype.is_nullable(),
            "cosine_similarity element dtype must be non-nullable"
        );

        let ptype = element_dtype.as_ptype();
        let nullability = Nullability::from(lhs.is_nullable() || rhs.is_nullable());
        Ok(DType::Primitive(ptype, nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        todo!("execute")
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // The result is null if either input tensor is null.
        let lhs_validity = expression.child(0).validity()?;
        let rhs_validity = expression.child(1).validity()?;

        Ok(Some(vortex::expr::and(lhs_validity, rhs_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        // TODO(connor): Is this correct since we need to canonicalize?
        false
    }
}
