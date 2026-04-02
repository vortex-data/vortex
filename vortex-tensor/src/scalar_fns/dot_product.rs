// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dot product (inner product) expression for tensor-like extension arrays
//! ([`FixedShapeTensor`](crate::fixed_shape::FixedShapeTensor) and
//! [`Vector`](crate::vector::Vector)).

use std::fmt::Formatter;

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::compute::cosine_similarity;
use crate::matcher::AnyTensor;
use crate::scalar_fns::ApproxOptions;
use crate::utils::extension_element_ptype;
use crate::utils::extension_list_size;
use crate::utils::extension_storage;
use crate::utils::extract_flat_elements;

/// Dot product (inner product) of two tensor or vector columns.
///
/// Computes `<a, b> = sum(a_i * b_i)` over the flat backing buffers.
///
/// Both inputs must be tensor-like extension arrays with the same float element type
/// and dimensions. The output is a float column of the same float type.
#[derive(Clone)]
pub struct DotProduct;

impl ScalarFnVTable for DotProduct {
    type Options = ApproxOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.tensor.dot_product")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("DotProduct must have exactly two children"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "dot_product(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        let lhs_ext = lhs
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("DotProduct lhs must be an extension type, got {lhs}"))?;

        vortex_ensure!(
            lhs_ext.is::<AnyTensor>(),
            "DotProduct inputs must be an `AnyTensor`, got {lhs}"
        );

        let lhs_ptype = extension_element_ptype(lhs_ext)?;
        vortex_ensure!(
            lhs_ptype.is_float(),
            "DotProduct element dtype must be a float primitive, got {lhs_ptype}"
        );

        let rhs_ext = rhs
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("DotProduct rhs must be an extension type, got {rhs}"))?;

        vortex_ensure!(
            rhs_ext.is::<AnyTensor>(),
            "DotProduct inputs must be an `AnyTensor`, got {rhs}"
        );

        let rhs_ptype = extension_element_ptype(rhs_ext)?;
        vortex_ensure!(
            lhs_ptype == rhs_ptype,
            "DotProduct inputs must have the same element type, got {lhs_ptype} and {rhs_ptype}"
        );

        let lhs_dim = extension_list_size(lhs_ext)?;
        let rhs_dim = extension_list_size(rhs_ext)?;
        vortex_ensure!(
            lhs_dim == rhs_dim,
            "DotProduct inputs must have the same dimension, got {lhs_dim} and {rhs_dim}"
        );

        let nullability = Nullability::from(lhs.is_nullable() || rhs.is_nullable());
        Ok(DType::Primitive(lhs_ptype, nullability))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;
        let row_count = args.row_count();

        let ext = lhs.dtype().as_extension_opt().ok_or_else(|| {
            vortex_err!(
                "dot_product input must be an extension type, got {}",
                lhs.dtype()
            )
        })?;
        let list_size = extension_list_size(ext)? as usize;

        let lhs_storage = extension_storage(&lhs)?;
        let rhs_storage = extension_storage(&rhs)?;

        // TurboQuant approximate path: norm_a * norm_b * quantized unit-norm dot.
        if *options == ApproxOptions::Approximate {
            if let (Some(lhs_tq), Some(rhs_tq)) = (
                lhs_storage.as_opt::<TurboQuant>(),
                rhs_storage.as_opt::<TurboQuant>(),
            ) {
                return cosine_similarity::dot_product_quantized_column(lhs_tq, rhs_tq, ctx);
            }
        }

        let lhs_flat = extract_flat_elements(&lhs_storage, list_size, ctx)?;
        let rhs_flat = extract_flat_elements(&rhs_storage, list_size, ctx)?;

        match_each_float_ptype!(lhs_flat.ptype(), |T| {
            let result: PrimitiveArray = (0..row_count)
                .map(|i| dot_product_row(lhs_flat.row::<T>(i), rhs_flat.row::<T>(i)))
                .collect();

            Ok(result.into_array())
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let lhs_validity = expression.child(0).validity()?;
        let rhs_validity = expression.child(1).validity()?;

        Ok(Some(vortex_array::expr::and(lhs_validity, rhs_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Computes the dot product (inner product) of two float slices.
fn dot_product_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, v| acc + v)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::scalar_fn::ScalarFn;
    use vortex_error::VortexResult;

    use crate::scalar_fns::ApproxOptions;
    use crate::scalar_fns::dot_product::DotProduct;
    use crate::test::SESSION;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::vector_array;

    fn eval_dot_product(
        lhs: ArrayRef,
        rhs: ArrayRef,
        len: usize,
        options: ApproxOptions,
    ) -> VortexResult<Vec<f64>> {
        let mut ctx = SESSION.create_execution_ctx();
        let scalar_fn = ScalarFn::new(DotProduct, options).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
        let prim = result.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        Ok(prim.as_slice::<f64>().to_vec())
    }

    #[rstest]
    #[case::orthogonal(&[1.0, 0.0], &[0.0, 1.0], 0.0)]
    #[case::parallel(&[3.0, 4.0], &[3.0, 4.0], 25.0)]
    #[case::antiparallel(&[1.0, 2.0], &[-1.0, -2.0], -5.0)]
    #[case::scaled(&[2.0, 0.0], &[3.0, 0.0], 6.0)]
    fn known_dot_products(
        #[case] a: &[f64],
        #[case] b: &[f64],
        #[case] expected: f64,
    ) -> VortexResult<()> {
        #[allow(clippy::cast_possible_truncation)]
        let dim = a.len() as u32;
        let lhs = vector_array(dim, a)?;
        let rhs = vector_array(dim, b)?;
        assert_close(
            &eval_dot_product(lhs, rhs, 1, ApproxOptions::Exact)?,
            &[expected],
        );
        Ok(())
    }

    #[test]
    fn multiple_rows() -> VortexResult<()> {
        let lhs = vector_array(2, &[1.0, 0.0, 3.0, 4.0])?;
        let rhs = vector_array(2, &[0.0, 1.0, 3.0, 4.0])?;
        assert_close(
            &eval_dot_product(lhs, rhs, 2, ApproxOptions::Exact)?,
            &[0.0, 25.0],
        );
        Ok(())
    }
}
