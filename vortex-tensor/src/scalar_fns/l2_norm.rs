// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 norm expression for tensor-like extension arrays
//! ([`FixedShapeTensor`](crate::fixed_shape::FixedShapeTensor) and
//! [`Vector`](crate::vector::Vector)).

use std::fmt::Formatter;

use num_traits::Float;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_float_ptype;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
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

use crate::matcher::AnyTensor;
use crate::scalar_fns::utils::extension_element_ptype;
use crate::scalar_fns::utils::extension_list_size;
use crate::scalar_fns::utils::extension_storage;
use crate::scalar_fns::utils::extract_flat_elements;

/// L2 norm (Euclidean norm) of a tensor or vector column.
///
/// Computes `||v|| = sqrt(sum(v_i^2))` over the flat backing buffer of each tensor-like type.
///
/// The input must be a tensor-like extension array with a float element type. The output is a float
/// column of the same float type.
#[derive(Clone)]
pub struct L2Norm;

impl ScalarFnVTable for L2Norm {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.tensor.l2_norm")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("L2Norm must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "l2_norm(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];

        // Input must be a tensor-like extension type.
        let ext = input_dtype.as_extension_opt().ok_or_else(|| {
            vortex_err!("L2Norm input must be an extension type, got {input_dtype}")
        })?;

        vortex_ensure!(
            ext.is::<AnyTensor>(),
            "L2Norm input must be an `AnyTensor`, got {input_dtype}"
        );

        let ptype = extension_element_ptype(ext)?;
        vortex_ensure!(
            ptype.is_float(),
            "L2Norm element dtype must be a float primitive, got {ptype}"
        );

        let nullability = Nullability::from(input_dtype.is_nullable());
        Ok(DType::Primitive(ptype, nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let row_count = args.row_count();

        // Get list size (dimensions) from the dtype.
        let ext = input.dtype().as_extension_opt().ok_or_else(|| {
            vortex_err!(
                "l2_norm input must be an extension type, got {}",
                input.dtype()
            )
        })?;
        let list_size = extension_list_size(ext)?;

        let storage = extension_storage(&input)?;
        let flat = extract_flat_elements(&storage, list_size)?;

        match_each_float_ptype!(flat.ptype(), |T| {
            let result: PrimitiveArray = (0..row_count)
                .map(|i| l2_norm_row(flat.row::<T>(i)))
                .collect();

            Ok(result.into_array())
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // The result is null if the input tensor is null.
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Computes the L2 norm (Euclidean norm) of a float slice.
///
/// Returns `sqrt(sum(v_i^2))`. A zero-length or all-zero input produces `0.0`.
fn l2_norm_row<T: Float + NativePType>(v: &[T]) -> T {
    let mut sum_sq = T::zero();
    for &x in v {
        sum_sq = sum_sq + x * x;
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::ToCanonical;
    use vortex::array::arrays::ScalarFnArray;
    use vortex::error::VortexResult;
    use vortex::scalar_fn::EmptyOptions;
    use vortex::scalar_fn::ScalarFn;

    use crate::scalar_fns::l2_norm::L2Norm;
    use crate::scalar_fns::utils::test_helpers::assert_close;
    use crate::scalar_fns::utils::test_helpers::tensor_array;
    use crate::scalar_fns::utils::test_helpers::vector_array;

    /// Evaluates L2 norm on a tensor/vector array and returns the result as `Vec<f64>`.
    fn eval_l2_norm(input: vortex::array::ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(L2Norm, EmptyOptions).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![input], len)?;
        let prim = result.to_primitive();
        Ok(prim.as_slice::<f64>().to_vec())
    }

    #[rstest]
    #[case::three_four_five(&[2], &[3.0, 4.0], &[5.0])]
    #[case::zero_vector(&[3], &[0.0, 0.0, 0.0], &[0.0])]
    #[case::single_element(&[1], &[7.0], &[7.0])]
    #[case::negative_elements(&[2], &[-3.0, -4.0], &[5.0])]
    fn known_norms(
        #[case] shape: &[usize],
        #[case] elements: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let arr = tensor_array(shape, elements)?;
        assert_close(&eval_l2_norm(arr, 1)?, expected);
        Ok(())
    }

    #[test]
    fn multiple_rows() -> VortexResult<()> {
        let arr = tensor_array(
            &[3],
            &[
                3.0, 4.0, 0.0, // norm = 5.0
                0.0, 0.0, 0.0, // norm = 0.0
                1.0, 1.0, 1.0, // norm = sqrt(3)
            ],
        )?;
        assert_close(&eval_l2_norm(arr, 3)?, &[5.0, 0.0, 3.0_f64.sqrt()]);
        Ok(())
    }

    #[test]
    fn vector_multiple_rows() -> VortexResult<()> {
        let arr = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // norm = 1.0
                3.0, 4.0, 0.0, // norm = 5.0
            ],
        )?;
        assert_close(&eval_l2_norm(arr, 2)?, &[1.0, 5.0]);
        Ok(())
    }
}
