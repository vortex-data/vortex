// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity expression for tensor-like types.

use std::fmt::Formatter;

use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFn;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::matcher::AnyTensor;
use crate::scalar_fns::ApproxOptions;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::utils::extension_element_ptype;

/// Cosine similarity between two columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor or vector.
/// The shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone)]
pub struct CosineSimilarity;

impl CosineSimilarity {
    /// Creates a new [`ScalarFn`] wrapping the cosine similarity operation with the given
    /// [`ApproxOptions`] controlling approximation behavior.
    pub fn new(options: &ApproxOptions) -> ScalarFn<CosineSimilarity> {
        ScalarFn::new(CosineSimilarity, options.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily computes the cosine similarity between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new_array(
        options: &ApproxOptions,
        lhs: ArrayRef,
        rhs: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(CosineSimilarity::new(options).erased(), vec![lhs, rhs], len)
    }
}

impl ScalarFnVTable for CosineSimilarity {
    type Options = ApproxOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.tensor.cosine_similarity")
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
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        // Both must have the same dtype (ignoring top-level nullability).
        vortex_ensure!(
            lhs.eq_ignore_nullability(rhs),
            "CosineSimilarity requires both inputs to have the same dtype, got {lhs} and {rhs}"
        );

        // We don't need to look at rhs anymore since we know lhs and rhs are equal.

        // Both inputs must be tensor-like extension types.
        let lhs_ext = lhs.as_extension_opt().ok_or_else(|| {
            vortex_err!("CosineSimilarity lhs must be an extension type, got {lhs}")
        })?;

        vortex_ensure!(
            lhs_ext.is::<AnyTensor>(),
            "CosineSimilarity inputs must be an `AnyTensor`, got {lhs}"
        );

        let ptype = extension_element_ptype(lhs_ext)?;
        vortex_ensure!(
            ptype.is_float(),
            "CosineSimilarity element dtype must be a float primitive, got {ptype}"
        );

        let nullability = Nullability::from(lhs.is_nullable() || rhs.is_nullable());
        Ok(DType::Primitive(ptype, nullability))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?.execute::<ExtensionArray>(ctx)?.into_array();
        let rhs = args.get(1)?.execute::<ExtensionArray>(ctx)?.into_array();

        let len = args.row_count();

        // Compute combined validity.
        let validity = lhs.validity()?.and(rhs.validity()?)?;

        // Compute inner product and norms as columnar operations, and propagate the options.
        let norm_lhs_arr = L2Norm::try_new_array(options, lhs.clone(), len)?;
        let norm_rhs_arr = L2Norm::try_new_array(options, rhs.clone(), len)?;
        let dot_arr = InnerProduct::try_new_array(options, lhs, rhs, len)?;

        // Execute to get PrimitiveArrays.
        let dot: PrimitiveArray = dot_arr.into_array().execute(ctx)?;
        let norm_l: PrimitiveArray = norm_lhs_arr.into_array().execute(ctx)?;
        let norm_r: PrimitiveArray = norm_rhs_arr.into_array().execute(ctx)?;

        // Divide element-wise, guarding against zero norms.
        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let norms_l = norm_l.as_slice::<T>();
            let norms_r = norm_r.as_slice::<T>();
            let buffer: Buffer<T> = (0..len)
                .map(|i| {
                    // TODO(connor): Would it be better to make this a binary multiply?
                    // What happens when this overflows???
                    let denom = norms_l[i] * norms_r[i];

                    if denom == T::zero() {
                        T::zero()
                    } else {
                        dots[i] / denom
                    }
                })
                .collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // The result is null if either input tensor is null.
        let lhs_validity = expression.child(0).validity()?;
        let rhs_validity = expression.child(1).validity()?;

        Ok(Some(and(lhs_validity, rhs_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::scalar_fn::ScalarFn;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::ApproxOptions;
    use crate::scalar_fns::cosine_similarity::CosineSimilarity;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::constant_tensor_array;
    use crate::utils::test_helpers::constant_vector_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates cosine similarity between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_cosine_similarity(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(CosineSimilarity, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
        let prim = result.as_array().to_primitive();
        Ok(prim.as_slice::<f64>().to_vec())
    }

    #[test]
    fn unit_vectors_1d() -> VortexResult<()> {
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // Tensor 1
                0.0, 1.0, 0.0, // Tensor 2
            ],
        )?;
        let rhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // Tensor 1
                1.0, 0.0, 0.0, // Tensor 2
            ],
        )?;

        // Row 0: identical -> 1.0, row 1: orthogonal -> 0.0.
        assert_close(&eval_cosine_similarity(lhs, rhs, 2)?, &[1.0, 0.0]);
        Ok(())
    }

    /// Single-row cosine similarity for various vector pairs.
    #[rstest]
    // Antiparallel -> -1.0.
    #[case::opposite(&[3], &[1.0, 0.0, 0.0],  &[-1.0, 0.0, 0.0], &[-1.0])]
    // dot=24, both magnitudes=5 -> 24/25 = 0.96.
    #[case::non_unit(&[2], &[3.0, 4.0],        &[4.0, 3.0],       &[0.96])]
    // Zero vector -> guarded to 0.0.
    #[case::zero_norm(&[2], &[0.0, 0.0],       &[1.0, 0.0],       &[0.0])]
    fn single_row(
        #[case] shape: &[usize],
        #[case] lhs_elems: &[f64],
        #[case] rhs_elems: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let lhs = tensor_array(shape, lhs_elems)?;
        let rhs = tensor_array(shape, rhs_elems)?;
        assert_close(&eval_cosine_similarity(lhs, rhs, 1)?, expected);
        Ok(())
    }

    /// Self-similarity across various tensor shapes should always produce 1.0.
    #[rstest]
    // 2x3 matrix, flattened to 6 elements.
    #[case::matrix_2d(
        &[2, 3],
        &[
            1.0, 0.0, 0.0, // row 0
            0.0, 0.0, 0.0, // row 1
        ],
    )]
    // 2x2x2 tensor, 8 elements.
    #[case::tensor_3d(&[2, 2, 2], &[1.0; 8])]
    fn self_similarity(#[case] shape: &[usize], #[case] elements: &[f64]) -> VortexResult<()> {
        let lhs = tensor_array(shape, elements)?;
        let rhs = tensor_array(shape, elements)?;
        assert_close(&eval_cosine_similarity(lhs, rhs, 1)?, &[1.0]);
        Ok(())
    }

    #[test]
    fn scalar_0d() -> VortexResult<()> {
        // 0-dimensional tensor: each "tensor" is a single scalar value.
        let lhs = tensor_array(&[], &[5.0, 3.0])?;
        let rhs = tensor_array(&[], &[5.0, -3.0])?;

        // Same sign -> 1.0, opposite sign -> -1.0.
        assert_close(&eval_cosine_similarity(lhs, rhs, 2)?, &[1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn many_rows() -> VortexResult<()> {
        // 5 tensors of shape [4] compared against themselves -> all 1.0.
        let lhs = tensor_array(
            &[4],
            &[
                1.0, 2.0, 3.0, 4.0, // tensor 0
                0.0, 1.0, 0.0, 0.0, // tensor 1
                5.0, 0.0, 5.0, 0.0, // tensor 2
                1.0, 1.0, 1.0, 1.0, // tensor 3
                0.0, 0.0, 0.0, 7.0, // tensor 4
            ],
        )?;
        let rhs = lhs.clone();

        assert_close(
            &eval_cosine_similarity(lhs, rhs, 5)?,
            &[1.0, 1.0, 1.0, 1.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn constant_query_tensor() -> VortexResult<()> {
        // Compare 4 tensors of shape [3] against a single constant query tensor [1,0,0].
        let data = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // tensor 0
                0.0, 1.0, 0.0, // tensor 1
                0.0, 0.0, 1.0, // tensor 2
                1.0, 0.0, 0.0, // tensor 3
            ],
        )?;
        let query = constant_tensor_array(&[3], &[1.0, 0.0, 0.0], 4)?;

        assert_close(
            &eval_cosine_similarity(data, query, 4)?,
            &[1.0, 0.0, 0.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn vector_unit_vectors() -> VortexResult<()> {
        let lhs = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // vector 0
                0.0, 1.0, 0.0, // vector 1
            ],
        )?;
        let rhs = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // vector 0
                1.0, 0.0, 0.0, // vector 1
            ],
        )?;

        // Row 0: identical -> 1.0, row 1: orthogonal -> 0.0.
        assert_close(&eval_cosine_similarity(lhs, rhs, 2)?, &[1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn vector_constant_query() -> VortexResult<()> {
        let data = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, // vector 0
                0.0, 1.0, 0.0, // vector 1
                0.0, 0.0, 1.0, // vector 2
                1.0, 0.0, 0.0, // vector 3
            ],
        )?;
        let query = constant_vector_array(&[1.0, 0.0, 0.0], 4)?;

        assert_close(
            &eval_cosine_similarity(data, query, 4)?,
            &[1.0, 0.0, 0.0, 1.0],
        );
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 2 rows of dim-2 vectors. Row 1 of rhs is masked as null.
        let lhs = tensor_array(&[2], &[3.0, 4.0, 1.0, 0.0])?;
        let rhs = tensor_array(&[2], &[3.0, 4.0, 0.0, 1.0])?;
        let rhs = MaskedArray::try_new(rhs, Validity::from_iter([true, false]))?.into_array();

        let scalar_fn = ScalarFn::new(CosineSimilarity, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 2)?;
        let prim = result.as_array().to_primitive();

        // Row 0: self-similarity = 1.0, row 1: null.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[1.0]);
        Ok(())
    }
}
