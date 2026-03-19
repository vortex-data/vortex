// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity expression for tensor-like extension arrays
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

impl ScalarFnVTable for CosineSimilarity {
    type Options = EmptyOptions;

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
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;
        let row_count = args.row_count();

        // Get list size from the dtype. Both sides should have the same dtype.
        let ext = lhs.dtype().as_extension_opt().ok_or_else(|| {
            vortex_err!(
                "cosine_similarity input must be an extension type, got {}",
                lhs.dtype()
            )
        })?;
        let list_size = extension_list_size(ext)?;

        // Extract the storage array from each extension input. We pass the storage (FSL) rather
        // than the extension array to avoid canonicalizing the extension wrapper.
        let lhs_storage = extension_storage(&lhs)?;
        let rhs_storage = extension_storage(&rhs)?;

        let lhs_flat = extract_flat_elements(&lhs_storage, list_size)?;
        let rhs_flat = extract_flat_elements(&rhs_storage, list_size)?;

        match_each_float_ptype!(lhs_flat.ptype(), |T| {
            let result: PrimitiveArray = (0..row_count)
                .map(|i| cosine_similarity_row(lhs_flat.row::<T>(i), rhs_flat.row::<T>(i)))
                .collect();

            Ok(result.into_array())
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

        Ok(Some(vortex::expr::and(lhs_validity, rhs_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

// TODO(connor): We should try to use a more performant library instead of doing this ourselves.
/// Computes cosine similarity between two equal-length float slices.
///
/// Returns `dot(a, b) / (||a|| * ||b||)`. When either vector has zero norm, this naturally
/// produces `NaN` via `0.0 / 0.0`, matching standard floating-point semantics.
fn cosine_similarity_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    let mut dot = T::zero();
    let mut norm_a = T::zero();
    let mut norm_b = T::zero();
    for i in 0..a.len() {
        dot = dot + a[i] * b[i];
        norm_a = norm_a + a[i] * a[i];
        norm_b = norm_b + b[i] * b[i];
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::ArrayRef;
    use vortex::array::ToCanonical;
    use vortex::array::arrays::ScalarFnArray;
    use vortex::error::VortexResult;
    use vortex::scalar_fn::EmptyOptions;
    use vortex::scalar_fn::ScalarFn;

    use crate::scalar_fns::cosine_similarity::CosineSimilarity;
    use crate::scalar_fns::utils::test_helpers::assert_close;
    use crate::scalar_fns::utils::test_helpers::constant_tensor_array;
    use crate::scalar_fns::utils::test_helpers::constant_vector_array;
    use crate::scalar_fns::utils::test_helpers::tensor_array;
    use crate::scalar_fns::utils::test_helpers::vector_array;

    /// Evaluates cosine similarity between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_cosine_similarity(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(CosineSimilarity, EmptyOptions).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
        let prim = result.to_primitive();
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
    // Zero vector -> 0/0 -> NaN.
    #[case::zero_norm(&[2], &[0.0, 0.0],       &[1.0, 0.0],       &[f64::NAN])]
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
}
