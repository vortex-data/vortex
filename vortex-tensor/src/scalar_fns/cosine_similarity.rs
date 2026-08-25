// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity expression for tensor-like types.

use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::serde::ArrayChildren;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::normalized::NormalizedOrientation;
use crate::encodings::normalized::try_build_constant_normalized;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::utils::BinaryTensorOpMetadata;
use crate::utils::extract_normalized_children;
use crate::utils::validate_binary_tensor_float_inputs;

/// Cosine similarity between two columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor or vector.
/// The shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// When either input is [`Normalized`]-encoded, this operator treats the stored norms and
/// normalized children as authoritative. For lossy normalized children, that means the optimized
/// read-through path may intentionally differ slightly from decoding both sides to dense
/// coordinates and recomputing cosine from scratch.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
/// [`Normalized`]: crate::encodings::normalized::Normalized
#[derive(Clone)]
pub struct CosineSimilarity;

impl CosineSimilarity {
    /// Constructs a [`ScalarFnArray`] that lazily computes the cosine similarity between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(CosineSimilarity.bind(EmptyOptions), vec![lhs, rhs])
    }
}

impl ScalarFnVTable for CosineSimilarity {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.cosine_similarity");
        *ID
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

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        let tensor_match = validate_binary_tensor_float_inputs(lhs, rhs)?;
        let ptype = tensor_match.element_ptype();
        let nullability = Nullability::from(lhs.is_nullable() || rhs.is_nullable());
        Ok(DType::Primitive(ptype, nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let mut lhs_ref = args.get(0)?;
        let mut rhs_ref = args.get(1)?;
        let len = args.row_count();

        // Normalize extension-level constants so the encoded fast path can use them.
        if let Some(normalized_array) = try_build_constant_normalized(&lhs_ref, len, ctx)? {
            lhs_ref = normalized_array.into_array();
        }
        if let Some(normalized_array) = try_build_constant_normalized(&rhs_ref, len, ctx)? {
            rhs_ref = normalized_array.into_array();
        }

        // Take any Normalized read-through fast path that applies.
        match NormalizedOrientation::classify(&lhs_ref, &rhs_ref) {
            NormalizedOrientation::Both { lhs, rhs } => {
                return self.execute_both_normalized(lhs, rhs, len, ctx);
            }
            NormalizedOrientation::One {
                normalized_array,
                plain,
            } => {
                return self.execute_one_normalized(normalized_array, plain, len, ctx);
            }
            NormalizedOrientation::Neither => {}
        }

        // Compute combined validity.
        let validity = lhs_ref.validity()?.and(rhs_ref.validity()?)?;

        // Compute inner product and norms as columnar operations, and propagate the options.
        let norm_lhs_arr = L2Norm::try_new(lhs_ref.clone())?;
        let norm_rhs_arr = L2Norm::try_new(rhs_ref.clone())?;
        let dot_arr = InnerProduct::try_new(lhs_ref, rhs_ref)?;

        // Execute to get the inner product and norms of the arrays. We only fully decompress
        // because we need to perform special logic (guard against 0) during division.
        let dot: PrimitiveArray = dot_arr.into_array().execute(ctx)?;
        let norm_l: PrimitiveArray = norm_lhs_arr.into_array().execute(ctx)?;
        let norm_r: PrimitiveArray = norm_rhs_arr.into_array().execute(ctx)?;

        // TODO(connor): Ideally we would have a `SafeDiv` binary numeric operation.
        // TODO(connor): This can be written in a more SIMD-friendly manner.
        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let norms_l = norm_l.as_slice::<T>();
            let norms_r = norm_r.as_slice::<T>();
            let buffer: Buffer<T> = (0..len)
                .map(|i| {
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
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

impl ScalarFnArrayVTable for CosineSimilarity {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(BinaryTensorOpMetadata::encode_from_view(view)?))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let reconstructed =
            BinaryTensorOpMetadata::decode_children(metadata, len, children, session)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: reconstructed,
        })
    }
}

impl CosineSimilarity {
    /// Both sides are [`Normalized`]-encoded: treat the normalized children as authoritative, so
    /// `cosine_similarity = dot(n_l, n_r)`.
    ///
    /// [`Normalized`]: crate::encodings::normalized::Normalized
    fn execute_both_normalized(
        &self,
        lhs_ref: &ArrayRef,
        rhs_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = lhs_ref.validity()?.and(rhs_ref.validity()?)?;

        let (normalized_l, norms_l) = extract_normalized_children(lhs_ref);
        let (normalized_r, norms_r) = extract_normalized_children(rhs_ref);

        // `Normalized` makes the normalized children authoritative, so their dot product is the
        // cosine similarity even for lossy storage wrappers, except that a zero stored norm still
        // represents a zero vector.
        let dot: PrimitiveArray = InnerProduct::try_new(normalized_l, normalized_r)?
            .into_array()
            .execute(ctx)?;
        let norms_l: PrimitiveArray = norms_l.execute(ctx)?;
        let norms_r: PrimitiveArray = norms_r.execute(ctx)?;

        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let norms_l = norms_l.as_slice::<T>();
            let norms_r = norms_r.as_slice::<T>();
            let buffer: Buffer<T> = (0..len)
                .map(|i| {
                    if norms_l[i] == T::zero() || norms_r[i] == T::zero() {
                        T::zero()
                    } else {
                        dots[i]
                    }
                })
                .collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }

    /// One side is [`Normalized`]-encoded: treat the normalized child as authoritative, so
    /// `cosine_similarity = dot(n, b) / ||b||`.
    ///
    /// [`Normalized`]: crate::encodings::normalized::Normalized
    ///
    /// The caller must pass the [`Normalized`] array as `normalized_ref` and the plain array as `plain_ref`.
    fn execute_one_normalized(
        &self,
        normalized_ref: &ArrayRef,
        plain_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = normalized_ref.validity()?.and(plain_ref.validity()?)?;

        let (normalized, normalized_norms) = extract_normalized_children(normalized_ref);

        let dot_arr = InnerProduct::try_new(normalized, plain_ref.clone())?;
        let dot: PrimitiveArray = dot_arr.into_array().execute(ctx)?;

        let normalized_norms: PrimitiveArray = normalized_norms.execute(ctx)?;

        let norm_arr = L2Norm::try_new(plain_ref.clone())?;
        let plain_norm: PrimitiveArray = norm_arr.into_array().execute(ctx)?;

        // TODO(connor): Ideally we would have a `SafeDiv` binary numeric operation.
        // TODO(connor): This can be written in a more SIMD-friendly manner.
        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let normalized_norms = normalized_norms.as_slice::<T>();
            let plain_norms = plain_norm.as_slice::<T>();
            let buffer: Buffer<T> = (0..len)
                .map(|i| {
                    if normalized_norms[i] == T::zero() || plain_norms[i] == T::zero() {
                        T::zero()
                    } else {
                        dots[i] / plain_norms[i]
                    }
                })
                .collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }
}

#[cfg(test)]
mod tests {

    use rstest::rstest;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::encodings::normalized::Normalized;
    use crate::scalar_fns::cosine_similarity::CosineSimilarity;
    use crate::tests::SESSION;
    use crate::types::vector::Vector;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::constant_tensor_array;
    use crate::utils::test_helpers::normalized_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates cosine similarity between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_cosine_similarity(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<Vec<f64>> {
        let result = CosineSimilarity::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
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
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
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
        assert_close(&eval_cosine_similarity(lhs, rhs)?, expected);
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
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0]);
        Ok(())
    }

    #[test]
    fn scalar_0d() -> VortexResult<()> {
        // 0-dimensional tensor: each "tensor" is a single scalar value.
        let lhs = tensor_array(&[], &[5.0, 3.0])?;
        let rhs = tensor_array(&[], &[5.0, -3.0])?;

        // Same sign -> 1.0, opposite sign -> -1.0.
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, -1.0]);
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
            &eval_cosine_similarity(lhs, rhs)?,
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

        assert_close(&eval_cosine_similarity(data, query)?, &[1.0, 0.0, 0.0, 1.0]);
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
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
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
        let query = Vector::constant_array(&[1.0, 0.0, 0.0], 4)?;

        assert_close(&eval_cosine_similarity(data, query)?, &[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 2 rows of dim-2 vectors. Row 1 of rhs is masked as null.
        let lhs = tensor_array(&[2], &[3.0, 4.0, 1.0, 0.0])?;
        let rhs = tensor_array(&[2], &[3.0, 4.0, 0.0, 1.0])?;
        let rhs = MaskedArray::try_new(rhs, Validity::from_iter([true, false]))?.into_array();

        let result = CosineSimilarity::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: self-similarity = 1.0, row 1: null.
        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[1.0]);
        Ok(())
    }

    #[test]
    fn both_normalized_self_similarity() -> VortexResult<()> {
        // [3.0, 4.0] has norm 5.0, normalized [0.6, 0.8].
        // [1.0, 0.0] has norm 1.0, normalized [1.0, 0.0].
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;
        let rhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

        // Self-similarity should always be 1.0.
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn both_normalized_orthogonal() -> VortexResult<()> {
        // [3.0, 0.0] normalized [1.0, 0.0], norm 3.0.
        // [0.0, 4.0] normalized [0.0, 1.0], norm 4.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[1.0, 0.0], &[3.0], &mut ctx)?;
        let rhs = normalized_array(&[2], &[0.0, 1.0], &[4.0], &mut ctx)?;

        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0]);
        Ok(())
    }

    #[test]
    fn both_normalized_zero_norm() -> VortexResult<()> {
        // Zero-norm row: normalized is [0.0, 0.0], norm is 0.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8, 0.0, 0.0], &[5.0, 0.0], &mut ctx)?;
        let rhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

        // Row 0: dot([0.6, 0.8], [0.6, 0.8]) = 1.0, row 1: dot([0,0], [1,0]) = 0.0.
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn one_side_normalized_lhs() -> VortexResult<()> {
        // LHS is Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // RHS is plain [3.0, 4.0].
        // cosine_similarity([3.0, 4.0], [3.0, 4.0]) = 1.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;
        let rhs = tensor_array(&[2], &[3.0, 4.0])?;

        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0]);
        Ok(())
    }

    #[test]
    fn one_side_normalized_rhs() -> VortexResult<()> {
        // LHS is plain [1.0, 0.0], RHS is Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // cosine_similarity([1.0, 0.0], [3.0, 4.0]) = 3.0 / (1.0 * 5.0) = 0.6.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = tensor_array(&[2], &[1.0, 0.0])?;
        let rhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;

        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.6]);
        Ok(())
    }

    #[test]
    fn both_normalized_null_rows() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

        let normalized_r = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
        let norms_r = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();
        let validity = Validity::from_iter([true, false]);
        let rhs = Normalized::try_new(normalized_r, norms_r, validity, &mut ctx)?.into_array();

        let result = CosineSimilarity::try_new(lhs, rhs)?;
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[1.0]);
        Ok(())
    }

    #[test]
    fn both_normalized_lossy_zero_stored_norm_returns_zero() -> VortexResult<()> {
        // Mimics a lossy encoding where the stored norm is authoritative but
        // the decoded normalized child is physically nonzero. With a stored norm of `0.0`, cosine
        // similarity for that row must be `0.0` even though the dot product of the normalized
        // children is nonzero.
        let normalized_l = tensor_array(&[2], &[0.6, 0.8])?;
        let norms_l = PrimitiveArray::from_iter([0.0f64]).into_array();
        // Intentionally violates the unit-norm invariant by pairing a nonzero normalized row
        // with a stored norm of `0.0`, mimicking lossy storage.
        // SAFETY: The children are structurally valid.
        let lhs =
            unsafe { Normalized::new_unchecked(normalized_l, norms_l, Validity::NonNullable) }
                .into_array();

        let normalized_r = tensor_array(&[2], &[0.6, 0.8])?;
        let norms_r = PrimitiveArray::from_iter([0.0f64]).into_array();
        // Same as above for the rhs operand.
        // SAFETY: The children are structurally valid.
        let rhs =
            unsafe { Normalized::new_unchecked(normalized_r, norms_r, Validity::NonNullable) }
                .into_array();

        // `dot(normalized_l, normalized_r) = 1.0`, but the authoritative stored norms are both
        // `0.0`, so cosine similarity must be `0.0`.
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0]);
        Ok(())
    }

    #[test]
    fn one_side_normalized_lossy_zero_stored_norm_returns_zero() -> VortexResult<()> {
        // Mimics a lossy encoding where the stored norm is authoritative but
        // the decoded normalized child is physically nonzero. The plain side is a normal nonzero
        // tensor with positive norm. cosine similarity must still be `0.0` because the
        // authoritative stored norm on the normalized_array side is `0.0`.
        let normalized = tensor_array(&[2], &[0.6, 0.8])?;
        let norms = PrimitiveArray::from_iter([0.0f64]).into_array();
        // Intentionally pairs a nonzero normalized row with a stored norm of `0.0`, mimicking
        // lossy storage where the stored norm is authoritative.
        // SAFETY: The children are structurally valid.
        let normalized_array =
            unsafe { Normalized::new_unchecked(normalized, norms, Validity::NonNullable) }
                .into_array();

        let plain = tensor_array(&[2], &[1.0, 0.0])?;

        // Normalized encoding on the lhs: `One { normalized_array: lhs, plain: rhs }`.
        assert_close(
            &eval_cosine_similarity(normalized_array.clone(), plain.clone())?,
            &[0.0],
        );

        // Normalized encoding on the rhs: `One { normalized_array: rhs, plain: lhs }`. The same
        // zero-norm guard must fire regardless of operand order.
        assert_close(&eval_cosine_similarity(plain, normalized_array)?, &[0.0]);
        Ok(())
    }

    #[test]
    fn constant_lhs_matches_plain_tensor() -> VortexResult<()> {
        // The constant query `[1, 2, 2]` has norm 3, so its normalized form is `[1/3, 2/3, 2/3]`.
        // Expected cosine similarity against each row is `dot([1, 2, 2], row) / (3 * ||row||)`.
        let lhs = constant_tensor_array(&[3], &[1.0, 2.0, 2.0], 4)?;
        let rhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // dot=1, ||rhs||=1, expected=1/3
                1.0, 2.0, 2.0, // dot=9, ||rhs||=3, expected=1
                0.0, 0.0, 1.0, // dot=2, ||rhs||=1, expected=2/3
                2.0, 1.0, 2.0, // dot=8, ||rhs||=3, expected=8/9
            ],
        )?;
        assert_close(
            &eval_cosine_similarity(lhs, rhs)?,
            &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
        );
        Ok(())
    }

    #[test]
    fn constant_rhs_matches_plain_tensor() -> VortexResult<()> {
        // Mirror of `constant_lhs_matches_plain_tensor` with the constant on the right.
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, //
                1.0, 2.0, 2.0, //
                0.0, 0.0, 1.0, //
                2.0, 1.0, 2.0, //
            ],
        )?;
        let rhs = constant_tensor_array(&[3], &[1.0, 2.0, 2.0], 4)?;
        assert_close(
            &eval_cosine_similarity(lhs, rhs)?,
            &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
        );
        Ok(())
    }

    #[test]
    fn both_constant_tensors() -> VortexResult<()> {
        // `[1, 0, 0]` vs `[1, 1, 0]`. dot=1, ||lhs||=1, ||rhs||=sqrt(2), expected=1/sqrt(2).
        let lhs = constant_tensor_array(&[3], &[1.0, 0.0, 0.0], 3)?;
        let rhs = constant_tensor_array(&[3], &[1.0, 1.0, 0.0], 3)?;
        let expected = 1.0 / 2.0_f64.sqrt();
        assert_close(
            &eval_cosine_similarity(lhs, rhs)?,
            &[expected, expected, expected],
        );
        Ok(())
    }

    #[test]
    fn constant_zero_norm_query() -> VortexResult<()> {
        // A zero-norm constant query must produce `0.0` for every row via the zero-norm guard in
        // `execute_one_normalized` and `execute_both_normalized`.
        let lhs = constant_tensor_array(&[3], &[0.0, 0.0, 0.0], 3)?;
        let rhs = tensor_array(
            &[3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        )?;
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn constant_self_similarity_nonunit() -> VortexResult<()> {
        // A non-unit constant query compared to itself must produce `1.0`. This exercises the
        // helper's division: after normalization, both sides must be exactly unit so the
        // Normalized fast path's inner product yields 1.
        let lhs = constant_tensor_array(&[3], &[3.0, 4.0, 0.0], 5)?;
        let rhs = constant_tensor_array(&[3], &[3.0, 4.0, 0.0], 5)?;
        assert_close(&eval_cosine_similarity(lhs, rhs)?, &[1.0; 5]);
        Ok(())
    }

    #[test]
    fn vector_constant_matches_plain() -> VortexResult<()> {
        // Exercise the `Vector` extension variant through the new pre-pass.
        let lhs = Vector::constant_array(&[1.0, 2.0, 2.0], 4)?;
        let rhs = vector_array(
            3,
            &[
                1.0, 0.0, 0.0, //
                1.0, 2.0, 2.0, //
                0.0, 0.0, 1.0, //
                2.0, 1.0, 2.0, //
            ],
        )?;
        assert_close(
            &eval_cosine_similarity(lhs, rhs)?,
            &[1.0 / 3.0, 1.0, 2.0 / 3.0, 8.0 / 9.0],
        );
        Ok(())
    }

    #[rstest]
    #[case::vector(cosine_vector_lhs(), cosine_vector_rhs())]
    #[case::fixed_shape_tensor(cosine_tensor_lhs(), cosine_tensor_rhs())]
    fn serde_round_trip(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) -> VortexResult<()> {
        let original = CosineSimilarity::try_new(lhs.clone(), rhs.clone())?.into_array();

        let plugin = ScalarFnArrayPlugin::new(CosineSimilarity);
        let metadata = plugin
            .serialize(&original, &SESSION)?
            .expect("CosineSimilarity serialize must produce metadata");

        let children = vec![lhs, rhs];
        let recovered = plugin.deserialize(
            original.dtype(),
            original.len(),
            &metadata,
            &[],
            &children,
            &SESSION,
        )?;

        assert_eq!(recovered.dtype(), original.dtype());
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered.encoding_id(), original.encoding_id());
        Ok(())
    }

    fn cosine_vector_lhs() -> ArrayRef {
        vector_array(3, &[1.0, 0.0, 0.0, 3.0, 4.0, 0.0]).expect("valid vector array")
    }

    fn cosine_vector_rhs() -> ArrayRef {
        vector_array(3, &[0.0, 1.0, 0.0, 3.0, 4.0, 0.0]).expect("valid vector array")
    }

    fn cosine_tensor_lhs() -> ArrayRef {
        tensor_array(&[2], &[1.0, 0.0, 3.0, 4.0]).expect("valid tensor array")
    }

    fn cosine_tensor_rhs() -> ArrayRef {
        tensor_array(&[2], &[0.0, 1.0, 3.0, 4.0]).expect("valid tensor array")
    }
}
