// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inner product expression for tensor-like types.

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::normalized::NormalizedOrientation;
use crate::matcher::AnyTensor;
use crate::utils::BinaryTensorOpMetadata;
use crate::utils::extract_flat_elements;
use crate::utils::extract_normalized_children;
use crate::utils::validate_binary_tensor_float_inputs;

/// Inner product (dot product) between two columns.
///
/// Computes `sum(a_i * b_i)` over the flat backing buffer of each tensor or vector. For vectors
/// this is the standard dot product; for higher-rank ([`FixedShapeTensor`]) arrays this is the
/// Frobenius inner product.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone)]
pub struct InnerProduct;

impl InnerProduct {
    /// Constructs a [`ScalarFnArray`] that lazily computes the inner product between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(InnerProduct.bind(EmptyOptions), vec![lhs, rhs])
    }
}

impl ScalarFnVTable for InnerProduct {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.inner_product");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("InnerProduct must have exactly two children"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        // TODO(connor): relax the float-only gate once integer tensors are supported.
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
        let lhs_ref = args.get(0)?;
        let rhs_ref = args.get(1)?;
        let len = args.row_count();

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

        // Canonicalize so we can perform the math directly.
        let lhs: ExtensionArray = lhs_ref.execute(ctx)?;
        let rhs: ExtensionArray = rhs_ref.execute(ctx)?;

        // We validated that both inputs have the same type.
        let ext = lhs.dtype().as_extension();
        let tensor_match = ext
            .metadata_opt::<AnyTensor>()
            .vortex_expect("we already validated this in `return_dtype`");
        let dimensions = tensor_match.list_size() as usize;

        // Extract the storage array from each extension input. We pass the storage (FSL) rather
        // than the extension array to avoid canonicalizing the extension wrapper.
        let lhs_storage = lhs.storage_array();
        let rhs_storage = rhs.storage_array();

        let lhs_flat = extract_flat_elements(lhs_storage, dimensions, ctx)?;
        let rhs_flat = extract_flat_elements(rhs_storage, dimensions, ctx)?;

        match_each_float_ptype!(lhs_flat.ptype(), |T| {
            let buffer: Buffer<T> = (0..len)
                .map(|i| inner_product_row(lhs_flat.row::<T>(i), rhs_flat.row::<T>(i)))
                .collect();

            // SAFETY: The buffer length equals `row_count`, which matches the source validity
            // length.
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

impl ScalarFnArrayVTable for InnerProduct {
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

impl InnerProduct {
    /// Both sides are [`Normalized`]-encoded: `inner_product = s_l * s_r * dot(n_l, n_r)`.
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

        let norms_l: PrimitiveArray = norms_l.execute(ctx)?;
        let norms_r: PrimitiveArray = norms_r.execute(ctx)?;

        let dot: PrimitiveArray = InnerProduct::try_new(normalized_l, normalized_r)?
            .into_array()
            .execute(ctx)?;

        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let nl = norms_l.as_slice::<T>();
            let nr = norms_r.as_slice::<T>();
            let buffer: Buffer<T> = (0..len).map(|i| nl[i] * nr[i] * dots[i]).collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }

    /// One side is [`Normalized`]-encoded: `inner_product = s * dot(n, other)`.
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

        let (normalized, norms) = extract_normalized_children(normalized_ref);
        let normalized_norms: PrimitiveArray = norms.execute(ctx)?;

        let dot: PrimitiveArray = InnerProduct::try_new(normalized, plain_ref.clone())?
            .into_array()
            .execute(ctx)?;

        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let ns = normalized_norms.as_slice::<T>();
            let buffer: Buffer<T> = (0..len).map(|i| ns[i] * dots[i]).collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }
}

/// Computes the inner product (dot product) of two equal-length float slices.
///
/// Returns `sum(a_i * b_i)`.
fn inner_product_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, v| acc + v)
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
    use crate::scalar_fns::inner_product::InnerProduct;
    use crate::tests::SESSION;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::normalized_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates inner product between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_inner_product(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<Vec<f64>> {
        let result = InnerProduct::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
        Ok(prim.as_slice::<f64>().to_vec())
    }

    /// Single-row inner product for various vector pairs.
    #[rstest]
    // Orthogonal: [1, 0] . [0, 1] = 0.
    #[case::orthogonal(&[2], &[1.0, 0.0], &[0.0, 1.0], &[0.0])]
    // Parallel: [3, 4] . [3, 4] = 9 + 16 = 25.
    #[case::parallel(&[2], &[3.0, 4.0], &[3.0, 4.0], &[25.0])]
    // Antiparallel: [1, 2] . [-1, -2] = -1 + -4 = -5.
    #[case::antiparallel(&[2], &[1.0, 2.0], &[-1.0, -2.0], &[-5.0])]
    // Scaled: [2, 0] . [3, 0] = 6.
    #[case::scaled(&[2], &[2.0, 0.0], &[3.0, 0.0], &[6.0])]
    fn single_row(
        #[case] shape: &[usize],
        #[case] lhs_elems: &[f64],
        #[case] rhs_elems: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let lhs = tensor_array(shape, lhs_elems)?;
        let rhs = tensor_array(shape, rhs_elems)?;
        assert_close(&eval_inner_product(lhs, rhs)?, expected);
        Ok(())
    }

    #[test]
    fn multiple_rows() -> VortexResult<()> {
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // tensor 0
                3.0, 4.0, 0.0, // tensor 1
                1.0, 1.0, 1.0, // tensor 2
            ],
        )?;
        let rhs = tensor_array(
            &[3],
            &[
                0.0, 1.0, 0.0, // tensor 0: dot = 0
                3.0, 4.0, 0.0, // tensor 1: dot = 25
                2.0, 2.0, 2.0, // tensor 2: dot = 6
            ],
        )?;
        assert_close(&eval_inner_product(lhs, rhs)?, &[0.0, 25.0, 6.0]);
        Ok(())
    }

    #[test]
    fn vector_inner_product() -> VortexResult<()> {
        let lhs = vector_array(
            2,
            &[
                3.0, 4.0, // vector 0
                1.0, 0.0, // vector 1
            ],
        )?;
        let rhs = vector_array(
            2,
            &[
                3.0, 4.0, // vector 0: dot = 25
                0.0, 1.0, // vector 1: dot = 0
            ],
        )?;
        assert_close(&eval_inner_product(lhs, rhs)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 3 rows of dim-2 vectors. Row 1 of lhs is masked as null.
        let lhs = tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let rhs = tensor_array(&[2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
        let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

        let result = InnerProduct::try_new(lhs, rhs)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 1*7 + 2*8 = 23, row 1: null, row 2: 5*11 + 6*12 = 127.
        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert!(prim.is_valid(2, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[23.0]);
        assert_close(&[prim.as_slice::<f64>()[2]], &[127.0]);
        Ok(())
    }

    #[test]
    fn rejects_non_extension_dtype() {
        let lhs = PrimitiveArray::from_iter([1.0_f64, 2.0]).into_array();
        let rhs = PrimitiveArray::from_iter([3.0_f64, 4.0]).into_array();
        let result = InnerProduct::try_new(lhs, rhs);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_mismatched_dtypes() -> VortexResult<()> {
        let lhs = tensor_array(&[2], &[1.0_f64, 2.0])?;
        let rhs = vector_array(2, &[3.0_f64, 4.0])?;
        let result = InnerProduct::try_new(lhs, rhs);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn both_normalized() -> VortexResult<()> {
        // LHS: [3.0, 4.0] = Normalized([0.6, 0.8], 5.0).
        // RHS: [1.0, 0.0] = Normalized([1.0, 0.0], 1.0).
        // dot([3.0, 4.0], [1.0, 0.0]) = 3.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;
        let rhs = normalized_array(&[2], &[1.0, 0.0], &[1.0], &mut ctx)?;

        // Expected: 5.0 * 1.0 * dot([0.6, 0.8], [1.0, 0.0]) = 5.0 * 0.6 = 3.0.
        assert_close(&eval_inner_product(lhs, rhs)?, &[3.0]);
        Ok(())
    }

    #[test]
    fn both_normalized_multiple_rows() -> VortexResult<()> {
        // Row 0: [3.0, 4.0] dot [3.0, 4.0] = 25.0.
        // Row 1: [1.0, 0.0] dot [0.0, 1.0] = 0.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;
        let rhs = normalized_array(&[2], &[0.6, 0.8, 0.0, 1.0], &[5.0, 1.0], &mut ctx)?;

        assert_close(&eval_inner_product(lhs, rhs)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn one_side_normalized_lhs() -> VortexResult<()> {
        // LHS: Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // RHS: plain [1.0, 2.0].
        // dot([3.0, 4.0], [1.0, 2.0]) = 3.0 + 8.0 = 11.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;
        let rhs = tensor_array(&[2], &[1.0, 2.0])?;

        assert_close(&eval_inner_product(lhs, rhs)?, &[11.0]);
        Ok(())
    }

    #[test]
    fn one_side_normalized_rhs() -> VortexResult<()> {
        // LHS: plain [1.0, 2.0].
        // RHS: Normalized([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // dot([1.0, 2.0], [3.0, 4.0]) = 3.0 + 8.0 = 11.0.
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = tensor_array(&[2], &[1.0, 2.0])?;
        let rhs = normalized_array(&[2], &[0.6, 0.8], &[5.0], &mut ctx)?;

        assert_close(&eval_inner_product(lhs, rhs)?, &[11.0]);
        Ok(())
    }

    #[test]
    fn both_normalized_null_rows() -> VortexResult<()> {
        let normalized_l = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
        let norms_l = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let validity = Validity::from_iter([true, false]);
        let lhs = Normalized::try_new(normalized_l, norms_l, validity, &mut ctx)?.into_array();
        let rhs = normalized_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0], &mut ctx)?;

        let result = InnerProduct::try_new(lhs, rhs)?;
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 5.0 * 5.0 * dot([0.6, 0.8], [0.6, 0.8]) = 25.0, row 1: null.
        assert!(prim.is_valid(0, &mut ctx)?);
        assert!(!prim.is_valid(1, &mut ctx)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[25.0]);
        Ok(())
    }

    #[rstest]
    #[case::vector(inner_product_vector_lhs(), inner_product_vector_rhs())]
    #[case::fixed_shape_tensor(inner_product_tensor_lhs(), inner_product_tensor_rhs())]
    fn serde_round_trip(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) -> VortexResult<()> {
        let original = InnerProduct::try_new(lhs.clone(), rhs.clone())?.into_array();

        let plugin = ScalarFnArrayPlugin::new(InnerProduct);
        let metadata = plugin
            .serialize(&original, &SESSION)?
            .expect("InnerProduct serialize must produce metadata");

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

    fn inner_product_vector_lhs() -> ArrayRef {
        vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid vector array")
    }

    fn inner_product_vector_rhs() -> ArrayRef {
        vector_array(3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("valid vector array")
    }

    fn inner_product_tensor_lhs() -> ArrayRef {
        tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0]).expect("valid tensor array")
    }

    fn inner_product_tensor_rhs() -> ArrayRef {
        tensor_array(&[2], &[5.0, 6.0, 7.0, 8.0]).expect("valid tensor array")
    }
}
