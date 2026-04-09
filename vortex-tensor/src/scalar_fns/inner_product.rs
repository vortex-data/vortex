// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inner product expression for tensor-like types.

use std::fmt::Formatter;

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::matcher::AnyTensor;
use crate::scalar_fns::ApproxOptions;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::utils::extract_flat_elements;
use crate::utils::extract_l2_denorm_children;

/// Inner product (dot product) between two columns.
///
/// Computes `sum(a_i * b_i)` over the flat backing buffer of each tensor or vector. For vectors
/// this is the standard dot product; for higher-rank ([`FixedShapeTensor`]) arrays this is the
/// Frobenius inner product.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone)]
pub struct InnerProduct;

impl InnerProduct {
    /// Creates a new [`ScalarFn`] wrapping the inner product operation with the given
    /// [`ApproxOptions`] controlling approximation behavior.
    pub fn new(options: &ApproxOptions) -> ScalarFn<InnerProduct> {
        ScalarFn::new(InnerProduct, options.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily computes the inner product between `lhs` and
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
        ScalarFnArray::try_new(InnerProduct::new(options).erased(), vec![lhs, rhs], len)
    }
}

impl ScalarFnVTable for InnerProduct {
    type Options = ApproxOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.tensor.inner_product")
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

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "inner_product(")?;
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
            "InnerProduct requires both inputs to have the same dtype, got {lhs} and {rhs}"
        );

        // Both inputs must be tensor-like extension types.
        let lhs_ext = lhs
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("InnerProduct lhs must be an extension type, got {lhs}"))?;

        vortex_ensure!(
            lhs_ext.is::<AnyTensor>(),
            "InnerProduct inputs must be an `AnyTensor`, got {lhs}"
        );

        let tensor_match = lhs_ext
            .metadata_opt::<AnyTensor>()
            .ok_or_else(|| vortex_err!("InnerProduct inputs must be an `AnyTensor`, got {lhs}"))?;
        let ptype = tensor_match.element_ptype();
        // TODO(connor): This should support integer tensors!
        vortex_ensure!(
            ptype.is_float(),
            "InnerProduct element dtype must be a float primitive, got {ptype}"
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
        let mut lhs_ref = args.get(0)?;
        let mut rhs_ref = args.get(1)?;
        let len = args.row_count();

        // Check if any of our children have be already normalized.
        {
            let lhs_is_denorm = lhs_ref.is::<ExactScalarFn<L2Denorm>>();
            let rhs_is_denorm = rhs_ref.is::<ExactScalarFn<L2Denorm>>();

            if lhs_is_denorm && rhs_is_denorm {
                return self.execute_both_denorm(options, &lhs_ref, &rhs_ref, len, ctx);
            } else if lhs_is_denorm || rhs_is_denorm {
                if rhs_is_denorm {
                    (lhs_ref, rhs_ref) = (rhs_ref, lhs_ref);
                }
                return self.execute_one_denorm(options, &lhs_ref, &rhs_ref, len, ctx);
            }
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
        let dimensions = tensor_match.list_size();

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

impl InnerProduct {
    /// Both sides are `L2Denorm`: `inner_product = s_l * s_r * dot(n_l, n_r)`.
    fn execute_both_denorm(
        &self,
        options: &ApproxOptions,
        lhs_ref: &ArrayRef,
        rhs_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = lhs_ref.validity()?.and(rhs_ref.validity()?)?;

        let (normalized_l, norms_l) = extract_l2_denorm_children(lhs_ref);
        let (normalized_r, norms_r) = extract_l2_denorm_children(rhs_ref);

        let norms_l: PrimitiveArray = norms_l.execute(ctx)?;
        let norms_r: PrimitiveArray = norms_r.execute(ctx)?;

        let dot: PrimitiveArray =
            InnerProduct::try_new_array(options, normalized_l, normalized_r, len)?
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

    /// One side is `L2Denorm`: `inner_product = s * dot(n, other)`.
    ///
    /// The caller must pass the denorm array as `denorm_ref` and the plain array as `plain_ref`.
    fn execute_one_denorm(
        &self,
        options: &ApproxOptions,
        denorm_ref: &ArrayRef,
        plain_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = denorm_ref.validity()?.and(plain_ref.validity()?)?;

        let (normalized, norms) = extract_l2_denorm_children(denorm_ref);
        let denorm_norms: PrimitiveArray = norms.execute(ctx)?;

        let dot: PrimitiveArray =
            InnerProduct::try_new_array(options, normalized, plain_ref.clone(), len)?
                .into_array()
                .execute(ctx)?;

        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let ns = denorm_norms.as_slice::<T>();
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::scalar_fn::ScalarFn;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::scalar_fns::ApproxOptions;
    use crate::scalar_fns::inner_product::InnerProduct;
    use crate::scalar_fns::l2_denorm::L2Denorm;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Evaluates inner product between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_inner_product(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(InnerProduct, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
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
        assert_close(&eval_inner_product(lhs, rhs, 1)?, expected);
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
        assert_close(&eval_inner_product(lhs, rhs, 3)?, &[0.0, 25.0, 6.0]);
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
        assert_close(&eval_inner_product(lhs, rhs, 2)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 3 rows of dim-2 vectors. Row 1 of lhs is masked as null.
        let lhs = tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let rhs = tensor_array(&[2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
        let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

        let scalar_fn = ScalarFn::new(InnerProduct, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 1*7 + 2*8 = 23, row 1: null, row 2: 5*11 + 6*12 = 127.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert!(prim.is_valid(2)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[23.0]);
        assert_close(&[prim.as_slice::<f64>()[2]], &[127.0]);
        Ok(())
    }

    #[test]
    fn rejects_non_extension_dtype() {
        let lhs = PrimitiveArray::from_iter([1.0_f64, 2.0]).into_array();
        let rhs = PrimitiveArray::from_iter([3.0_f64, 4.0]).into_array();
        let result = InnerProduct::try_new_array(&ApproxOptions::Exact, lhs, rhs, 2);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_mismatched_dtypes() -> VortexResult<()> {
        let lhs = tensor_array(&[2], &[1.0_f64, 2.0])?;
        let rhs = vector_array(2, &[3.0_f64, 4.0])?;
        let result = InnerProduct::try_new_array(&ApproxOptions::Exact, lhs, rhs, 1);
        assert!(result.is_err());
        Ok(())
    }

    /// Creates an `L2Denorm` scalar function array from pre-normalized elements and norms.
    fn l2_denorm_array(
        shape: &[usize],
        normalized_elements: &[f64],
        norms: &[f64],
    ) -> VortexResult<ArrayRef> {
        use vortex_array::IntoArray;

        let len = norms.len();
        let normalized = tensor_array(shape, normalized_elements)?;
        let norms = PrimitiveArray::from_iter(norms.iter().copied()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        Ok(
            L2Denorm::try_new_array(&ApproxOptions::Exact, normalized, norms, len, &mut ctx)?
                .into_array(),
        )
    }

    #[test]
    fn both_denorm() -> VortexResult<()> {
        // LHS: [3.0, 4.0] = L2Denorm([0.6, 0.8], 5.0).
        // RHS: [1.0, 0.0] = L2Denorm([1.0, 0.0], 1.0).
        // dot([3.0, 4.0], [1.0, 0.0]) = 3.0.
        let lhs = l2_denorm_array(&[2], &[0.6, 0.8], &[5.0])?;
        let rhs = l2_denorm_array(&[2], &[1.0, 0.0], &[1.0])?;

        // Expected: 5.0 * 1.0 * dot([0.6, 0.8], [1.0, 0.0]) = 5.0 * 0.6 = 3.0.
        assert_close(&eval_inner_product(lhs, rhs, 1)?, &[3.0]);
        Ok(())
    }

    #[test]
    fn both_denorm_multiple_rows() -> VortexResult<()> {
        // Row 0: [3.0, 4.0] dot [3.0, 4.0] = 25.0.
        // Row 1: [1.0, 0.0] dot [0.0, 1.0] = 0.0.
        let lhs = l2_denorm_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0])?;
        let rhs = l2_denorm_array(&[2], &[0.6, 0.8, 0.0, 1.0], &[5.0, 1.0])?;

        assert_close(&eval_inner_product(lhs, rhs, 2)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn one_side_denorm_lhs() -> VortexResult<()> {
        // LHS: L2Denorm([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // RHS: plain [1.0, 2.0].
        // dot([3.0, 4.0], [1.0, 2.0]) = 3.0 + 8.0 = 11.0.
        let lhs = l2_denorm_array(&[2], &[0.6, 0.8], &[5.0])?;
        let rhs = tensor_array(&[2], &[1.0, 2.0])?;

        assert_close(&eval_inner_product(lhs, rhs, 1)?, &[11.0]);
        Ok(())
    }

    #[test]
    fn one_side_denorm_rhs() -> VortexResult<()> {
        // LHS: plain [1.0, 2.0].
        // RHS: L2Denorm([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // dot([1.0, 2.0], [3.0, 4.0]) = 3.0 + 8.0 = 11.0.
        let lhs = tensor_array(&[2], &[1.0, 2.0])?;
        let rhs = l2_denorm_array(&[2], &[0.6, 0.8], &[5.0])?;

        assert_close(&eval_inner_product(lhs, rhs, 1)?, &[11.0]);
        Ok(())
    }

    #[test]
    fn both_denorm_null_norms() -> VortexResult<()> {
        // Row 0: valid, row 1: null (via nullable norms on lhs).
        let normalized_l = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
        let norms_l = PrimitiveArray::from_option_iter([Some(5.0f64), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let lhs =
            L2Denorm::try_new_array(&ApproxOptions::Exact, normalized_l, norms_l, 2, &mut ctx)?
                .into_array();
        let rhs = l2_denorm_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0])?;

        let scalar_fn = ScalarFn::new(InnerProduct, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 2)?;
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 5.0 * 5.0 * dot([0.6, 0.8], [0.6, 0.8]) = 25.0, row 1: null.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[25.0]);
        Ok(())
    }
}
