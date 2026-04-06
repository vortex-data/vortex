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

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::compute::cosine_similarity;
use crate::matcher::AnyTensor;
use crate::scalar_fns::ApproxOptions;
use crate::utils::extract_flat_elements;

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
        ScalarFnId::new_ref("vortex.tensor.inner_product")
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
        let lhs_ref = args.get(0)?;
        let rhs_ref = args.get(1)?;

        let row_count = args.row_count();

        // We validated that both inputs have the same type.
        let ext = lhs_ref.dtype().as_extension();
        let tensor_match = ext
            .metadata_opt::<AnyTensor>()
            .vortex_expect("we already validated this in `return_dtype`");
        let dimensions = tensor_match.list_size();

        // TurboQuant approximate path: check encoding before executing.
        if options.is_approx()
            && let (Some(lhs_tq), Some(rhs_tq)) = (
                lhs_ref.as_opt::<TurboQuant>(),
                rhs_ref.as_opt::<TurboQuant>(),
            )
        {
            return cosine_similarity::dot_product_quantized_column(lhs_tq, rhs_tq, ctx);
        }

        let lhs: ExtensionArray = lhs_ref.execute(ctx)?;
        let rhs: ExtensionArray = rhs_ref.execute(ctx)?;

        // Compute combined validity.
        let rhs_validity = rhs.as_ref().validity()?;
        let validity = lhs.as_ref().validity()?.and(rhs_validity)?;

        // Extract the storage array from each extension input. We pass the storage (FSL) rather
        // than the extension array to avoid canonicalizing the extension wrapper.
        let lhs_storage = lhs.storage_array();
        let rhs_storage = rhs.storage_array();

        let lhs_flat = extract_flat_elements(lhs_storage, dimensions, ctx)?;
        let rhs_flat = extract_flat_elements(rhs_storage, dimensions, ctx)?;

        match_each_float_ptype!(lhs_flat.ptype(), |T| {
            let buffer: Buffer<T> = (0..row_count)
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::scalar_fn::ScalarFn;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::ApproxOptions;
    use crate::scalar_fns::inner_product::InnerProduct;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates inner product between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_inner_product(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(InnerProduct, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
        let prim = result.as_array().to_primitive();
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
        let prim = result.as_array().to_primitive();

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
}
