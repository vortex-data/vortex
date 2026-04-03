// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 norm expression for tensor-like types.

use std::fmt::Formatter;

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
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

use crate::encodings::turboquant::TurboQuant;
use crate::matcher::AnyTensor;
use crate::scalar_fns::ApproxOptions;
use crate::utils::extension_element_ptype;
use crate::utils::extension_list_size;
use crate::utils::extract_flat_elements;

/// L2 norm (Euclidean norm) of a tensor or vector column.
///
/// Computes `||v|| = sqrt(sum(v_i^2))` over the flat backing buffer of each tensor-like type.
///
/// The input must be a tensor-like extension array with a float element type. The output is a float
/// column of the same float type.
#[derive(Clone)]
pub struct L2Norm;

impl L2Norm {
    /// Creates a new [`ScalarFn`] wrapping the L2 norm operation with the given [`ApproxOptions`]
    /// controlling approximation behavior.
    pub fn new(options: &ApproxOptions) -> ScalarFn<L2Norm> {
        ScalarFn::new(L2Norm, options.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily computes the L2 norm over `child`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new_array(
        options: &ApproxOptions,
        child: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Norm::new(options).erased(), vec![child], len)
    }
}

impl ScalarFnVTable for L2Norm {
    type Options = ApproxOptions;

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
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input_ref = args.get(0)?;
        let row_count = args.row_count();

        // TurboQuant stores exact precomputed norms -- no decompression needed.
        // Norms are currently stored as f32; cast to the target dtype if needed
        // (e.g., if the input extension has f64 elements).
        if let Some(tq) = input_ref.as_opt::<TurboQuant>() {
            let ext = input_ref.dtype().as_extension();
            let target_ptype = extension_element_ptype(ext)?;
            let norms: PrimitiveArray = tq.norms().clone().execute(ctx)?;
            let target_dtype = DType::Primitive(target_ptype, input_ref.dtype().nullability());
            return norms.into_array().cast(target_dtype);
        }

        let input: ExtensionArray = input_ref.execute(ctx)?;
        let validity = input.as_ref().validity()?;

        // Get element ptype and list size from the dtype (validated by `return_dtype`).
        let ext = input.dtype().as_extension();
        let list_size = extension_list_size(ext)? as usize;

        let storage = input.data().storage_array();
        let flat = extract_flat_elements(storage, list_size, ctx)?;

        match_each_float_ptype!(flat.ptype(), |T| {
            let buffer: Buffer<T> = (0..row_count)
                .map(|i| l2_norm_row(flat.row::<T>(i)))
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::scalar_fn::ScalarFn;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::ApproxOptions;
    use crate::scalar_fns::l2_norm::L2Norm;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates L2 norm on a tensor/vector array and returns the result as `Vec<f64>`.
    fn eval_l2_norm(input: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(L2Norm, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![input], len)?;
        let prim = result.as_array().to_primitive();
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

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 2 rows of dim-2 vectors. Row 1 is masked as null.
        let arr = tensor_array(&[2], &[3.0, 4.0, 0.0, 0.0])?;
        let arr = MaskedArray::try_new(arr, Validity::from_iter([true, false]))?.into_array();

        let scalar_fn = ScalarFn::new(L2Norm, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![arr], 2)?;
        let prim = result.as_array().to_primitive();

        // Row 0: norm = 5.0, row 1: null.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[5.0]);
        Ok(())
    }
}
