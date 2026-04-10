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
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFn;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// L2 norm (Euclidean norm) of a tensor or vector column.
///
/// Computes `||v|| = sqrt(sum(v_i^2))` over the flat backing buffer of each tensor-like type.
///
/// The input must be a tensor-like extension array with a float element type. The output is a float
/// column of the same float type.
///
/// When the input is wrapped in [`L2Denorm`], this operator treats the stored norms as
/// authoritative. For lossy encodings such as TurboQuant, that means `L2Norm` may intentionally
/// read the stored norms instead of re-deriving them from fully decoded coordinates. That behavior
/// is part of the lossy storage contract, not a separate lossy-compute mode.
#[derive(Clone)]
pub struct L2Norm;

impl L2Norm {
    /// Creates a new [`ScalarFn`] wrapping the L2 norm operation.
    pub fn new() -> ScalarFn<L2Norm> {
        ScalarFn::new(L2Norm, EmptyOptions)
    }

    /// Constructs a [`ScalarFnArray`] that lazily computes the L2 norm over `child`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new_array(child: ArrayRef, len: usize) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Norm::new().erased(), vec![child], len)
    }
}

impl ScalarFnVTable for L2Norm {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.tensor.l2_norm")
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
        let tensor_match = validate_tensor_float_input(input_dtype)?;
        let ptype = tensor_match.element_ptype();

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

        let ext = input_ref.dtype().as_extension();
        let tensor_match = ext
            .metadata_opt::<AnyTensor>()
            .vortex_expect("we already validated this in `return_dtype`");
        let tensor_flat_size = tensor_match.list_size();
        let element_ptype = tensor_match.element_ptype();

        // L2Norm(L2Denorm(normalized, norms)) is defined to read back the authoritative stored
        // norms. Exact callers of lossy encodings like TurboQuant opt into that storage semantics
        // instead of forcing a decode-and-recompute path here.
        if let Some(sfn) = input_ref.as_opt::<ExactScalarFn<L2Denorm>>() {
            let norms = sfn
                .nth_child(1)
                .vortex_expect("L2Denom must have at 2 children");

            vortex_ensure_eq!(
                norms.dtype(),
                &DType::Primitive(element_ptype, input_ref.dtype().nullability())
            );

            return Ok(norms);
        }

        let input: ExtensionArray = input_ref.execute(ctx)?;
        let validity = input.as_ref().validity()?;

        let storage = input.storage_array();
        let flat = extract_flat_elements(storage, tensor_flat_size, ctx)?;

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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::scalar_fns::l2_norm::L2Norm;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Evaluates L2 norm on a tensor/vector array and returns the result as `Vec<f64>`.
    fn eval_l2_norm(input: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = L2Norm::new().erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![input], len)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
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

        let scalar_fn = L2Norm::new().erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![arr], 2)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: norm = 5.0, row 1: null.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[5.0]);
        Ok(())
    }
}
