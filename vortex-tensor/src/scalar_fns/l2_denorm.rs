// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 denormalization expression for tensor-like types.

use std::fmt::Formatter;

use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFn;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;

use crate::matcher::AnyTensor;
use crate::scalar_fns::ApproxOptions;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// Re-applies L2 norms to a normalized tensor column.
///
/// Computes `normalized * norm` on each row over the flat backing buffer of each tensor-like type.
///
/// The normalized input must be a tensor-like extension array with a float element type and each
/// non-null row is semantically required to already be L2-normalized.
///
/// The norms input must be a primitive float column with the same element type as the normalized
/// tensor elements.
#[derive(Clone)]
pub struct L2Denorm;

impl L2Denorm {
    /// Creates a new [`ScalarFn`] wrapping the L2 denormalization operation with the given
    /// [`ApproxOptions`] controlling approximation behavior.
    pub fn new(options: &ApproxOptions) -> ScalarFn<L2Denorm> {
        ScalarFn::new(L2Denorm, options.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily re-applies `norms` to `normalized`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new_array(
        options: &ApproxOptions,
        normalized: ArrayRef,
        norms: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        // TODO(connor): Should figure out a way to do validation here instead of inside
        // `return_dtype()` which this calls?
        ScalarFnArray::try_new(
            L2Denorm::new(options).erased(),
            vec![normalized, norms],
            len,
        )
    }
}

impl ScalarFnVTable for L2Denorm {
    type Options = ApproxOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.tensor.l2_denorm")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("normalized"),
            1 => ChildName::from("norms"),
            _ => unreachable!("L2Denorm must have exactly two children"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "l2_denorm(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let normalized = &arg_dtypes[0];
        let norms = &arg_dtypes[1];

        let tensor_match = validate_tensor_float_input(normalized)?;
        let element_ptype = tensor_match.element_ptype();

        let DType::Primitive(norms_ptype, _) = norms else {
            vortex_bail!("L2Denorm norms must be a primitive float array, got {norms}");
        };
        vortex_ensure_eq!(
            *norms_ptype,
            element_ptype,
            "L2Denorm norms dtype must match normalized element dtype ({element_ptype}), \
                got {norms_ptype}",
        );

        Ok(normalized.union_nullability(norms.nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let normalized: ExtensionArray = args.get(0)?.execute(ctx)?;
        let norms: PrimitiveArray = args.get(1)?.execute(ctx)?;
        let row_count = args.row_count();

        let tensor_match = normalized
            .dtype()
            .as_extension()
            .metadata_opt::<AnyTensor>()
            .vortex_expect("we already validated this in `return_dtype`");
        let tensor_flat_size = tensor_match.list_size();

        let validity = normalized.as_ref().validity()?.and(norms.validity()?)?;
        let output_dtype = normalized
            .dtype()
            .union_nullability(norms.dtype().nullability());
        let flat = extract_flat_elements(normalized.storage_array(), tensor_flat_size, ctx)?;

        // TODO(connor): Theoretically we could model this as a multiplication between the
        // normalized array and a `RunEnd(Sequence(0, dimensions), norms)`. But since we have
        // already canonicalized the array, it is probably not faster to do that.
        match_each_float_ptype!(flat.ptype(), |T| {
            let norms = norms.as_slice::<T>();

            let elements: Buffer<T> = (0..row_count)
                .flat_map(|i| {
                    let norm = norms[i];
                    flat.row::<T>(i).iter().map(move |&x| x * norm)
                })
                .collect();

            build_tensor_array(
                output_dtype,
                tensor_flat_size,
                row_count,
                validity,
                elements,
            )
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let normalized_validity = expression.child(0).validity()?;
        let norms_validity = expression.child(1).validity()?;

        Ok(Some(and(normalized_validity, norms_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Builds an unexecuted [`L2Denorm`] expression by normalizing `input` and reattaching the exact
/// norms as the norms child.
///
/// The returned array is a lazy `L2Denorm(normalized, norms)` scalar function array.
#[allow(dead_code, reason = "TODO(connor): Use this in a scheme")]
fn normalize_as_l2_denorm(
    options: &ApproxOptions,
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnArray> {
    let row_count = input.len();
    let tensor_match = validate_tensor_float_input(input.dtype())?;
    let tensor_flat_size = tensor_match.list_size();

    let norms_sfn = L2Norm::try_new_array(options, input.clone(), row_count)?;
    let norms_array: ArrayRef = norms_sfn.into_array().execute(ctx)?;
    let norms: PrimitiveArray = norms_array.clone().execute(ctx)?;

    let input: ExtensionArray = input.execute(ctx)?;
    let validity = input.as_ref().validity()?;
    let normalized_dtype = input.dtype().clone();
    let flat = extract_flat_elements(input.storage_array(), tensor_flat_size, ctx)?;

    let normalized = match_each_float_ptype!(flat.ptype(), |T| {
        let norm_values = norms.as_slice::<T>();
        let elements: Buffer<T> = (0..row_count)
            .flat_map(|i| {
                let norm = norm_values[i];
                flat.row::<T>(i).iter().map(move |&x| {
                    if norm == T::zero() {
                        T::zero()
                    } else {
                        x / norm
                    }
                })
            })
            .collect();

        build_tensor_array(
            normalized_dtype,
            tensor_flat_size,
            row_count,
            validity,
            elements,
        )
    })?;

    // TODO(connor): Need to figure out a way to not run validation.
    L2Denorm::try_new_array(options, normalized, norms_array, row_count)
}

/// Rebuilds a tensor-like extension array from flat primitive elements.
///
/// # Errors
///
/// Returns an error if the elements are invalid (have incorrect lengths for the
/// `FixedSizeListArray` storage array).
fn build_tensor_array<T: NativePType>(
    dtype: DType,
    tensor_flat_size: usize,
    row_count: usize,
    validity: Validity,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    let list_size =
        u32::try_from(tensor_flat_size).vortex_expect("tensor flat size must fit into `u32`");

    // SAFETY: Validity has no length (because tensor elements are always non-nullable).
    let elements = unsafe { PrimitiveArray::new_unchecked(elements, Validity::NonNullable) };

    let storage =
        FixedSizeListArray::try_new(elements.into_array(), list_size, validity, row_count)?;

    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage.into_array()).into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar_fn::ScalarFn;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::fixed_shape::FixedShapeTensor;
    use crate::fixed_shape::FixedShapeTensorMetadata;
    use crate::scalar_fns::ApproxOptions;
    use crate::scalar_fns::l2_denorm::L2Denorm;
    use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::constant_tensor_array;
    use crate::utils::test_helpers::constant_vector_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Evaluates L2 denorm on a tensor/vector array and returns the executed array.
    fn eval_l2_denorm(normalized: ArrayRef, norms: ArrayRef, len: usize) -> VortexResult<ArrayRef> {
        let scalar_fn = ScalarFn::new(L2Denorm, ApproxOptions::Exact).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![normalized, norms], len)?;
        let mut ctx = SESSION.create_execution_ctx();
        result.into_array().execute(&mut ctx)
    }

    fn integer_tensor_array(shape: &[usize], elements: &[i32]) -> VortexResult<ArrayRef> {
        let list_size: u32 = shape.iter().product::<usize>().max(1).try_into().unwrap();
        let row_count = elements.len() / list_size as usize;

        let elems: ArrayRef = Buffer::copy_from(elements).into_array();
        let fsl = FixedSizeListArray::new(elems, list_size, Validity::NonNullable, row_count);

        let metadata = FixedShapeTensorMetadata::new(shape.to_vec());
        let ext_dtype =
            ExtDType::<FixedShapeTensor>::try_new(metadata, fsl.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    fn non_tensor_extension_array() -> VortexResult<ArrayRef> {
        let storage = PrimitiveArray::from_iter([1i32, 2]).into_array();
        let ext_dtype =
            ExtDType::<Date>::try_new(TimeUnit::Days, storage.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    fn tensor_snapshot(array: ArrayRef) -> VortexResult<(DType, Vec<bool>, Vec<f64>)> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext: ExtensionArray = array.execute(&mut ctx)?;
        let validity = (0..ext.len())
            .map(|i| ext.is_valid(i))
            .collect::<VortexResult<Vec<_>>>()?;
        let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
        Ok((
            ext.dtype().clone(),
            validity,
            elements.as_slice::<f64>().to_vec(),
        ))
    }

    fn assert_tensor_arrays_eq(actual: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
        let (actual_dtype, actual_validity, actual_elements) = tensor_snapshot(actual)?;
        let (expected_dtype, expected_validity, expected_elements) = tensor_snapshot(expected)?;

        assert_eq!(actual_dtype, expected_dtype);
        assert_eq!(actual_validity, expected_validity);
        assert_close(&actual_elements, &expected_elements);
        Ok(())
    }

    #[test]
    fn l2_denorm_vectors() -> VortexResult<()> {
        let lhs = vector_array(3, &[0.6, 0.8, 0.0, 0.0, 0.0, 0.0])?;
        let rhs = PrimitiveArray::from_iter([5.0f64, 0.0]).into_array();
        let actual = eval_l2_denorm(lhs, rhs, 2)?;
        let expected = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;

        assert_tensor_arrays_eq(actual, expected)?;
        Ok(())
    }

    #[test]
    fn l2_denorm_fixed_shape_tensors() -> VortexResult<()> {
        let lhs = tensor_array(&[2, 2], &[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0])?;
        let rhs = PrimitiveArray::from_iter([4.0f64, 2.0]).into_array();
        let actual = eval_l2_denorm(lhs, rhs, 2)?;
        let expected = tensor_array(&[2, 2], &[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])?;

        assert_tensor_arrays_eq(actual, expected)?;
        Ok(())
    }

    #[test]
    fn l2_denorm_null_propagation() -> VortexResult<()> {
        let lhs = vector_array(2, &[0.6, 0.8, 1.0, 0.0, 0.0, 0.0])?;
        let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

        let rhs = PrimitiveArray::from_option_iter([Some(5.0f64), Some(2.0), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let actual: ExtensionArray = eval_l2_denorm(lhs, rhs, 3)?.execute(&mut ctx)?;
        let storage: FixedSizeListArray = actual.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

        assert!(actual.is_valid(0)?);
        assert!(!actual.is_valid(1)?);
        assert!(!actual.is_valid(2)?);
        assert_close(&elements.as_slice::<f64>()[..2], &[3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_non_extension_lhs() {
        let lhs = PrimitiveArray::from_iter([1.0f64, 2.0]).into_array();
        let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        let result = L2Denorm::try_new_array(&ApproxOptions::Exact, lhs, rhs, 2);
        assert!(result.is_err());
    }

    #[test]
    fn l2_denorm_rejects_non_tensor_extension_lhs() -> VortexResult<()> {
        let lhs = non_tensor_extension_array()?;
        let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        let result = L2Denorm::try_new_array(&ApproxOptions::Exact, lhs, rhs, 2);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_integer_tensor_lhs() -> VortexResult<()> {
        let lhs = integer_tensor_array(&[2], &[1, 2, 3, 4])?;
        let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        let result = L2Denorm::try_new_array(&ApproxOptions::Exact, lhs, rhs, 2);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_mismatched_rhs_ptype() -> VortexResult<()> {
        let lhs = vector_array(2, &[1.0, 0.0, 0.0, 1.0])?;
        let rhs = PrimitiveArray::from_iter([1.0f32, 1.0]).into_array();

        let result = L2Denorm::try_new_array(&ApproxOptions::Exact, lhs, rhs, 2);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_roundtrips_vectors() -> VortexResult<()> {
        let input = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(&ApproxOptions::Exact, input.clone(), &mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_roundtrips_fixed_shape_tensors() -> VortexResult<()> {
        let input = tensor_array(&[2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(&ApproxOptions::Exact, input.clone(), &mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_supports_constant_tensors() -> VortexResult<()> {
        let input = constant_tensor_array(&[2], &[3.0, 4.0], 3)?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(&ApproxOptions::Exact, input.clone(), &mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_supports_constant_vectors() -> VortexResult<()> {
        let input = constant_vector_array(&[3.0, 4.0], 2)?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(&ApproxOptions::Exact, input.clone(), &mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_uses_zero_rows_for_zero_norms() -> VortexResult<()> {
        let input = vector_array(2, &[0.0, 0.0, 3.0, 4.0])?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(&ApproxOptions::Exact, input.clone(), &mut ctx)?;
        let normalized: ExtensionArray = roundtrip.child_at(0).clone().execute(&mut ctx)?;
        let storage: FixedSizeListArray = normalized.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_close(&elements.as_slice::<f64>()[..2], &[0.0, 0.0]);
        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }
}
