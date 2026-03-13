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
use vortex::dtype::extension::Matcher;
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
        ScalarFnId::new_ref("vortex.l2_norm")
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
        debug_assert_eq!(arg_dtypes.len(), 1);

        let input_dtype = &arg_dtypes[0];

        // Input must be a tensor-like extension type.
        let ext = input_dtype.as_extension_opt().ok_or_else(|| {
            vortex_err!("l2_norm input must be an extension type, got {input_dtype}")
        })?;

        vortex_ensure!(
            AnyTensor::matches(ext),
            "l2_norm input must be an `AnyTensor`, got {input_dtype}"
        );

        let ptype = extension_element_ptype(ext)?;
        vortex_ensure!(
            ptype.is_float(),
            "l2_norm element dtype must be a float primitive, got {ptype}"
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
        let (elems, stride) = extract_flat_elements(&storage, list_size)?;

        match_each_float_ptype!(elems.ptype(), |T| {
            let slice = elems.as_slice::<T>();

            let result: PrimitiveArray = (0..row_count)
                .map(|i| {
                    let v = &slice[i * stride..i * stride + list_size];
                    l2_norm_row(v)
                })
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
        // Canonicalization of the storage array can fail.
        true
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
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::ToCanonical;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::ScalarFnArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::extension::ExtDType;
    use vortex::error::VortexResult;
    use vortex::extension::EmptyMetadata;
    use vortex::scalar_fn::EmptyOptions;
    use vortex::scalar_fn::ScalarFn;

    use crate::fixed_shape::FixedShapeTensor;
    use crate::fixed_shape::FixedShapeTensorMetadata;
    use crate::scalar_fns::l2_norm::L2Norm;
    use crate::vector::Vector;

    /// Builds a [`FixedShapeTensor`] extension array from flat f64 elements and a logical shape.
    fn tensor_array(shape: &[usize], elements: &[f64]) -> VortexResult<ArrayRef> {
        let list_size: u32 = shape.iter().product::<usize>().max(1).try_into().unwrap();
        let row_count = elements.len() / list_size as usize;

        let elems: ArrayRef = Buffer::copy_from(elements).into_array();
        let fsl = FixedSizeListArray::new(elems, list_size, Validity::NonNullable, row_count);

        let metadata = FixedShapeTensorMetadata::new(shape.to_vec());
        let ext_dtype =
            ExtDType::<FixedShapeTensor>::try_new(metadata, fsl.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    /// Builds a [`Vector`] extension array from flat f64 elements and a vector dimension size.
    fn vector_array(dim: u32, elements: &[f64]) -> VortexResult<ArrayRef> {
        let row_count = elements.len() / dim as usize;

        let elems: ArrayRef = Buffer::copy_from(elements).into_array();
        let fsl = FixedSizeListArray::new(elems, dim, Validity::NonNullable, row_count);

        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();

        Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
    }

    /// Evaluates L2 norm on a tensor/vector array and returns the result as `Vec<f64>`.
    fn eval_l2_norm(input: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = ScalarFn::new(L2Norm, EmptyOptions).erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![input], len)?;
        let prim = result.to_primitive();
        Ok(prim.as_slice::<f64>().to_vec())
    }

    #[track_caller]
    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: got {} elements, expected {}",
            actual.len(),
            expected.len()
        );

        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "element {i}: got {a}, expected {e} (diff = {})",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn unit_vector_norm() -> VortexResult<()> {
        let arr = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // unit x
                0.0, 1.0, 0.0, // unit y
                0.0, 0.0, 1.0, // unit z
            ],
        )?;
        assert_close(&eval_l2_norm(arr, 3)?, &[1.0, 1.0, 1.0]);
        Ok(())
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
    fn vector_known_norm() -> VortexResult<()> {
        let arr = vector_array(2, &[3.0, 4.0])?;
        assert_close(&eval_l2_norm(arr, 1)?, &[5.0]);
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
