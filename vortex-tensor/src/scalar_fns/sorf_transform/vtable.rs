// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ScalarFnVTable`] implementation for [`SorfTransform`].

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::SorfOptions;
use super::SorfTransform;
use super::rotation::SorfMatrix;
use super::validate_sorf_options;
use crate::vector::AnyVector;
use crate::vector::Vector;

impl ScalarFnVTable for SorfTransform {
    type Options = SorfOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.tensor.sorf_transform")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("rotated"),
            _ => unreachable!("SorfTransform must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "sorf_transform(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        validate_sorf_options(options)?;

        let child_dtype = &arg_dtypes[0];
        let vector_metadata = child_dtype
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<AnyVector>())
            .ok_or_else(|| {
                vortex_err!("SorfTransform child must be a Vector extension, got {child_dtype}")
            })?;

        let expected_padded = options.dimension.next_power_of_two();
        vortex_ensure_eq!(
            vector_metadata.dimensions(),
            expected_padded,
            "SorfTransform child Vector must have dimension {expected_padded} (next power of two \
             for dimension {})",
            options.dimension,
        );

        // For now, the child Vector storage must be f32. TurboQuant stores its centroids as f32,
        // and the SORF transform itself operates in f32, so any other input type would require an
        // implicit cast that we do not yet support. The output element type is independently
        // specified via `options.element_ptype` and is built below.
        vortex_ensure_eq!(
            vector_metadata.element_ptype(),
            PType::F32,
            "SorfTransform child Vector storage must be f32 (for now), got {}",
            vector_metadata.element_ptype(),
        );

        let output_elem_dtype = DType::Primitive(options.element_ptype, Nullability::NonNullable);
        let storage_dtype = DType::FixedSizeList(
            Arc::new(output_elem_dtype),
            options.dimension,
            child_dtype.nullability(),
        );

        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?.erased();
        Ok(DType::Extension(ext_dtype))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        validate_sorf_options(options)?;
        let dim = options.dimension as usize;
        let num_rows = args.row_count();

        if num_rows == 0 {
            let child_nullability = args.get(0)?.dtype().nullability();
            let validity = Validity::from(child_nullability);

            return match_each_float_ptype!(options.element_ptype, |T| {
                let elements = PrimitiveArray::empty::<T>(Nullability::NonNullable);
                let fsl = FixedSizeListArray::try_new(
                    elements.into_array(),
                    options.dimension,
                    validity,
                    0,
                )?;
                let ext_dtype =
                    ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
                Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
            });
        }

        // Execute the child to get the Vector extension wrapping an FSL of f32 coordinates. The
        // `return_dtype` check guarantees the child is a `Vector<padded_dim, f32>`, so the
        // materialized FSL elements are always f32.
        let child_ext: ExtensionArray = args.get(0)?.execute(ctx)?;
        let child_validity = child_ext.as_ref().validity()?;
        let child_fsl: FixedSizeListArray = child_ext.storage_array().clone().execute(ctx)?;
        let padded_dim =
            usize::try_from(child_fsl.list_size()).vortex_expect("list_size fits usize");

        let elements_prim: PrimitiveArray = child_fsl.elements().clone().execute(ctx)?;
        let f32_elements = elements_prim.into_buffer::<f32>();

        // Reconstruct the orthogonal transform matrix from the seed.
        let rotation = SorfMatrix::try_new(options.seed, dim, options.num_rounds as usize)?;

        // Inverse transform each row, truncate to original dimension, cast to target type.
        match_each_float_ptype!(options.element_ptype, |T| {
            inverse_rotate_typed::<T>(
                &f32_elements,
                &rotation,
                dim,
                padded_dim,
                num_rows,
                child_validity,
            )
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Convert an f32 value to a float type `T`.
///
/// `FromPrimitive::from_f32` is infallible for all Vortex float types: f16 saturates via the
/// inherent `f16::from_f32()`, f32 is identity, f64 is lossless widening.
fn float_from_f32<T: Float + FromPrimitive>(v: f32) -> T {
    FromPrimitive::from_f32(v).vortex_expect("f32-to-float conversion is infallible")
}

/// Apply the inverse SORF transform on f32 data, truncate to the original dimension, cast each
/// element to `T`, and build the output [`Vector`] extension array.
fn inverse_rotate_typed<T: NativePType + Float + FromPrimitive>(
    f32_elements: &[f32],
    rotation: &SorfMatrix,
    dim: usize,
    padded_dim: usize,
    num_rows: usize,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let dim_u32 = u32::try_from(dim).vortex_expect("dimension fits u32");
    let mut output = BufferMut::<T>::with_capacity(num_rows * dim);
    let mut unrotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let row_data = &f32_elements[row * padded_dim..(row + 1) * padded_dim];

        rotation.inverse_rotate(row_data, &mut unrotated);

        for idx in 0..dim {
            // SAFETY: We allocated enough memory above.
            unsafe { output.push_unchecked(float_from_f32::<T>(unrotated[idx])) };
        }
    }

    let elements = PrimitiveArray::new::<T>(output.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(elements.into_array(), dim_u32, validity, num_rows)?;

    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}
