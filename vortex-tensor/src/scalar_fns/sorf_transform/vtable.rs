// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ScalarFnVTable`] implementation for [`SorfTransform`].

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use num_traits::Float;
use num_traits::FromPrimitive;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
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
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::SorfOptions;
use super::SorfTransform;
use super::rotation::SorfMatrix;
use super::validate_sorf_options;
use crate::matcher::AnyTensor;
use crate::types::normalized_vector::NormalizedVector;
use crate::types::normalized_vector::inner_vector_array;
use crate::types::vector::AnyVector;
use crate::types::vector::Vector;

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
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "sorf_transform(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", {options})")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        validate_sorf_options(options)?;

        let child_dtype = &arg_dtypes[0];
        let vector_metadata = child_dtype
            .as_extension_opt()
            .and_then(|ext| ext.metadata_opt::<AnyTensor>())
            .ok_or_else(|| {
                vortex_err!(
                    "SorfTransform child must be a Vector or NormalizedVector extension, got \
                     {child_dtype}"
                )
            })?;

        let expected_padded = options.dimensions.next_power_of_two();
        vortex_ensure_eq!(
            vector_metadata.list_size(),
            expected_padded,
            "SorfTransform child Vector must have dimension {expected_padded} (next power of two \
             for dimension {})",
            options.dimensions,
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
        let fsl_dtype = DType::FixedSizeList(
            Arc::new(output_elem_dtype),
            options.dimensions,
            child_dtype.nullability(),
        );

        // The output mirrors the child's wrapper kind: if the child was a `NormalizedVector` the
        // output is also surfaced as a `NormalizedVector` (the orthogonal inverse rotation
        // preserves L2 norm and the truncation drops coordinates that were zero pre-rotation, so
        // the output is approximately unit-norm under the same lossy contract that
        // `NormalizedVector::new_unchecked` documents).
        if vector_metadata.is_normalized() {
            let inner_vector = ExtDType::<Vector>::try_new(EmptyMetadata, fsl_dtype)?.erased();
            let outer = ExtDType::<NormalizedVector>::try_new(
                EmptyMetadata,
                DType::Extension(inner_vector),
            )?
            .erased();
            Ok(DType::Extension(outer))
        } else {
            let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl_dtype)?.erased();
            Ok(DType::Extension(ext_dtype))
        }
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let dim = options.dimensions as usize;
        let num_rows = args.row_count();

        let child_arg = args.get(0)?;
        let is_normalized_child = child_arg
            .dtype()
            .as_extension_opt()
            .is_some_and(|ext| ext.is::<NormalizedVector>());

        let fsl_array: ArrayRef = if num_rows == 0 {
            let validity = Validity::from(child_arg.dtype().nullability());

            match_each_float_ptype!(options.element_ptype, |T| {
                let elements = PrimitiveArray::empty::<T>(Nullability::NonNullable);
                FixedSizeListArray::try_new(elements.into_array(), options.dimensions, validity, 0)
            })?
            .into_array()
        } else {
            // Execute the child to get either a `Vector` extension or a `NormalizedVector`
            // wrapping a `Vector` over an FSL of f32 coordinates. The `return_dtype` check
            // guarantees the shape is `Vector<padded_dim, f32>` at the FSL level, so drill past
            // any `NormalizedVector` wrapper before unpacking.
            let child_ref = inner_vector_array(&child_arg, ctx)?;
            let child_ext: ExtensionArray = child_ref.execute(ctx)?;
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
            })?
        };

        // SAFETY: When `is_normalized_child` is `true`, the input child was a
        // `NormalizedVector` (its dtype was checked above), so every valid row was unit-norm or
        // zero by type. Inverse SORF is orthogonal (norm-preserving), and the truncated tail
        // coordinates were zero pre-rotation up to quantization noise — so each row of
        // `fsl_array` is approximately unit-norm under the same lossy contract that
        // [`NormalizedVector::new_unchecked`] documents. When `is_normalized_child` is `false`
        // [`wrap_output`] takes the trivially-safe `Vector` branch.
        unsafe { wrap_output(fsl_array, is_normalized_child) }
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

/// Metadata for a serialized [`SorfTransform`] array.
///
/// Stores the full [`SorfOptions`] inline. The child dtype is fully derivable from the parent
/// dtype: the parent's outer wrapper (plain `Vector` or `NormalizedVector`) mirrors the child's
/// wrapper kind, the inner FSL nullability is propagated through `return_dtype`, and
/// `padded_dim`/`f32` are determined by [`SorfOptions`].
#[derive(Clone, prost::Message)]
pub(super) struct SorfTransformMetadata {
    #[prost(uint64, tag = "1")]
    seed: u64,
    /// Rust `u8` widened to `u32` for protobuf (no `u8` on the wire).
    #[prost(uint32, tag = "2")]
    num_rounds: u32,
    #[prost(uint32, tag = "3")]
    dimension: u32,
    #[prost(enumeration = "PType", tag = "4")]
    element_ptype: i32,
}

impl ScalarFnArrayVTable for SorfTransform {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let metadata = SorfTransformMetadata::from(view.options);
        Ok(Some(metadata.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let _ = session;
        let metadata = SorfTransformMetadata::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode SorfTransformMetadata: {e}"))?;
        let options = metadata.to_options()?;

        // The parent dtype must be a vector-shaped extension produced by `return_dtype`: either
        // a plain `Vector` (when the child was a plain `Vector`) or a `NormalizedVector` (when
        // the child was a `NormalizedVector`). `AnyVector` matches both, and its `try_match`
        // panics on a structurally malformed `NormalizedVector`, so a successful match also
        // guarantees the inner drill below is well-formed.
        let parent_ext = dtype
            .as_extension_opt()
            .filter(|ext| ext.is::<AnyVector>())
            .ok_or_else(|| {
                vortex_err!(
                    "SorfTransform parent dtype must be a `Vector` or `NormalizedVector` \
                     extension, got {dtype}",
                )
            })?;
        let is_normalized = parent_ext.is::<NormalizedVector>();

        // The child's FSL nullability matches the parent's inner FSL nullability (set by
        // `return_dtype` from the original child's outer nullability). Drill into the parent
        // wrapper to recover it; `AnyVector` already validated the structural shape.
        let parent_fsl_dtype = if is_normalized {
            let DType::Extension(inner) = parent_ext.storage_dtype() else {
                unreachable!(
                    "`AnyVector` matcher guarantees a `NormalizedVector` parent wraps a \
                     `Vector` extension"
                )
            };
            inner.storage_dtype()
        } else {
            parent_ext.storage_dtype()
        };
        let fsl_nullability = parent_fsl_dtype.nullability();

        let padded_dim = options.dimensions.next_power_of_two();
        let child_fsl = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            padded_dim,
            fsl_nullability,
        );
        let inner_vector = ExtDType::<Vector>::try_new(EmptyMetadata, child_fsl)?.erased();
        let child_dtype = if is_normalized {
            let nv = ExtDType::<NormalizedVector>::try_new(
                EmptyMetadata,
                DType::Extension(inner_vector),
            )?
            .erased();
            DType::Extension(nv)
        } else {
            DType::Extension(inner_vector)
        };
        let child = children.get(0, &child_dtype, len)?;

        Ok(ScalarFnArrayParts {
            options,
            children: vec![child],
        })
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
/// element to `T`, and return the resulting `FixedSizeList` storage array. The caller is
/// responsible for wrapping the FSL in the appropriate vector-family extension via
/// [`wrap_output`].
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
    Ok(fsl.into_array())
}

/// Wraps `fsl` as either a [`Vector`] or [`NormalizedVector`] extension array, mirroring the kind
/// of the upstream `SorfTransform` child.
///
/// # Safety
///
/// When `is_normalized` is `true`, every valid row of `fsl` must be approximately unit-norm or
/// zero in the lossy sense documented by [`NormalizedVector::new_unchecked`].
///
/// When `is_normalized` is `false` the function takes the safe `Vector` branch.
unsafe fn wrap_output(fsl: ArrayRef, is_normalized: bool) -> VortexResult<ArrayRef> {
    if is_normalized {
        // SAFETY: Forwarded from the function-level safety contract above.
        unsafe { NormalizedVector::new_unchecked(fsl) }
    } else {
        Vector::try_new_vector_array(fsl)
    }
}

impl From<&SorfOptions> for SorfTransformMetadata {
    fn from(options: &SorfOptions) -> Self {
        Self {
            seed: options.seed,
            num_rounds: u32::from(options.num_rounds),
            dimension: options.dimensions,
            element_ptype: options.element_ptype as i32,
        }
    }
}

impl SorfTransformMetadata {
    /// Rebuild the [`SorfOptions`] this metadata was serialized from, validating that the wire
    /// values are in range.
    fn to_options(&self) -> VortexResult<SorfOptions> {
        let num_rounds = u8::try_from(self.num_rounds).map_err(|_| {
            vortex_err!(
                "SorfTransform num_rounds {} does not fit in u8",
                self.num_rounds
            )
        })?;
        let options = SorfOptions {
            seed: self.seed,
            num_rounds,
            dimensions: self.dimension,
            element_ptype: self.element_ptype(),
        };
        validate_sorf_options(&options)?;
        Ok(options)
    }
}
