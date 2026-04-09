// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SORF (Structured Orthogonal Random Features) inverse-rotation scalar function.
//!
//! Wraps an FSL child (e.g. `FSL(Dict(codes, centroids))`) and applies the inverse
//! Walsh-Hadamard-based structured rotation at execution time, producing a [`Vector`]
//! extension array with the original (pre-padding) dimensionality.
//!
//! The rotation parameters are stored as a deterministic PRNG seed in [`SorfOptions`],
//! so the `RotationMatrix` is reconstructed cheaply at decode time.
//!
//! [`Vector`]: crate::vector::Vector

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
use vortex_array::arrays::ScalarFnArray;
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
use vortex_array::scalar_fn::ScalarFn;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::encodings::turboquant::rotation::RotationMatrix;
use crate::vector::Vector;

/// Options for the SORF inverse-rotation scalar function.
///
/// Stored in the [`ScalarFnArray`] and used to deterministically reconstruct the
/// `RotationMatrix` at decode time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SorfOptions {
    /// PRNG seed used to generate the random sign diagonals.
    pub seed: u64,
    /// Number of sign-diagonal + WHT rounds in the structured rotation.
    pub num_rounds: u8,
    /// Original vector dimension (before power-of-2 padding).
    pub dimension: u32,
    /// Target output element type (e.g. `F16`, `F32`, `F64`).
    pub element_ptype: PType,
}

impl fmt::Display for SorfOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SorfOptions(seed={}, rounds={}, dim={}, ptype={})",
            self.seed, self.num_rounds, self.dimension, self.element_ptype
        )
    }
}

/// Inverse SORF rotation scalar function.
///
/// Applies the inverse structured Walsh-Hadamard rotation to an FSL child, truncates
/// from padded dimension to the original dimension, casts to the target element type,
/// and wraps in a [`Vector`] extension array.
#[derive(Clone)]
pub struct SorfTransform;

impl SorfTransform {
    /// Creates a new [`ScalarFn`] wrapping the SORF inverse rotation with the given options.
    pub fn new(options: &SorfOptions) -> ScalarFn<SorfTransform> {
        ScalarFn::new(SorfTransform, options.clone())
    }

    /// Constructs a validated [`ScalarFnArray`] that lazily applies inverse SORF rotation.
    ///
    /// The `child` must be a `FixedSizeList` (or array that executes to one) with
    /// `list_size == padded_dim` (i.e. `dimension.next_power_of_two()`).
    pub fn try_new_array(
        options: &SorfOptions,
        child: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(SorfTransform::new(options).erased(), vec![child], len)
    }
}

/// Convert an f32 value to a float type `T`.
///
/// `FromPrimitive::from_f32` is infallible for all Vortex float types: f16 saturates via the
/// inherent `f16::from_f32()`, f32 is identity, f64 is lossless widening.
fn float_from_f32<T: Float + FromPrimitive>(v: f32) -> T {
    FromPrimitive::from_f32(v).vortex_expect("f32-to-float conversion is infallible")
}

impl ScalarFnVTable for SorfTransform {
    type Options = SorfOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.tensor.sorf_transform")
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
        let child_dtype = &arg_dtypes[0];

        let DType::FixedSizeList(elem_dtype, padded_dim, nullability) = child_dtype else {
            vortex_bail!("SorfTransform child must be a FixedSizeList, got {child_dtype}");
        };

        let expected_padded = options.dimension.next_power_of_two();
        vortex_ensure!(
            *padded_dim == expected_padded,
            "SorfTransform child list_size must be {expected_padded} (padded_dim), got {padded_dim}"
        );

        // The child elements can be any type that executes to f32 (e.g. Dict<u8, f32>).
        vortex_ensure!(
            !elem_dtype.is_extension(),
            "SorfTransform child element dtype must not be an extension type, got {elem_dtype}"
        );

        // Build the output Vector extension dtype: Ext<Vector, FSL<element_ptype, dimension>>.
        let output_elem_dtype = DType::Primitive(options.element_ptype, Nullability::NonNullable);
        let storage_dtype =
            DType::FixedSizeList(Arc::new(output_elem_dtype), options.dimension, *nullability);

        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?.erased();
        Ok(DType::Extension(ext_dtype))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let dim = options.dimension as usize;
        let num_rows = args.row_count();

        if num_rows == 0 {
            return match_each_float_ptype!(options.element_ptype, |T| {
                let elements = PrimitiveArray::empty::<T>(Nullability::NonNullable);
                let fsl = FixedSizeListArray::try_new(
                    elements.into_array(),
                    options.dimension,
                    Validity::NonNullable,
                    0,
                )?;
                let ext_dtype =
                    ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
                Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
            });
        }

        // Execute the child to get the FSL of dequantized (or raw) f32 coordinates.
        let child_fsl: FixedSizeListArray = args.get(0)?.execute(ctx)?;
        let padded_dim =
            usize::try_from(child_fsl.list_size()).vortex_expect("list_size fits usize");

        // Get flat f32 elements from the executed FSL.
        let elements_prim: PrimitiveArray = child_fsl.elements().clone().execute(ctx)?;
        let f32_elements = to_f32_slice(&elements_prim)?;

        // Reconstruct the rotation matrix from the seed.
        let rotation = RotationMatrix::try_new(options.seed, dim, options.num_rounds as usize)?;

        // Inverse rotate each row, truncate to original dimension, cast to target type.
        match_each_float_ptype!(options.element_ptype, |T| {
            inverse_rotate_typed::<T>(&f32_elements, &rotation, dim, padded_dim, num_rows)
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

/// Convert executed primitive elements to an owned `Vec<f32>`.
///
/// All rotation happens in f32. f16 is upcast; f64 is truncated to f32 precision.
fn to_f32_slice(prim: &PrimitiveArray) -> VortexResult<Vec<f32>> {
    match prim.ptype() {
        PType::F16 => Ok(prim
            .as_slice::<half::f16>()
            .iter()
            .map(|&v| f32::from(v))
            .collect()),
        PType::F32 => Ok(prim.as_slice::<f32>().to_vec()),
        PType::F64 => Ok(prim
            .as_slice::<f64>()
            .iter()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "intentional f64 -> f32 truncation for rotation"
                )]
                let v = v as f32;
                v
            })
            .collect()),
        other => vortex_bail!("SorfTransform requires float elements, got {other:?}"),
    }
}

/// Inverse rotate + truncate + cast to T, then build the output Vector extension array.
fn inverse_rotate_typed<T: NativePType + Float + FromPrimitive>(
    f32_elements: &[f32],
    rotation: &RotationMatrix,
    dim: usize,
    padded_dim: usize,
    num_rows: usize,
) -> VortexResult<ArrayRef> {
    let dim_u32 = u32::try_from(dim).vortex_expect("dimension fits u32");
    let mut output = BufferMut::<T>::with_capacity(num_rows * dim);
    let mut unrotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let row_data = &f32_elements[row * padded_dim..(row + 1) * padded_dim];

        rotation.inverse_rotate(row_data, &mut unrotated);

        for idx in 0..dim {
            output.push(float_from_f32::<T>(unrotated[idx]));
        }
    }

    let elements = PrimitiveArray::new::<T>(output.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        dim_u32,
        Validity::NonNullable,
        num_rows,
    )?;

    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}
