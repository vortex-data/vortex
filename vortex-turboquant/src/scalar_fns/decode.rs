// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant decode scalar function.

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_tensor::vector::Vector;

use crate::centroids::compute_or_get_centroids;
use crate::sorf::SorfMatrix;
use crate::vector::storage::parse_storage;
use crate::vector::tq_padded_dim;
use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_metadata;

/// Lazy TurboQuant vector decode scalar function.
#[derive(Clone)]
pub struct TQDecode;

impl TQDecode {
    /// Creates a new [`TypedScalarFnInstance`] wrapping TurboQuant decoding.
    pub fn new() -> TypedScalarFnInstance<TQDecode> {
        TypedScalarFnInstance::new(TQDecode, EmptyMetadata)
    }

    /// Constructs a [`ScalarFnArray`] that lazily decodes a `TurboQuant` child into a `Vector`.
    pub fn try_new_array(child: ArrayRef) -> VortexResult<ScalarFnArray> {
        let len = child.len();
        ScalarFnArray::try_new(TQDecode::new().erased(), vec![child], len)
    }
}

impl ScalarFnVTable for TQDecode {
    type Options = EmptyMetadata;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.turboquant.decode")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(
            metadata.is_empty(),
            "TQDecode options metadata must be empty"
        );

        Ok(EmptyMetadata)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("turboquant"),
            _ => unreachable!("TQDecode must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "tq_decode(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        let metadata = tq_metadata(child_dtype)?;

        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(
                metadata.element_ptype,
                Nullability::NonNullable,
            )),
            metadata.dimensions,
            child_dtype.nullability(),
        );
        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?.erased();

        Ok(DType::Extension(ext_dtype))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        decode_vector(args.get(0)?, ctx)
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

/// Decode a `TurboQuant` extension array back into a `Vector` extension array.
///
/// The decoded directions are inverse-transformed, truncated to the original dimension, and
/// multiplied by the stored row norms. The conversion is lossy and does not roundtrip with
/// [`TQEncode`](crate::TQEncode).
pub(crate) fn decode_vector(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let parsed = parse_storage(input, ctx)?;
    let metadata = parsed.metadata;
    if parsed.len == 0 {
        return build_empty_vector(metadata, parsed.vector_validity);
    }

    let padded_dim = tq_padded_dim(metadata.dimensions)?;
    let transform = SorfMatrix::try_new(padded_dim, metadata.num_rounds as usize, metadata.seed)?;
    let padded_dim = u32::try_from(padded_dim)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;

    let centroids = compute_or_get_centroids(padded_dim, metadata.bit_width)?;

    match_each_float_ptype!(metadata.element_ptype, |T| {
        decode_typed::<T>(
            DecodeInputs {
                metadata: &metadata,
                sorf_matrix: &transform,
                centroids: &centroids,
                norms: &parsed.norms,
                codes: &parsed.codes,
            },
            parsed.vector_validity,
            parsed.len,
            ctx,
        )
    })
}

fn build_empty_vector(
    metadata: TurboQuantMetadata,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    match_each_float_ptype!(metadata.element_ptype, |T| {
        let elements = PrimitiveArray::empty::<T>(Nullability::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            metadata.dimensions,
            vector_validity,
            0,
        )?;

        Vector::try_new_vector_array(fsl.into_array())
    })
}

/// Borrowed bundle of the per-array decode inputs passed to the typed inner loop.
///
/// Packaged as a struct rather than positional arguments because `decode_typed` runs through
/// [`vortex_array::match_each_float_ptype!`] which expands once per supported element ptype.
/// Each expansion takes the same set of inputs, and the struct keeps the call site short.
struct DecodeInputs<'a> {
    /// TurboQuant metadata recovered from the input extension dtype.
    metadata: &'a TurboQuantMetadata,
    /// SORF transform reconstructed from `metadata.seed` and `metadata.num_rounds`.
    sorf_matrix: &'a SorfMatrix,
    /// Centroid codebook for `(padded_dim, bit_width)`, in f32.
    centroids: &'a [f32],
    /// Per-row stored L2 norm of the original input vector, in the element ptype.
    norms: &'a PrimitiveArray,
    /// Flat per-row centroid indices, `num_vectors * padded_dim` bytes.
    codes: &'a PrimitiveArray,
}

fn decode_typed<T>(
    decode: DecodeInputs<'_>,
    vector_validity: Validity,
    num_vectors: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + Float + FromPrimitive,
{
    let metadata = decode.metadata;
    let dimensions = usize::try_from(metadata.dimensions)
        .vortex_expect("dimensions stays representable as usize");
    let padded_dim = decode.sorf_matrix.padded_dim();
    let centroids = decode.centroids;
    let norms = decode.norms.as_slice::<T>();
    let codes = decode.codes.as_slice::<u8>();
    let mask = vector_validity.execute_mask(num_vectors, ctx)?;

    let output_len = num_vectors
        .checked_mul(dimensions)
        .ok_or_else(|| vortex_err!("TurboQuant decoded vector length overflow"))?;
    let mut output = BufferMut::<T>::with_capacity(output_len);

    let mut decoded = vec![0.0f32; padded_dim];
    let mut inverse = vec![0.0f32; padded_dim];

    let mut decode_row = |output: &mut BufferMut<T>, i: usize| {
        let code_row = &codes[i * padded_dim..][..padded_dim];

        for (dst, &code) in decoded.iter_mut().zip(code_row.iter()) {
            *dst = *centroids
                .get(usize::from(code))
                .vortex_expect("TurboQuant code exceeds centroid count");
        }

        decode.sorf_matrix.inverse_transform(&decoded, &mut inverse);

        let norm = norms[i];
        for &value in inverse.iter().take(dimensions) {
            // `T::from_f32` is infallible for the supported float ptypes (`f16`, `f32`,
            // `f64`): values outside `f16` range saturate to `±inf` rather than returning
            // `None`.
            let value = T::from_f32(value)
                .vortex_expect("from_f32 is infallible for supported float types");

            // SAFETY: total pushes across all match arms equal `output_len`.
            unsafe { output.push_unchecked(value * norm) };
        }
    };

    match &mask {
        Mask::AllFalse(_) => {
            // SAFETY: `output` was allocated with capacity `output_len`, and this push writes
            // exactly `output_len` zero placeholders.
            unsafe { output.push_n_unchecked(T::zero(), output_len) };
        }
        Mask::AllTrue(_) => {
            for i in 0..num_vectors {
                decode_row(&mut output, i);
            }
        }
        Mask::Values(values_mask) => {
            let mut cursor = 0;

            for &(start, end) in values_mask.slices() {
                if start > cursor {
                    // SAFETY: total pushes across all arms equal `output_len`.
                    unsafe { output.push_n_unchecked(T::zero(), (start - cursor) * dimensions) };
                }

                for i in start..end {
                    decode_row(&mut output, i);
                }

                cursor = end;
            }

            if cursor < num_vectors {
                // SAFETY: total pushes across all arms equal `output_len`.
                unsafe { output.push_n_unchecked(T::zero(), (num_vectors - cursor) * dimensions) };
            }
        }
    }

    let elements = PrimitiveArray::new::<T>(output.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        metadata.dimensions,
        vector_validity,
        num_vectors,
    )?;

    Vector::try_new_vector_array(fsl.into_array())
}
