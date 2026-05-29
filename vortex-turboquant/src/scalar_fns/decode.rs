// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant decode scalar function.
//!
//! Reverses the per-row, per-block encode pipeline in [`crate::vector::quantize`]. For each
//! block `i` of each valid input row, the decoder:
//!
//! 1. Reads that block's per-row codes and gathers the matching centroid values from a
//!    `2^bit_width`-entry centroid table built for width `block_sizes[i]`.
//! 2. Applies the inverse SORF of width `block_sizes[i]` seeded with the same
//!    `derive_block_seed(metadata.seed, i)` the encoder used.
//! 3. Multiplies the rotated coordinates by the per-row block norm stored in that block's
//!    `norms` column.
//! 4. Writes the result into a row-aligned scratch buffer of width `sum(block_sizes)` at offsets
//!    `[offset_i .. offset_i + block_sizes[i])`, the same offsets the encoder sliced from.
//!
//! Once every block is reconstructed for a row, the first `dimensions` coordinates of the
//! scratch buffer are copied into the output `Vector`, dropping any overspilling coordinates
//! the encoder zero-padded.

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
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
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_tensor::vector::Vector;

use crate::centroids::compute_or_get_codebook;
use crate::sorf::splitmix64::derive_block_seed;
use crate::sorf::transform::SorfMatrix;
use crate::vector::dequantize::DecodeInputs;
use crate::vector::dequantize::decode_typed;
use crate::vector::storage::parse_storage;
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
/// Decodes each block by looking up centroid values from per-block codes, applying the inverse
/// SORF transform, and scaling by the stored per-row norm.
///
/// Results are assembled into a scratch buffer of width `sum(block_sizes)`, then truncated to the
/// first `dimensions` coordinates to produce the output `Vector`.
pub(crate) fn decode_vector(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let parsed = parse_storage(input, ctx)?;
    if parsed.len == 0 {
        return build_empty_vector(parsed.metadata, parsed.vector_validity);
    }

    let metadata = parsed.metadata;
    let block_sizes: Vec<usize> = metadata
        .block_sizes
        .iter()
        .map(|&b| {
            usize::try_from(b).map_err(|_| vortex_err!("TurboQuant block {b} does not fit usize"))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let total_width: usize = block_sizes.iter().sum();

    let mut transforms = Vec::with_capacity(block_sizes.len());
    let mut centroids = Vec::with_capacity(block_sizes.len());

    for (index, (&block, &block_u32)) in block_sizes
        .iter()
        .zip(metadata.block_sizes.iter())
        .enumerate()
    {
        let seed_i = derive_block_seed(metadata.seed, index);

        transforms.push(SorfMatrix::try_new(
            block,
            metadata.num_rounds as usize,
            seed_i,
        )?);
        centroids.push(
            compute_or_get_codebook(block_u32, metadata.bit_width)?
                .centroids
                .clone(),
        );
    }

    match_each_float_ptype!(metadata.element_ptype, |T| {
        decode_typed::<T>(
            DecodeInputs {
                metadata: &metadata,
                block_sizes: &block_sizes,
                total_width,
                sorf_matrices: &transforms,
                centroid_tables: &centroids,
                block_storages: &parsed.blocks,
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
