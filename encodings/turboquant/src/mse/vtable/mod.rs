// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VTable implementation for TurboQuant MSE encoding.

use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use super::TurboQuantMSE;
use super::array::TurboQuantMSEArray;
use super::array::TurboQuantMSEMetadata;
use crate::decompress::execute_decompress_mse;

impl VTable for TurboQuantMSE {
    type Array = TurboQuantMSEArray;
    type Metadata = ProstMetadata<TurboQuantMSEMetadata>;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &TurboQuantMSE
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &TurboQuantMSEArray) -> usize {
        array.norms.len()
    }

    fn dtype(array: &TurboQuantMSEArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &TurboQuantMSEArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &TurboQuantMSEArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.dimension.hash(state);
        array.bit_width.hash(state);
        array.padded_dim.hash(state);
        array.rotation_seed.hash(state);
        array.codes.array_hash(state, precision);
        array.norms.array_hash(state, precision);
        array.centroids.array_hash(state, precision);
        array.rotation_signs.array_hash(state, precision);
    }

    fn array_eq(
        array: &TurboQuantMSEArray,
        other: &TurboQuantMSEArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array.dimension == other.dimension
            && array.bit_width == other.bit_width
            && array.padded_dim == other.padded_dim
            && array.rotation_seed == other.rotation_seed
            && array.codes.array_eq(&other.codes, precision)
            && array.norms.array_eq(&other.norms, precision)
            && array.centroids.array_eq(&other.centroids, precision)
            && array
                .rotation_signs
                .array_eq(&other.rotation_signs, precision)
    }

    fn nbuffers(_array: &TurboQuantMSEArray) -> usize {
        0
    }

    fn buffer(_array: &TurboQuantMSEArray, idx: usize) -> BufferHandle {
        vortex_panic!("TurboQuantMSEArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &TurboQuantMSEArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &TurboQuantMSEArray) -> usize {
        4
    }

    fn child(array: &TurboQuantMSEArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.codes.clone(),
            1 => array.norms.clone(),
            2 => array.centroids.clone(),
            3 => array.rotation_signs.clone(),
            _ => vortex_panic!("TurboQuantMSEArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &TurboQuantMSEArray, idx: usize) -> String {
        match idx {
            0 => "codes".to_string(),
            1 => "norms".to_string(),
            2 => "centroids".to_string(),
            3 => "rotation_signs".to_string(),
            _ => vortex_panic!("TurboQuantMSEArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &TurboQuantMSEArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(TurboQuantMSEMetadata {
            dimension: array.dimension,
            bit_width: array.bit_width as u32,
            padded_dim: array.padded_dim,
            rotation_seed: array.rotation_seed,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<TurboQuantMSEMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<TurboQuantMSEArray> {
        let bit_width = u8::try_from(metadata.bit_width)?;
        let padded_dim = metadata.padded_dim as usize;
        let num_centroids = 1usize << bit_width;

        let codes_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let codes = children.get(0, &codes_dtype, len * padded_dim)?;

        let norms_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let norms = children.get(1, &norms_dtype, len)?;

        let centroids = children.get(2, &norms_dtype, num_centroids)?;

        let signs_dtype = DType::Bool(Nullability::NonNullable);
        let rotation_signs = children.get(3, &signs_dtype, 3 * padded_dim)?;

        Ok(TurboQuantMSEArray {
            dtype: dtype.clone(),
            codes,
            norms,
            centroids,
            rotation_signs,
            dimension: metadata.dimension,
            bit_width,
            padded_dim: metadata.padded_dim,
            rotation_seed: metadata.rotation_seed,
            stats_set: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 4,
            "TurboQuantMSEArray expects 4 children, got {}",
            children.len()
        );
        let mut iter = children.into_iter();
        array.codes = iter.next().vortex_expect("codes child");
        array.norms = iter.next().vortex_expect("norms child");
        array.centroids = iter.next().vortex_expect("centroids child");
        array.rotation_signs = iter.next().vortex_expect("rotation_signs child");
        Ok(())
    }

    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = Arc::try_unwrap(array).unwrap_or_else(|arc| (*arc).clone());
        Ok(ExecutionResult::done(execute_decompress_mse(array, ctx)?))
    }
}

impl ValidityChild<TurboQuantMSE> for TurboQuantMSE {
    fn validity_child(array: &TurboQuantMSEArray) -> &ArrayRef {
        array.codes()
    }
}
