// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VTable implementation for TurboQuant encoding.

use std::hash::Hash;
use std::ops::Deref;
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
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::array::QjlCorrection;
use crate::array::TurboQuant;
use crate::array::TurboQuantArray;
use crate::array::TurboQuantMetadata;
use crate::decompress::execute_decompress;

const MSE_CHILDREN: usize = 4;
const QJL_CHILDREN: usize = 3;

impl VTable for TurboQuant {
    type Array = TurboQuantArray;
    type Metadata = ProstMetadata<TurboQuantMetadata>;
    type OperationsVTable = TurboQuant;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &TurboQuant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &TurboQuantArray) -> usize {
        array.norms.len()
    }

    fn dtype(array: &TurboQuantArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &TurboQuantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &TurboQuantArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.dimension.hash(state);
        array.bit_width.hash(state);
        array.has_qjl().hash(state);
        array.codes.array_hash(state, precision);
        array.norms.array_hash(state, precision);
        array.centroids.array_hash(state, precision);
        array.rotation_signs.array_hash(state, precision);
        if let Some(qjl) = &array.qjl {
            qjl.signs.array_hash(state, precision);
            qjl.residual_norms.array_hash(state, precision);
            qjl.rotation_signs.array_hash(state, precision);
        }
    }

    fn array_eq(array: &TurboQuantArray, other: &TurboQuantArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.dimension == other.dimension
            && array.bit_width == other.bit_width
            && array.has_qjl() == other.has_qjl()
            && array.codes.array_eq(&other.codes, precision)
            && array.norms.array_eq(&other.norms, precision)
            && array.centroids.array_eq(&other.centroids, precision)
            && array
                .rotation_signs
                .array_eq(&other.rotation_signs, precision)
            && match (&array.qjl, &other.qjl) {
                (Some(a), Some(b)) => {
                    a.signs.array_eq(&b.signs, precision)
                        && a.residual_norms.array_eq(&b.residual_norms, precision)
                        && a.rotation_signs.array_eq(&b.rotation_signs, precision)
                }
                (None, None) => true,
                _ => false,
            }
    }

    fn nbuffers(_array: &TurboQuantArray) -> usize {
        0
    }

    fn buffer(_array: &TurboQuantArray, idx: usize) -> BufferHandle {
        vortex_panic!("TurboQuantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &TurboQuantArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(array: &TurboQuantArray) -> usize {
        if array.has_qjl() {
            MSE_CHILDREN + QJL_CHILDREN
        } else {
            MSE_CHILDREN
        }
    }

    fn child(array: &TurboQuantArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.codes.clone(),
            1 => array.norms.clone(),
            2 => array.centroids.clone(),
            3 => array.rotation_signs.clone(),
            4 => array
                .qjl
                .as_ref()
                .vortex_expect("QJL child requested but has_qjl is false")
                .signs
                .clone(),
            5 => array
                .qjl
                .as_ref()
                .vortex_expect("QJL child requested but has_qjl is false")
                .residual_norms
                .clone(),
            6 => array
                .qjl
                .as_ref()
                .vortex_expect("QJL child requested but has_qjl is false")
                .rotation_signs
                .clone(),
            _ => vortex_panic!("TurboQuantArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &TurboQuantArray, idx: usize) -> String {
        match idx {
            0 => "codes".to_string(),
            1 => "norms".to_string(),
            2 => "centroids".to_string(),
            3 => "rotation_signs".to_string(),
            4 => "qjl_signs".to_string(),
            5 => "qjl_residual_norms".to_string(),
            6 => "qjl_rotation_signs".to_string(),
            _ => vortex_panic!("TurboQuantArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &TurboQuantArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(TurboQuantMetadata {
            dimension: array.dimension,
            bit_width: array.bit_width as u32,
            has_qjl: array.has_qjl(),
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
            <ProstMetadata<TurboQuantMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<TurboQuantArray> {
        let bit_width = u8::try_from(metadata.bit_width)?;
        let padded_dim = metadata.dimension.next_power_of_two() as usize;
        let num_centroids = 1usize << bit_width;

        let u8_nn = DType::Primitive(PType::U8, Nullability::NonNullable);
        let f32_nn = DType::Primitive(PType::F32, Nullability::NonNullable);
        let codes_dtype = DType::FixedSizeList(
            Arc::new(u8_nn.clone()),
            padded_dim as u32,
            Nullability::NonNullable,
        );
        let codes = children.get(0, &codes_dtype, len)?;

        let norms = children.get(1, &f32_nn, len)?;
        let centroids = children.get(2, &f32_nn, num_centroids)?;

        let signs_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let rotation_signs = children.get(3, &signs_dtype, 3 * padded_dim)?;

        let qjl = if metadata.has_qjl {
            let qjl_signs_dtype =
                DType::FixedSizeList(Arc::new(u8_nn), padded_dim as u32, Nullability::NonNullable);
            let qjl_signs = children.get(4, &qjl_signs_dtype, len)?;
            let qjl_residual_norms = children.get(5, &f32_nn, len)?;
            let qjl_rotation_signs = children.get(6, &signs_dtype, 3 * padded_dim)?;
            Some(QjlCorrection {
                signs: qjl_signs,
                residual_norms: qjl_residual_norms,
                rotation_signs: qjl_rotation_signs,
            })
        } else {
            None
        };

        Ok(TurboQuantArray {
            dtype: dtype.clone(),
            codes,
            norms,
            centroids,
            rotation_signs,
            qjl,
            dimension: metadata.dimension,
            bit_width,
            stats_set: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        let expected = if array.has_qjl() {
            MSE_CHILDREN + QJL_CHILDREN
        } else {
            MSE_CHILDREN
        };
        vortex_ensure!(
            children.len() == expected,
            "TurboQuantArray expects {expected} children, got {}",
            children.len()
        );
        let mut iter = children.into_iter();
        array.codes = iter.next().vortex_expect("codes child");
        array.norms = iter.next().vortex_expect("norms child");
        array.centroids = iter.next().vortex_expect("centroids child");
        array.rotation_signs = iter.next().vortex_expect("rotation_signs child");
        if let Some(qjl) = &mut array.qjl {
            qjl.signs = iter.next().vortex_expect("qjl_signs child");
            qjl.residual_norms = iter.next().vortex_expect("qjl_residual_norms child");
            qjl.rotation_signs = iter.next().vortex_expect("qjl_rotation_signs child");
        }
        Ok(())
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::compute::rules::RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::compute::rules::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = Arc::try_unwrap(array)
            .map(|a| a.into_inner())
            .unwrap_or_else(|arc| arc.as_ref().deref().clone());
        Ok(ExecutionResult::done(execute_decompress(inner, ctx)?))
    }
}

impl ValidityChild<TurboQuant> for TurboQuant {
    fn validity_child(array: &TurboQuantArray) -> &ArrayRef {
        array.codes()
    }
}
