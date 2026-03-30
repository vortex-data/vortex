// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VTable implementation for TurboQuant QJL encoding.

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
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::matcher::Matcher;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::Array;
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

use super::TurboQuantQJL;
use super::array::TurboQuantQJLArray;
use super::array::TurboQuantQJLMetadata;
use crate::TurboQuantMSE;
use crate::decompress::execute_decompress_qjl;

impl VTable for TurboQuantQJL {
    type Array = TurboQuantQJLArray;
    type Metadata = ProstMetadata<TurboQuantQJLMetadata>;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &TurboQuantQJL
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &TurboQuantQJLArray) -> usize {
        array.residual_norms.len()
    }

    fn dtype(array: &TurboQuantQJLArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &TurboQuantQJLArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &TurboQuantQJLArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        (*array.mse_inner)
            .clone()
            .into_array()
            .array_hash(state, precision);
        array.qjl_signs.array_hash(state, precision);
        array.residual_norms.array_hash(state, precision);
        array.rotation_signs.array_hash(state, precision);
    }

    fn array_eq(
        array: &TurboQuantQJLArray,
        other: &TurboQuantQJLArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && (*array.mse_inner)
                .clone()
                .into_array()
                .array_eq(&(*other.mse_inner).clone().into_array(), precision)
            && array.qjl_signs.array_eq(&other.qjl_signs, precision)
            && array
                .residual_norms
                .array_eq(&other.residual_norms, precision)
            && array
                .rotation_signs
                .array_eq(&other.rotation_signs, precision)
    }

    fn nbuffers(_array: &TurboQuantQJLArray) -> usize {
        0
    }

    fn buffer(_array: &TurboQuantQJLArray, idx: usize) -> BufferHandle {
        vortex_panic!("TurboQuantQJLArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &TurboQuantQJLArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &TurboQuantQJLArray) -> usize {
        4
    }

    fn child(array: &TurboQuantQJLArray, idx: usize) -> ArrayRef {
        match idx {
            0 => (*array.mse_inner).clone().into_array(),
            1 => array.qjl_signs.clone(),
            2 => array.residual_norms.clone(),
            3 => array.rotation_signs.clone(),
            _ => vortex_panic!("TurboQuantQJLArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &TurboQuantQJLArray, idx: usize) -> String {
        match idx {
            0 => "mse_inner".to_string(),
            1 => "qjl_signs".to_string(),
            2 => "residual_norms".to_string(),
            3 => "rotation_signs".to_string(),
            _ => vortex_panic!("TurboQuantQJLArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &TurboQuantQJLArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(TurboQuantQJLMetadata {
            bit_width: array.bit_width() as u32,
            padded_dim: array.padded_dim(),
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
            <ProstMetadata<TurboQuantQJLMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<TurboQuantQJLArray> {
        let padded_dim = metadata.padded_dim as usize;

        // Child 0 is a TurboQuantMSEArray — downcast from the type-erased ArrayRef.
        let mse_inner_ref = children.get(0, dtype, len)?;
        let mse_inner = Arc::new(
            mse_inner_ref
                .as_opt::<TurboQuantMSE>()
                .vortex_expect("QJL child 0 must be a TurboQuantMSEArray")
                .clone(),
        );

        let signs_dtype = DType::Bool(Nullability::NonNullable);
        let qjl_signs = children.get(1, &signs_dtype, len * padded_dim)?;

        let norms_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let residual_norms = children.get(2, &norms_dtype, len)?;

        let rotation_signs = children.get(3, &signs_dtype, 3 * padded_dim)?;

        Ok(TurboQuantQJLArray {
            dtype: dtype.clone(),
            mse_inner,
            qjl_signs,
            residual_norms,
            rotation_signs,
            stats_set: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 4,
            "TurboQuantQJLArray expects 4 children, got {}",
            children.len()
        );
        let mut iter = children.into_iter();
        let mse_ref = iter.next().vortex_expect("mse_inner child");
        array.mse_inner = Arc::new(
            mse_ref
                .as_opt::<TurboQuantMSE>()
                .vortex_expect("child 0 must be a TurboQuantMSEArray")
                .clone(),
        );
        array.qjl_signs = iter.next().vortex_expect("qjl_signs child");
        array.residual_norms = iter.next().vortex_expect("residual_norms child");
        array.rotation_signs = iter.next().vortex_expect("rotation_signs child");
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = Arc::try_unwrap(array)
            .map(|a| a.into_inner())
            .unwrap_or_else(|arc| arc.as_ref().deref().clone());
        Ok(ExecutionResult::done(execute_decompress_qjl(inner, ctx)?))
    }
}

impl ValidityChild<TurboQuantQJL> for TurboQuantQJL {
    fn validity_child(array: &TurboQuantQJLArray) -> &ArrayRef {
        array.mse_inner.codes()
    }
}
