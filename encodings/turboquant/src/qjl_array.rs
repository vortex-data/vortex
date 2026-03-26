// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant QJL array: inner-product-preserving quantization (MSE + QJL residual).
//!
//! Wraps a [`TurboQuantMSEArray`] (at `bit_width - 1`) and adds a 1-bit QJL
//! residual correction for unbiased inner product estimation.

use std::fmt::Debug;
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
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
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

use crate::decompress::execute_decompress_qjl;

vtable!(TurboQuantQJL);

/// Encoding marker type for TurboQuant QJL.
#[derive(Clone, Debug)]
pub struct TurboQuantQJL;

impl TurboQuantQJL {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant.qjl");
}

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
        array.bit_width.hash(state);
        array.padded_dim.hash(state);
        array.rotation_seed.hash(state);
        array.mse_inner.array_hash(state, precision);
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
            && array.bit_width == other.bit_width
            && array.padded_dim == other.padded_dim
            && array.rotation_seed == other.rotation_seed
            && array.mse_inner.array_eq(&other.mse_inner, precision)
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
            0 => array.mse_inner.clone(),
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

        // Child 0: mse_inner (TurboQuantMSEArray, opaque ArrayRef).
        // We pass the parent dtype and len — the MSE array has the same logical shape.
        let mse_inner = children.get(0, dtype, len)?;

        // Child 1: qjl_signs (BoolArray, length num_rows * padded_dim).
        let signs_dtype = DType::Bool(Nullability::NonNullable);
        let qjl_signs = children.get(1, &signs_dtype, len * padded_dim)?;

        // Child 2: residual_norms (f32, one per row).
        let norms_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let residual_norms = children.get(2, &norms_dtype, len)?;

        // Child 3: rotation_signs (BoolArray, length 3 * padded_dim).
        let rotation_signs = children.get(3, &signs_dtype, 3 * padded_dim)?;

        Ok(TurboQuantQJLArray {
            dtype: dtype.clone(),
            mse_inner,
            qjl_signs,
            residual_norms,
            rotation_signs,
            bit_width: u8::try_from(metadata.bit_width)?,
            padded_dim: metadata.padded_dim,
            rotation_seed: metadata.rotation_seed,
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
        array.mse_inner = iter.next().vortex_expect("mse_inner child");
        array.qjl_signs = iter.next().vortex_expect("qjl_signs child");
        array.residual_norms = iter.next().vortex_expect("residual_norms child");
        array.rotation_signs = iter.next().vortex_expect("rotation_signs child");
        Ok(())
    }

    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = Arc::try_unwrap(array).unwrap_or_else(|arc| (*arc).clone());
        Ok(ExecutionResult::done(execute_decompress_qjl(array, ctx)?))
    }
}

/// Protobuf metadata for TurboQuant QJL encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantQJLMetadata {
    /// Total bit width (2-9, including QJL bit; MSE child uses bit_width - 1).
    #[prost(uint32, tag = "1")]
    pub bit_width: u32,
    /// Padded dimension (next power of 2 >= dimension).
    #[prost(uint32, tag = "2")]
    pub padded_dim: u32,
    /// QJL rotation seed (for debugging/reproducibility).
    #[prost(uint64, tag = "3")]
    pub rotation_seed: u64,
}

/// TurboQuant QJL array: wraps a TurboQuantMSEArray with QJL residual correction.
#[derive(Clone, Debug)]
pub struct TurboQuantQJLArray {
    /// The original dtype (FixedSizeList of floats).
    pub(crate) dtype: DType,
    /// Child 0: inner TurboQuantMSEArray (at bit_width - 1).
    pub(crate) mse_inner: ArrayRef,
    /// Child 1: QJL sign bits (BoolArray, length num_rows * padded_dim).
    pub(crate) qjl_signs: ArrayRef,
    /// Child 2: f32 residual norms, one per row.
    pub(crate) residual_norms: ArrayRef,
    /// Child 3: QJL rotation signs (BoolArray, length 3 * padded_dim, inverse order).
    pub(crate) rotation_signs: ArrayRef,
    /// Total bit width (including QJL bit).
    pub(crate) bit_width: u8,
    /// Padded dimension.
    pub(crate) padded_dim: u32,
    /// QJL rotation seed.
    pub(crate) rotation_seed: u64,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantQJLArray {
    /// Build a new TurboQuantQJLArray.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        mse_inner: ArrayRef,
        qjl_signs: ArrayRef,
        residual_norms: ArrayRef,
        rotation_signs: ArrayRef,
        bit_width: u8,
        padded_dim: u32,
        rotation_seed: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (2..=9).contains(&bit_width),
            "QJL bit_width must be 2-9, got {bit_width}"
        );
        Ok(Self {
            dtype,
            mse_inner,
            qjl_signs,
            residual_norms,
            rotation_signs,
            bit_width,
            padded_dim,
            rotation_seed,
            stats_set: Default::default(),
        })
    }

    /// Total bit width (including QJL bit).
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Padded dimension.
    pub fn padded_dim(&self) -> u32 {
        self.padded_dim
    }

    /// QJL rotation seed.
    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// The inner MSE array child.
    pub fn mse_inner(&self) -> &ArrayRef {
        &self.mse_inner
    }

    /// The QJL sign bits child (BoolArray).
    pub fn qjl_signs(&self) -> &ArrayRef {
        &self.qjl_signs
    }

    /// The residual norms child.
    pub fn residual_norms(&self) -> &ArrayRef {
        &self.residual_norms
    }

    /// The QJL rotation signs child (BoolArray).
    pub fn rotation_signs(&self) -> &ArrayRef {
        &self.rotation_signs
    }
}

impl ValidityChild<TurboQuantQJL> for TurboQuantQJL {
    fn validity_child(array: &TurboQuantQJLArray) -> &ArrayRef {
        array.mse_inner()
    }
}
