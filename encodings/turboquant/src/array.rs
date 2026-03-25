// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::decompress::execute_decompress;

vtable!(TurboQuant);

/// The TurboQuant variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TurboQuantVariant {
    /// MSE-optimal quantization.
    Mse = 0,
    /// Inner-product-optimal quantization (MSE + QJL residual).
    Prod = 1,
}

impl TurboQuantVariant {
    fn from_u32(v: u32) -> VortexResult<Self> {
        match v {
            0 => Ok(Self::Mse),
            1 => Ok(Self::Prod),
            _ => vortex_bail!("Invalid TurboQuant variant: {v}"),
        }
    }
}

impl VTable for TurboQuant {
    type Array = TurboQuantArray;
    type Metadata = ProstMetadata<TurboQuantMetadata>;
    type OperationsVTable = NotSupported;
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
        array.codes.array_hash(state, precision);
        array.norms.array_hash(state, precision);
        array.dimension.hash(state);
        array.bit_width.hash(state);
        array.rotation_seed.hash(state);
        array.variant.hash(state);
    }

    fn array_eq(array: &TurboQuantArray, other: &TurboQuantArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.dimension == other.dimension
            && array.bit_width == other.bit_width
            && array.rotation_seed == other.rotation_seed
            && array.variant == other.variant
            && array.codes.array_eq(&other.codes, precision)
            && array.norms.array_eq(&other.norms, precision)
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
        match array.variant {
            TurboQuantVariant::Mse => 2,
            TurboQuantVariant::Prod => 4,
        }
    }

    fn child(array: &TurboQuantArray, idx: usize) -> ArrayRef {
        match (idx, array.variant) {
            (0, _) => array.codes.clone(),
            (1, _) => array.norms.clone(),
            (2, TurboQuantVariant::Prod) => array
                .qjl_signs
                .as_ref()
                .unwrap_or_else(|| vortex_panic!("TurboQuantArray child 2 out of bounds"))
                .clone(),
            (3, TurboQuantVariant::Prod) => array
                .residual_norms
                .as_ref()
                .unwrap_or_else(|| vortex_panic!("TurboQuantArray child 3 out of bounds"))
                .clone(),
            _ => vortex_panic!("TurboQuantArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &TurboQuantArray, idx: usize) -> String {
        match idx {
            0 => "codes".to_string(),
            1 => "norms".to_string(),
            2 => "qjl_signs".to_string(),
            3 => "residual_norms".to_string(),
            _ => vortex_panic!("TurboQuantArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &TurboQuantArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(TurboQuantMetadata {
            dimension: array.dimension,
            bit_width: array.bit_width as u32,
            rotation_seed: array.rotation_seed,
            variant: array.variant as u32,
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
        let variant = TurboQuantVariant::from_u32(metadata.variant)?;
        let bit_width = u8::try_from(metadata.bit_width)?;
        let d = metadata.dimension as usize;

        // Codes child: flat u8 array of quantized indices (num_rows * d elements), bitpacked.
        let codes_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let codes = children.get(0, &codes_dtype, len * d)?;

        // Norms child: f32 array, one per row.
        let norms_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let norms = children.get(1, &norms_dtype, len)?;

        let (qjl_signs, residual_norms) = if variant == TurboQuantVariant::Prod {
            // QJL signs: packed u8 bytes.
            let sign_bytes_count = (len * d).div_ceil(8);
            let signs = children.get(
                2,
                &DType::Primitive(PType::U8, Nullability::NonNullable),
                sign_bytes_count,
            )?;
            let res_norms = children.get(3, &norms_dtype, len)?;
            (Some(signs), Some(res_norms))
        } else {
            (None, None)
        };

        Ok(TurboQuantArray {
            dtype: dtype.clone(),
            codes,
            norms,
            qjl_signs,
            residual_norms,
            dimension: metadata.dimension,
            bit_width,
            rotation_seed: metadata.rotation_seed,
            variant,
            stats_set: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        let expected = match array.variant {
            TurboQuantVariant::Mse => 2,
            TurboQuantVariant::Prod => 4,
        };
        vortex_ensure!(
            children.len() == expected,
            "TurboQuantArray expects {expected} children, got {}",
            children.len()
        );

        let mut iter = children.into_iter();
        array.codes = iter.next().vortex_expect("codes child");
        array.norms = iter.next().vortex_expect("norms child");
        if array.variant == TurboQuantVariant::Prod {
            array.qjl_signs = Some(iter.next().vortex_expect("qjl_signs child"));
            array.residual_norms = Some(iter.next().vortex_expect("residual_norms child"));
        }
        Ok(())
    }

    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = Arc::try_unwrap(array).unwrap_or_else(|arc| (*arc).clone());
        Ok(ExecutionResult::done(execute_decompress(array, ctx)?))
    }

    // No parent kernels: TurboQuant decompresses fully via execute().
}

/// Protobuf metadata for TurboQuant encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantMetadata {
    /// Vector dimension d.
    #[prost(uint32, tag = "1")]
    pub dimension: u32,
    /// Bits per coordinate (1-4).
    #[prost(uint32, tag = "2")]
    pub bit_width: u32,
    /// Deterministic seed for rotation matrix Π.
    #[prost(uint64, tag = "3")]
    pub rotation_seed: u64,
    /// Variant: 0 = Mse, 1 = Prod.
    #[prost(uint32, tag = "4")]
    pub variant: u32,
}

/// The TurboQuant array stores quantized vector data.
#[derive(Clone, Debug)]
pub struct TurboQuantArray {
    /// The original dtype (FixedSizeList of floats).
    pub(crate) dtype: DType,
    /// Child 0: bit-packed quantized indices (via FastLanes BitPackedArray).
    pub(crate) codes: ArrayRef,
    /// Child 1: f32 norms, one per vector row.
    pub(crate) norms: ArrayRef,
    /// Child 2 (Prod only): QJL sign bits as a boolean array.
    pub(crate) qjl_signs: Option<ArrayRef>,
    /// Child 3 (Prod only): f32 residual norms, one per row.
    pub(crate) residual_norms: Option<ArrayRef>,
    /// Vector dimension.
    pub(crate) dimension: u32,
    /// Bits per coordinate.
    pub(crate) bit_width: u8,
    /// Rotation matrix seed.
    pub(crate) rotation_seed: u64,
    /// TurboQuant variant.
    pub(crate) variant: TurboQuantVariant,
    pub(crate) stats_set: ArrayStats,
}

/// Encoding marker type.
#[derive(Clone, Debug)]
pub struct TurboQuant;

impl TurboQuant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant");
}

impl TurboQuantArray {
    /// Build a new TurboQuantArray for the MSE variant.
    pub fn try_new_mse(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        dimension: u32,
        bit_width: u8,
        rotation_seed: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!((1..=4).contains(&bit_width), "bit_width must be 1-4");
        Ok(Self {
            dtype,
            codes,
            norms,
            qjl_signs: None,
            residual_norms: None,
            dimension,
            bit_width,
            rotation_seed,
            variant: TurboQuantVariant::Mse,
            stats_set: Default::default(),
        })
    }

    /// Build a new TurboQuantArray for the Prod variant.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_prod(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        qjl_signs: ArrayRef,
        residual_norms: ArrayRef,
        dimension: u32,
        bit_width: u8,
        rotation_seed: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (2..=4).contains(&bit_width),
            "Prod variant bit_width must be 2-4"
        );
        Ok(Self {
            dtype,
            codes,
            norms,
            qjl_signs: Some(qjl_signs),
            residual_norms: Some(residual_norms),
            dimension,
            bit_width,
            rotation_seed,
            variant: TurboQuantVariant::Prod,
            stats_set: Default::default(),
        })
    }

    /// The vector dimension d.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// Bits per coordinate.
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// The rotation matrix seed.
    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// The TurboQuant variant.
    pub fn variant(&self) -> TurboQuantVariant {
        self.variant
    }

    /// The bit-packed codes child.
    pub fn codes(&self) -> &ArrayRef {
        &self.codes
    }

    /// The norms child.
    pub fn norms(&self) -> &ArrayRef {
        &self.norms
    }

    /// The QJL signs child (Prod variant only).
    pub fn qjl_signs(&self) -> Option<&ArrayRef> {
        self.qjl_signs.as_ref()
    }

    /// The residual norms child (Prod variant only).
    pub fn residual_norms(&self) -> Option<&ArrayRef> {
        self.residual_norms.as_ref()
    }
}

impl ValidityChild<TurboQuant> for TurboQuant {
    fn validity_child(array: &TurboQuantArray) -> &ArrayRef {
        array.norms()
    }
}
