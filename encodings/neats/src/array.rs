// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// A NeaTS-encoded Vortex array.
pub type NeaTSArray = Array<NeaTS>;

/// The NeaTS encoding marker.
#[derive(Clone, Debug)]
pub struct NeaTS;

/// On-disk metadata for a NeaTS array.
///
/// The bulk of the encoded information lives in the array's six child slots
/// (`piece_starts`, `model_ids`, `coeff_a`, `coeff_b`, `coeff_c`, `residuals`).
/// Only the parameters that aren't already captured by a child array live here.
#[derive(Clone, prost::Message)]
pub struct NeaTSMetadata {
    /// The logical (decoded) primitive type. Always `PType::F32` or `PType::F64`.
    #[prost(enumeration = "PType", tag = "1")]
    pub value_ptype: i32,
    /// The on-disk residual ptype. Always a signed integer; defaults to `PType::I64`.
    #[prost(enumeration = "PType", tag = "2")]
    pub residual_ptype: i32,
    /// Number of pieces P. Set to the length of `piece_starts - 1`.
    #[prost(uint64, tag = "3")]
    pub num_pieces: u64,
    /// The bit pattern of the residual quantization scale (`f64::to_bits`).
    #[prost(uint64, tag = "4")]
    pub scale_bits: u64,
    /// The bit pattern of the per-value error bound (`f64::to_bits`). Zero means lossless.
    #[prost(uint64, tag = "5")]
    pub epsilon_bits: u64,
}

/// Runtime data for a NeaTS array.
#[derive(Clone, Debug)]
pub struct NeaTSData {
    scale: f64,
    epsilon: f64,
}

impl NeaTSData {
    /// Build a new [`NeaTSData`] with the given residual scale and error bound.
    pub fn new(scale: f64, epsilon: f64) -> Self {
        Self { scale, epsilon }
    }

    /// The residual quantization scale. A decoded value at piece `p`, offset `t` is
    /// `model_p(t) + residual_i * scale`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The per-value absolute error bound. `0.0` indicates the array was compressed in
    /// lossless mode.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Display for NeaTSData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "scale: {}, epsilon: {}", self.scale, self.epsilon)
    }
}

impl ArrayHash for NeaTSData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.scale.to_bits().hash(state);
        self.epsilon.to_bits().hash(state);
    }
}

impl ArrayEq for NeaTSData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.scale.to_bits() == other.scale.to_bits()
            && self.epsilon.to_bits() == other.epsilon.to_bits()
    }
}

/// The slot layout of a NeaTS array. Fields are declared in slot-index order.
#[array_slots(NeaTS)]
pub struct NeaTSSlots {
    /// Monotonically non-decreasing piece boundaries (`u32`, length `P+1`). `piece_starts[0] == 0`
    /// and `piece_starts[P] == array_len`.
    pub piece_starts: ArrayRef,
    /// One [`crate::models::ModelKind`] byte per piece (`u8`, length `P`).
    pub model_ids: ArrayRef,
    /// Per-piece coefficient `a` (`f64`, length `P`).
    pub coeff_a: ArrayRef,
    /// Per-piece coefficient `b` (`f64`, length `P`).
    pub coeff_b: ArrayRef,
    /// Per-piece coefficient `c` (`f64`, length `P`).
    pub coeff_c: ArrayRef,
    /// Per-element quantized residual (`i64`, length `N`). Carries validity.
    pub residuals: ArrayRef,
}

impl VTable for NeaTS {
    type TypedArrayData = NeaTSData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.neats");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let view = NeaTSSlotsView::from_slots(slots);
        validate_parts(
            dtype,
            len,
            data,
            view.piece_starts,
            view.model_ids,
            view.coeff_a,
            view.coeff_b,
            view.coeff_c,
            view.residuals,
        )
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("NeaTSArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let data = array.data();
        let value_ptype = PType::try_from(array.dtype())?;
        let residual_ptype = PType::try_from(array.residuals().dtype())?;
        let num_pieces = array.model_ids().len() as u64;
        Ok(Some(
            NeaTSMetadata {
                value_ptype: value_ptype as i32,
                residual_ptype: residual_ptype as i32,
                num_pieces,
                scale_bits: data.scale().to_bits(),
                epsilon_bits: data.epsilon().to_bits(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = NeaTSMetadata::decode(metadata)?;
        if children.len() != NeaTSSlots::COUNT {
            vortex_bail!(
                "Expected {} children for NeaTS encoding, found {}",
                NeaTSSlots::COUNT,
                children.len()
            );
        }
        let value_ptype = PType::try_from(metadata.value_ptype)
            .map_err(|_| vortex_err!("Invalid value ptype {}", metadata.value_ptype))?;
        let residual_ptype = PType::try_from(metadata.residual_ptype)
            .map_err(|_| vortex_err!("Invalid residual ptype {}", metadata.residual_ptype))?;
        let p = usize::try_from(metadata.num_pieces)
            .map_err(|_| vortex_err!("num_pieces {} does not fit in usize", metadata.num_pieces))?;

        let piece_starts = children.get(
            NeaTSSlots::PIECE_STARTS,
            &DType::Primitive(PType::U32, Nullability::NonNullable),
            p + 1,
        )?;
        let model_ids = children.get(
            NeaTSSlots::MODEL_IDS,
            &DType::Primitive(PType::U8, Nullability::NonNullable),
            p,
        )?;
        let coeff_a = children.get(
            NeaTSSlots::COEFF_A,
            &DType::Primitive(PType::F64, Nullability::NonNullable),
            p,
        )?;
        let coeff_b = children.get(
            NeaTSSlots::COEFF_B,
            &DType::Primitive(PType::F64, Nullability::NonNullable),
            p,
        )?;
        let coeff_c = children.get(
            NeaTSSlots::COEFF_C,
            &DType::Primitive(PType::F64, Nullability::NonNullable),
            p,
        )?;
        let residuals = children.get(
            NeaTSSlots::RESIDUALS,
            &DType::Primitive(residual_ptype, dtype.nullability()),
            len,
        )?;

        // Cross-check ptypes.
        let expected = match value_ptype {
            PType::F32 => DType::Primitive(PType::F32, dtype.nullability()),
            PType::F64 => DType::Primitive(PType::F64, dtype.nullability()),
            other => vortex_bail!("NeaTS logical dtype must be f32 or f64, got {other}"),
        };
        vortex_ensure!(
            &expected == dtype,
            "NeaTS metadata value_ptype {} disagrees with array dtype {}",
            value_ptype,
            dtype,
        );

        let slots = smallvec![
            Some(piece_starts),
            Some(model_ids),
            Some(coeff_a),
            Some(coeff_b),
            Some(coeff_c),
            Some(residuals),
        ];
        let data = NeaTSData::new(
            f64::from_bits(metadata.scale_bits),
            f64::from_bits(metadata.epsilon_bits),
        );
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        NeaTSSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            crate::canonical::decode_to_primitive(&array, ctx)?.into_array(),
        ))
    }
}

impl ValidityChild<NeaTS> for NeaTS {
    fn validity_child(array: ArrayView<'_, NeaTS>) -> ArrayRef {
        array.residuals().clone()
    }
}

impl NeaTS {
    /// Build a new [`NeaTSArray`] from already-validated parts.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        data: NeaTSData,
        piece_starts: ArrayRef,
        model_ids: ArrayRef,
        coeff_a: ArrayRef,
        coeff_b: ArrayRef,
        coeff_c: ArrayRef,
        residuals: ArrayRef,
    ) -> VortexResult<NeaTSArray> {
        let len = residuals.len();
        validate_parts(
            &dtype,
            len,
            &data,
            &piece_starts,
            &model_ids,
            &coeff_a,
            &coeff_b,
            &coeff_c,
            &residuals,
        )?;
        let slots = smallvec![
            Some(piece_starts),
            Some(model_ids),
            Some(coeff_a),
            Some(coeff_b),
            Some(coeff_c),
            Some(residuals),
        ];
        Array::try_from_parts(ArrayParts::new(NeaTS, dtype, len, data).with_slots(slots))
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_parts(
    dtype: &DType,
    len: usize,
    _data: &NeaTSData,
    piece_starts: &ArrayRef,
    model_ids: &ArrayRef,
    coeff_a: &ArrayRef,
    coeff_b: &ArrayRef,
    coeff_c: &ArrayRef,
    residuals: &ArrayRef,
) -> VortexResult<()> {
    let DType::Primitive(value_ptype, dtype_nullability) = dtype else {
        vortex_bail!("NeaTS dtype must be primitive f32 or f64, got {dtype}");
    };
    vortex_ensure!(
        matches!(value_ptype, PType::F32 | PType::F64),
        "NeaTS only supports f32 or f64, got {value_ptype}",
    );
    vortex_ensure!(
        residuals.len() == len,
        "residuals len {} != logical len {len}",
        residuals.len(),
    );
    vortex_ensure!(
        residuals.dtype().is_signed_int() && !residuals.dtype().is_nullable()
            || residuals.dtype().is_signed_int()
                && residuals.dtype().nullability() == *dtype_nullability,
        "residuals must be signed integer with the same nullability as the array, got {}",
        residuals.dtype(),
    );

    let num_starts = piece_starts.len();
    vortex_ensure!(
        num_starts >= 1,
        "piece_starts must have length P+1 >= 1, got 0",
    );
    let p = num_starts - 1;
    vortex_ensure!(
        model_ids.len() == p,
        "model_ids len {} != P={p}",
        model_ids.len()
    );
    vortex_ensure!(coeff_a.len() == p, "coeff_a len {} != P={p}", coeff_a.len());
    vortex_ensure!(coeff_b.len() == p, "coeff_b len {} != P={p}", coeff_b.len());
    vortex_ensure!(coeff_c.len() == p, "coeff_c len {} != P={p}", coeff_c.len());

    vortex_ensure!(
        matches!(
            piece_starts.dtype(),
            DType::Primitive(PType::U32, Nullability::NonNullable)
        ),
        "piece_starts must be non-nullable u32, got {}",
        piece_starts.dtype(),
    );
    vortex_ensure!(
        matches!(
            model_ids.dtype(),
            DType::Primitive(PType::U8, Nullability::NonNullable)
        ),
        "model_ids must be non-nullable u8, got {}",
        model_ids.dtype(),
    );
    for (name, c) in [
        ("coeff_a", coeff_a),
        ("coeff_b", coeff_b),
        ("coeff_c", coeff_c),
    ] {
        vortex_ensure!(
            matches!(
                c.dtype(),
                DType::Primitive(PType::F64, Nullability::NonNullable)
            ),
            "{name} must be non-nullable f64, got {}",
            c.dtype(),
        );
    }
    Ok(())
}
