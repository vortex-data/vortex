// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::canonical::canonicalize_onpair;
use crate::canonical::onpair_decode_views;
use crate::rules::RULES;

/// An [`OnPair`]-encoded Vortex array.
pub type OnPairArray = Array<OnPair>;

/// Wire-format metadata persisted alongside the OnPair buffer + slot children.
///
/// On disk the layout is FSST-shape:
///
/// * Buffer 0 — `dict_bytes`: the read-padded dictionary blob built by the
///   OnPair trainer.
/// * Slots — see [`OnPairSlots`].
///
/// The four integer slot children flow through the standard `compress_child`
/// pipeline (see `vortex-btrblocks::schemes::string::OnPairScheme`), so any
/// encoding registered with the compressor can re-encode them — exactly the
/// same shape as FSST's `codes` `VarBinArray`.
#[derive(Clone, prost::Message)]
pub struct OnPairMetadata {
    /// Width of the per-row primitive `uncompressed_lengths` child.
    #[prost(enumeration = "PType", tag = "1")]
    pub uncompressed_lengths_ptype: i32,
    /// Number of dictionary tokens. `dict_offsets` has length `dict_size + 1`.
    /// Bounded by 65_536, so `u32` is comfortably wide.
    #[prost(uint32, tag = "3")]
    pub dict_size: u32,
    /// Total number of tokens across all rows. `codes` has this length;
    /// `codes_offsets.last() == total_tokens`.
    #[prost(uint64, tag = "4")]
    pub total_tokens: u64,
    /// PType of the `dict_offsets` slot child (defaults to U32, may be
    /// narrowed to U16/U8 by the cascading compressor when values fit).
    #[prost(enumeration = "PType", tag = "5")]
    pub dict_offsets_ptype: i32,
    /// PType of the `codes` slot child (typically U16, may be narrowed to U8
    /// when the dictionary has at most 256 tokens).
    #[prost(enumeration = "PType", tag = "6")]
    pub codes_ptype: i32,
    /// PType of the `codes_offsets` slot child.
    #[prost(enumeration = "PType", tag = "7")]
    pub codes_offsets_ptype: i32,
}

impl OnPairMetadata {
    /// Decode the recorded [`PType`] of the `uncompressed_lengths` slot child.
    pub fn get_uncompressed_lengths_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.uncompressed_lengths_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.uncompressed_lengths_ptype))
    }
}

#[array_slots(OnPair)]
pub struct OnPairSlots {
    /// `PrimitiveArray<u32>`, length `dict_size + 1`. Cascading compressor may
    /// narrow the ptype to U16/U8.
    pub dict_offsets: ArrayRef,
    /// Primitive integer token codes. Downstream integer compression may
    /// narrow or bit-pack this child independently of the OnPair metadata.
    pub codes: ArrayRef,
    /// `PrimitiveArray<u32>`, length `num_rows + 1`. FoR / RunEnd / etc. apply
    /// naturally via the cascading compressor.
    pub codes_offsets: ArrayRef,
    /// Integer `PrimitiveArray`, length `num_rows`. Used to size the canonical
    /// output buffer.
    pub uncompressed_lengths: ArrayRef,
    /// Optional validity child for the outer string column.
    pub validity: Option<ArrayRef>,
}

/// Inner data for an OnPair-encoded array.
///
/// Holds only the dictionary blob (buffer 0). Every other piece —
/// `dict_offsets`, the per-token `codes`, the per-row `codes_offsets`, the
/// per-row `uncompressed_lengths`, and the optional validity child — is a
/// Vortex slot child so it can be re-encoded by the cascading compressor.
#[derive(Clone)]
pub struct OnPairData {
    /// The dictionary blob (buffer 0).
    ///
    /// INVARIANT: this buffer is an OnPair compact dictionary byte buffer,
    /// including its trailing read padding.
    dict_bytes: BufferHandle,
    dict_size: u32,
    len: usize,
}

impl OnPairData {
    /// Build [`OnPairData`] from the dictionary blob, its token count, and the
    /// number of rows.
    pub fn new(dict_bytes: BufferHandle, dict_size: u32, len: usize) -> Self {
        Self {
            dict_bytes,
            dict_size,
            len,
        }
    }

    /// Number of rows in the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bits per token code, derived from the dictionary size (not stored).
    pub fn bits(&self) -> u32 {
        u32::from(onpair::code_bits_for_num_tokens(self.dict_size as usize))
    }

    /// The dictionary blob as a host byte buffer.
    pub fn dict_bytes(&self) -> &ByteBuffer {
        self.dict_bytes.as_host()
    }

    /// The [`BufferHandle`] holding the dictionary blob (buffer 0).
    pub fn dict_bytes_handle(&self) -> &BufferHandle {
        &self.dict_bytes
    }
}

impl Display for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "len: {}, dict_size: {}, bits: {}, dict_bytes_len: {}",
            self.len,
            self.dict_size,
            self.bits(),
            self.dict_bytes.len()
        )
    }
}

impl Debug for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnPairData")
            .field("len", &self.len)
            .field("dict_size", &self.dict_size)
            .field("bits", &self.bits())
            .field("dict_bytes_len", &self.dict_bytes.len())
            .finish()
    }
}

impl ArrayHash for OnPairData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.dict_bytes.as_host().array_hash(state, accuracy);
        state.write_u32(self.dict_size);
    }
}

impl ArrayEq for OnPairData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.dict_size == other.dict_size
            && self
                .dict_bytes
                .as_host()
                .array_eq(other.dict_bytes.as_host(), accuracy)
    }
}

/// Zero-sized VTable marker for the OnPair encoding.
#[derive(Clone, Debug)]
pub struct OnPair;

impl OnPair {
    /// Build an [`OnPairArray`] from already-materialised parts.
    pub fn try_new(
        dtype: DType,
        dict_bytes: BufferHandle,
        dict_offsets: ArrayRef,
        codes: ArrayRef,
        codes_offsets: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> VortexResult<OnPairArray> {
        let dict_size = validate_parts(
            &dtype,
            &dict_offsets,
            &codes,
            &codes_offsets,
            &uncompressed_lengths,
        )?;
        let len = uncompressed_lengths.len();
        let data = OnPairData::new(dict_bytes, dict_size, len);
        let slots = OnPairSlots {
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(OnPair, dtype, len, data).with_slots(slots))
        })
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        dict_bytes: BufferHandle,
        dict_offsets: ArrayRef,
        codes: ArrayRef,
        codes_offsets: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> OnPairArray {
        let len = uncompressed_lengths.len();
        let dict_size = match u32::try_from(dict_offsets.len().saturating_sub(1)) {
            Ok(dict_size) => dict_size,
            Err(_) => vortex_panic!("OnPair dict_size exceeds u32"),
        };
        let data = OnPairData::new(dict_bytes, dict_size, len);
        let slots = OnPairSlots {
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(OnPair, dtype, len, data).with_slots(slots))
        }
    }
}

fn validate_parts(
    dtype: &DType,
    dict_offsets: &ArrayRef,
    codes: &ArrayRef,
    codes_offsets: &ArrayRef,
    uncompressed_lengths: &ArrayRef,
) -> VortexResult<u32> {
    vortex_ensure!(
        matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
        "OnPair arrays must be Binary or Utf8, found {dtype}"
    );

    if !dict_offsets.dtype().is_int() || dict_offsets.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "dict_offsets must be non-nullable integer");
    }
    if !codes.dtype().is_int() || codes.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "codes must be non-nullable integer");
    }
    if !codes_offsets.dtype().is_int() || codes_offsets.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "codes_offsets must be non-nullable integer");
    }
    if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "uncompressed_lengths must be non-nullable integer");
    }
    let dict_size = dictionary_size_from_offsets(dict_offsets.len())?;
    if codes_offsets.len() != uncompressed_lengths.len() + 1 {
        vortex_bail!(InvalidArgument:
            "codes_offsets.len ({}) != uncompressed_lengths.len + 1 ({})",
            codes_offsets.len(),
            uncompressed_lengths.len() + 1
        );
    }
    Ok(dict_size)
}

fn dictionary_size_from_offsets(dict_offsets_len: usize) -> VortexResult<u32> {
    let dict_size = dict_offsets_len
        .checked_sub(1)
        .ok_or_else(|| vortex_err!(InvalidArgument: "OnPair dict_offsets must not be empty"))?;
    u32::try_from(dict_size)
        .map_err(|_| vortex_err!(InvalidArgument: "OnPair dictionary size exceeds u32"))
}

impl VTable for OnPair {
    type TypedArrayData = OnPairData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.onpair");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let s = OnPairSlotsView::from_slots(slots);
        let dict_size = validate_parts(
            dtype,
            s.dict_offsets,
            s.codes,
            s.codes_offsets,
            s.uncompressed_lengths,
        )?;
        if data.dict_size != dict_size {
            vortex_bail!(InvalidArgument:
                "OnPairData dict_size {} does not match dict_offsets size {dict_size}",
                data.dict_size
            );
        }
        if s.uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have same len as outer array");
        }
        if data.len != len {
            vortex_bail!(InvalidArgument: "OnPairData len {} != outer len {}", data.len, len);
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.dict_bytes_handle().clone(),
            _ => vortex_panic!("OnPairArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("dict_bytes".to_string()),
            _ => vortex_panic!("OnPairArray buffer_name index {idx} out of bounds"),
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );
        let mut data = array.data().clone();
        data.dict_bytes = buffers[0].clone();
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let dict_size = u32::try_from(array.dict_offsets().len().saturating_sub(1))
            .map_err(|_| vortex_err!("OnPair dict_size exceeds u32"))?;
        let total_tokens = array.codes().len() as u64;
        Ok(Some(
            OnPairMetadata {
                uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
                dict_size,
                total_tokens,
                dict_offsets_ptype: array.dict_offsets().dtype().as_ptype().into(),
                codes_ptype: array.codes().dtype().as_ptype().into(),
                codes_offsets_ptype: array.codes_offsets().dtype().as_ptype().into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        if buffers.len() != 1 {
            vortex_bail!(InvalidArgument: "Expected 1 buffer, got {}", buffers.len());
        }
        let metadata = OnPairMetadata::decode(metadata)?;
        let uncompressed_ptype = metadata.get_uncompressed_lengths_ptype()?;

        // Slot children. We pass `usize::MAX` for slots whose length we
        // don't know up front (`dict_offsets` and `codes`). `codes_offsets`
        // has known length `len + 1`.
        let dict_offsets_len = metadata.dict_size as usize + 1;
        let total_tokens = usize::try_from(metadata.total_tokens)
            .map_err(|_| vortex_err!("total_tokens {} overflows usize", metadata.total_tokens))?;
        // The cascading compressor may have narrowed any of these integer
        // children to a tighter ptype; the recorded ptype tells the framework
        // exactly which dtype to materialise as.
        let dict_offsets_ptype = PType::try_from(metadata.dict_offsets_ptype).map_err(|_| {
            vortex_err!("invalid dict_offsets_ptype {}", metadata.dict_offsets_ptype)
        })?;
        let codes_ptype = PType::try_from(metadata.codes_ptype)
            .map_err(|_| vortex_err!("invalid codes_ptype {}", metadata.codes_ptype))?;
        let codes_offsets_ptype = PType::try_from(metadata.codes_offsets_ptype).map_err(|_| {
            vortex_err!(
                "invalid codes_offsets_ptype {}",
                metadata.codes_offsets_ptype
            )
        })?;
        let dict_offsets = children.get(
            0,
            &DType::Primitive(dict_offsets_ptype, Nullability::NonNullable),
            dict_offsets_len,
        )?;
        let codes = children.get(
            1,
            &DType::Primitive(codes_ptype, Nullability::NonNullable),
            total_tokens,
        )?;
        let codes_offsets = children.get(
            2,
            &DType::Primitive(codes_offsets_ptype, Nullability::NonNullable),
            len + 1,
        )?;
        let uncompressed_lengths = children.get(
            3,
            &DType::Primitive(uncompressed_ptype, Nullability::NonNullable),
            len,
        )?;
        let validity = match children.len() {
            4 => Validity::from(dtype.nullability()),
            5 => Validity::Array(children.get(4, &Validity::DTYPE, len)?),
            other => vortex_bail!(InvalidArgument: "Expected 4 or 5 children, got {other}"),
        };

        let data = OnPairData::new(buffers[0].clone(), metadata.dict_size, len);
        let slots = OnPairSlots {
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        OnPairSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        canonicalize_onpair(array.as_view(), ctx).map(ExecutionResult::done)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            return array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array()
                .append_to_builder(builder, ctx);
        };

        let next_buffer_index = builder.completed_block_count() + u32::from(builder.in_progress());
        let (buffers, views) = onpair_decode_views(array, next_buffer_index, ctx)?;
        builder.push_buffer_and_adjusted_views(
            &buffers,
            &views,
            array
                .array()
                .validity()?
                .execute_mask(array.array().len(), ctx)?,
        );
        Ok(())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl ValidityVTable<OnPair> for OnPair {
    fn validity(array: ArrayView<'_, OnPair>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.slots()[OnPairSlots::VALIDITY].as_ref(),
            array.dtype().nullability(),
        ))
    }
}

/// Convenience methods on top of the macro-generated [`OnPairArraySlotsExt`].
pub trait OnPairArrayExt: OnPairArraySlotsExt {
    /// The array's [`Validity`], derived from the optional validity child and
    /// the outer dtype's nullability.
    fn array_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[OnPairSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: OnPairArraySlotsExt> OnPairArrayExt for T {}
