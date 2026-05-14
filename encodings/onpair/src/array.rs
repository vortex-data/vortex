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
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
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
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

/// An [`OnPair`]-encoded Vortex array.
pub type OnPairArray = Array<OnPair>;

/// Default bits-per-token preset used by [`crate::onpair_compress`]: 12-bit
/// codes, dictionary capped at 4 096 entries.
pub const DEFAULT_BITS: u32 = 12;

/// Wire-format metadata persisted alongside the OnPair buffers and children.
///
/// On disk the layout is:
///
/// * Buffer 0 — `dict_bytes`: dictionary blob built by the C++ trainer.
/// * Buffer 1 — `dict_offsets`: `dict_size + 1` u32 offsets into `dict_bytes`,
///   stored as raw little-endian bytes.
/// * Buffer 2 — `codes`: per-token `u16` ids, stored as raw little-endian
///   bytes. Each value only uses its low `bits` bits, but we keep the u16
///   width on disk so the decode loop is a straight indexed lookup without
///   bit-unpacking. Downstream compaction can still re-encode this buffer
///   externally.
/// * Buffer 3 — `codes_offsets`: `num_rows + 1` u32 offsets into `codes`,
///   stored as raw little-endian bytes.
/// * Slot 0   — `uncompressed_lengths`: `PrimitiveArray<integer>`.
/// * Slot 1   — optional validity child.
///
/// All integer arrays live as raw byte buffers (not primitive slot
/// children) because the Vortex flat-segment writer aligns a segment to the
/// alignment of its first buffer; nested children later in the same segment
/// may not be sufficiently aligned to load as `PrimitiveArray<uN>`. Raw
/// buffers go through `BufferHandle` and survive the round-trip
/// byte-identical regardless of how the writer batches them.
#[derive(Clone, prost::Message)]
pub struct OnPairMetadata {
    /// Width of the per-row primitive `uncompressed_lengths` child.
    #[prost(enumeration = "PType", tag = "1")]
    pub uncompressed_lengths_ptype: i32,
    /// Bits-per-token the column was compressed with (9..=16). Every value in
    /// the `codes` child only uses its low `bits` bits.
    #[prost(uint32, tag = "2")]
    pub bits: u32,
}

impl OnPairMetadata {
    pub fn get_uncompressed_lengths_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.uncompressed_lengths_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.uncompressed_lengths_ptype))
    }
}

/// Slot indices on the outer [`Array`].
pub(crate) const UNCOMPRESSED_LENGTHS_SLOT: usize = 0;
pub(crate) const VALIDITY_SLOT: usize = 1;
pub(crate) const NUM_SLOTS: usize = 2;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] = ["uncompressed_lengths", "validity"];

/// Buffer indices.
pub(crate) const DICT_BYTES_BUF: usize = 0;
pub(crate) const DICT_OFFSETS_BUF: usize = 1;
pub(crate) const CODES_BUF: usize = 2;
pub(crate) const CODES_OFFSETS_BUF: usize = 3;

/// Inner data for an OnPair-encoded array.
///
/// Holds the three byte buffers that carry the dictionary blob and the two
/// integer offset arrays. Their alignments (u32 for `dict_offsets` and
/// `codes_offsets`) are tracked by the underlying `ByteBuffer` so the
/// segment writer pads them correctly on disk.
#[derive(Clone)]
pub struct OnPairData {
    dict_bytes: BufferHandle,
    dict_offsets: BufferHandle,
    codes: BufferHandle,
    codes_offsets: BufferHandle,
    bits: u32,
    len: usize,
}

impl OnPairData {
    pub fn new(
        dict_bytes: BufferHandle,
        dict_offsets: BufferHandle,
        codes: BufferHandle,
        codes_offsets: BufferHandle,
        bits: u32,
        len: usize,
    ) -> Self {
        Self {
            dict_bytes,
            dict_offsets,
            codes,
            codes_offsets,
            bits,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bits(&self) -> u32 {
        self.bits
    }

    pub fn dict_bytes(&self) -> &ByteBuffer {
        self.dict_bytes.as_host()
    }

    pub fn dict_bytes_handle(&self) -> &BufferHandle {
        &self.dict_bytes
    }

    pub fn dict_offsets_bytes(&self) -> &ByteBuffer {
        self.dict_offsets.as_host()
    }

    pub fn dict_offsets_handle(&self) -> &BufferHandle {
        &self.dict_offsets
    }

    pub fn codes_bytes_raw(&self) -> &ByteBuffer {
        self.codes.as_host()
    }

    pub fn codes_handle(&self) -> &BufferHandle {
        &self.codes
    }

    pub fn codes_offsets_bytes(&self) -> &ByteBuffer {
        self.codes_offsets.as_host()
    }

    pub fn codes_offsets_handle(&self) -> &BufferHandle {
        &self.codes_offsets
    }
}

impl Display for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "len: {}, bits: {}, dict_bytes: {}",
            self.len,
            self.bits,
            self.dict_bytes.len()
        )
    }
}

impl Debug for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnPairData")
            .field("len", &self.len)
            .field("bits", &self.bits)
            .field("dict_bytes_len", &self.dict_bytes.len())
            .field("dict_offsets_len", &self.dict_offsets.len())
            .field("codes_len", &self.codes.len())
            .field("codes_offsets_len", &self.codes_offsets.len())
            .finish()
    }
}

impl ArrayHash for OnPairData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.dict_bytes.as_host().array_hash(state, precision);
        self.dict_offsets.as_host().array_hash(state, precision);
        self.codes.as_host().array_hash(state, precision);
        self.codes_offsets.as_host().array_hash(state, precision);
        state.write_u32(self.bits);
    }
}

impl ArrayEq for OnPairData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.bits == other.bits
            && self
                .dict_bytes
                .as_host()
                .array_eq(other.dict_bytes.as_host(), precision)
            && self
                .dict_offsets
                .as_host()
                .array_eq(other.dict_offsets.as_host(), precision)
            && self
                .codes
                .as_host()
                .array_eq(other.codes.as_host(), precision)
            && self
                .codes_offsets
                .as_host()
                .array_eq(other.codes_offsets.as_host(), precision)
    }
}

/// Zero-sized VTable marker for the OnPair encoding.
#[derive(Clone, Debug)]
pub struct OnPair;

impl OnPair {
    /// Build an [`OnPairArray`] from already-materialised parts.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        dict_bytes: BufferHandle,
        dict_offsets: BufferHandle,
        codes: BufferHandle,
        codes_offsets: BufferHandle,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
        bits: u32,
    ) -> VortexResult<OnPairArray> {
        validate_parts(
            &dtype,
            &dict_offsets,
            &codes,
            &codes_offsets,
            &uncompressed_lengths,
            bits,
        )?;
        let len = uncompressed_lengths.len();
        let data = OnPairData::new(dict_bytes, dict_offsets, codes, codes_offsets, bits, len);
        let slots: ArraySlots = smallvec![
            Some(uncompressed_lengths),
            validity_to_child(&validity, len),
        ];
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(OnPair, dtype, len, data).with_slots(slots))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        dict_bytes: BufferHandle,
        dict_offsets: BufferHandle,
        codes: BufferHandle,
        codes_offsets: BufferHandle,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
        bits: u32,
    ) -> OnPairArray {
        let len = uncompressed_lengths.len();
        let data = OnPairData::new(dict_bytes, dict_offsets, codes, codes_offsets, bits, len);
        let slots: ArraySlots = smallvec![
            Some(uncompressed_lengths),
            validity_to_child(&validity, len),
        ];
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(OnPair, dtype, len, data).with_slots(slots))
        }
    }
}

fn validate_parts(
    dtype: &DType,
    dict_offsets: &BufferHandle,
    codes: &BufferHandle,
    codes_offsets: &BufferHandle,
    uncompressed_lengths: &ArrayRef,
    bits: u32,
) -> VortexResult<()> {
    vortex_ensure!(
        matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
        "OnPair arrays must be Binary or Utf8, found {dtype}"
    );
    vortex_ensure!((9..=16).contains(&bits), "bits {bits} out of range [9, 16]");

    if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "uncompressed_lengths must be non-nullable integer");
    }

    let n = uncompressed_lengths.len();
    if codes_offsets.len() != (n + 1) * 4 {
        vortex_bail!(InvalidArgument:
            "codes_offsets buffer length ({}) != (n + 1) * 4 ({})",
            codes_offsets.len(),
            (n + 1) * 4
        );
    }
    if !codes.len().is_multiple_of(2) {
        vortex_bail!(InvalidArgument:
            "codes buffer length ({}) must be a multiple of 2 (u16 tokens)",
            codes.len()
        );
    }
    if dict_offsets.len() < 8 || !dict_offsets.len().is_multiple_of(4) {
        vortex_bail!(InvalidArgument:
            "dict_offsets buffer length ({}) must be a multiple of 4 and >= 8",
            dict_offsets.len()
        );
    }
    Ok(())
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
        let uncompressed_lengths = slots[UNCOMPRESSED_LENGTHS_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_err!("OnPairArray uncompressed_lengths slot missing"))?;
        validate_parts(
            dtype,
            &data.dict_offsets,
            &data.codes,
            &data.codes_offsets,
            uncompressed_lengths,
            data.bits,
        )?;
        if uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have same len as outer array");
        }
        if data.len != len {
            vortex_bail!(InvalidArgument: "OnPairData len {} != outer len {}", data.len, len);
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        4
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            DICT_BYTES_BUF => array.dict_bytes_handle().clone(),
            DICT_OFFSETS_BUF => array.dict_offsets_handle().clone(),
            CODES_BUF => array.codes_handle().clone(),
            CODES_OFFSETS_BUF => array.codes_offsets_handle().clone(),
            _ => vortex_panic!("OnPairArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            DICT_BYTES_BUF => Some("dict_bytes".to_string()),
            DICT_OFFSETS_BUF => Some("dict_offsets".to_string()),
            CODES_BUF => Some("codes".to_string()),
            CODES_OFFSETS_BUF => Some("codes_offsets".to_string()),
            _ => vortex_panic!("OnPairArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            OnPairMetadata {
                uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
                bits: array.bits(),
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
        if buffers.len() != 4 {
            vortex_bail!(InvalidArgument: "Expected 4 buffers, got {}", buffers.len());
        }
        let metadata = OnPairMetadata::decode(metadata)?;
        let uncompressed_ptype = metadata.get_uncompressed_lengths_ptype()?;

        let uncompressed_lengths = children.get(
            0,
            &DType::Primitive(uncompressed_ptype, Nullability::NonNullable),
            len,
        )?;
        let validity = match children.len() {
            1 => Validity::from(dtype.nullability()),
            2 => Validity::Array(children.get(1, &Validity::DTYPE, len)?),
            other => vortex_bail!(InvalidArgument: "Expected 1 or 2 children, got {other}"),
        };

        let data = OnPairData::new(
            buffers[DICT_BYTES_BUF].clone(),
            buffers[DICT_OFFSETS_BUF].clone(),
            buffers[CODES_BUF].clone(),
            buffers[CODES_OFFSETS_BUF].clone(),
            metadata.bits,
            len,
        );
        let slots: ArraySlots = smallvec![
            Some(uncompressed_lengths),
            validity_to_child(&validity, len),
        ];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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
            builder.extend_from_array(
                &array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array(),
            );
            return Ok(());
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

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
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
            array.slots()[VALIDITY_SLOT].as_ref(),
            array.dtype().nullability(),
        ))
    }
}

/// Convenience extension trait. Slot accessors live here; everything reachable
/// through `OnPairData` is available via `ArrayView -> Deref -> OnPairData`.
pub trait OnPairArrayExt: TypedArrayRef<OnPair> {
    fn uncompressed_lengths(&self) -> &ArrayRef {
        self.as_ref().slots()[UNCOMPRESSED_LENGTHS_SLOT]
            .as_ref()
            .unwrap_or_else(|| vortex_panic!("OnPairArray uncompressed_lengths slot missing"))
    }
    fn array_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[VALIDITY_SLOT].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: TypedArrayRef<OnPair>> OnPairArrayExt for T {}
