// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;

use parking_lot::Mutex;
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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_onpair_sys::Column;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::canonical::canonicalize_onpair;
use crate::canonical::onpair_decode_views;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

/// An [`OnPair`]-encoded Vortex array.
pub type OnPairArray = Array<OnPair>;

/// Default bits-per-token preset used by [`OnPair::compress`]: 12-bit codes,
/// dictionary capped at 4 096 entries.
pub const DEFAULT_BITS: u32 = 12;

/// Wire-format metadata persisted alongside the serialised OnPair column.
#[derive(Clone, prost::Message)]
pub struct OnPairMetadata {
    /// Width of the per-row primitive `uncompressed_lengths` child.
    #[prost(enumeration = "PType", tag = "1")]
    pub uncompressed_lengths_ptype: i32,
    /// Bits-per-token the column was compressed with (9..=16).
    #[prost(uint32, tag = "2")]
    pub bits: u32,
    /// Number of dictionary entries.
    #[prost(uint64, tag = "3")]
    pub dict_size: u64,
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

/// Inner data for an OnPair-encoded array.
///
/// Holds an owning handle over the C++ `OnPairColumn` and the serialised
/// bytes used both for persistence and for cheap clones (the column itself is
/// reconstructed lazily on the receiving side). The codes/dictionary are
/// stored inside the C++ object; on disk they live as a single opaque buffer.
#[derive(Clone)]
pub struct OnPairData {
    /// The opaque `ONPAIR01`-prefixed serialised column bytes. This is the
    /// single Vortex buffer at index 0.
    column_bytes: BufferHandle,
    /// Lazily reconstituted C++ column. Wrapped in an `Arc<Mutex<_>>` so that
    /// cloning the array is cheap and the C++ object is only built once.
    column: Arc<Mutex<Option<Arc<Column>>>>,
    /// Cached length.
    len: usize,
    /// Bits-per-token (mirrors what the C++ side stores).
    bits: u32,
    /// Cached dictionary size.
    dict_size: usize,
}

impl OnPairData {
    /// Build [`OnPairData`] from an in-memory [`Column`] plus its serialised bytes.
    /// The bytes are required so the array can be persisted without re-serialising.
    pub fn from_column(column: Column, column_bytes: BufferHandle) -> Self {
        let len = column.len();
        let bits = column.bits();
        let dict_size = column.dict_size();
        Self {
            column_bytes,
            column: Arc::new(Mutex::new(Some(Arc::new(column)))),
            len,
            bits,
            dict_size,
        }
    }

    /// Lazy-construct path used on deserialise. The C++ column is only built
    /// the first time it is needed (e.g. on canonicalisation or predicate
    /// pushdown), keeping clone-only paths cheap.
    pub fn from_bytes(column_bytes: BufferHandle, len: usize, bits: u32, dict_size: usize) -> Self {
        Self {
            column_bytes,
            column: Arc::new(Mutex::new(None)),
            len,
            bits,
            dict_size,
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

    pub fn dict_size(&self) -> usize {
        self.dict_size
    }

    pub fn column_bytes(&self) -> &ByteBuffer {
        self.column_bytes.as_host()
    }

    pub fn column_bytes_handle(&self) -> &BufferHandle {
        &self.column_bytes
    }

    /// Materialise the C++ column on demand.
    pub fn column(&self) -> VortexResult<Arc<Column>> {
        let mut slot = self.column.lock();
        if let Some(c) = slot.as_ref() {
            return Ok(Arc::clone(c));
        }
        let bytes = self.column_bytes.as_host();
        let column = Column::from_bytes(bytes.as_slice())
            .map_err(|e| vortex_err!("Failed to materialise OnPair column: {e}"))?;
        let column = Arc::new(column);
        *slot = Some(Arc::clone(&column));
        Ok(column)
    }
}

impl Display for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "len: {}, bits: {}, dict_size: {}",
            self.len, self.bits, self.dict_size
        )
    }
}

impl Debug for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnPairData")
            .field("len", &self.len)
            .field("bits", &self.bits)
            .field("dict_size", &self.dict_size)
            .field("column_bytes_len", &self.column_bytes.len())
            .finish()
    }
}

impl ArrayHash for OnPairData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        // The serialised column is canonical for a given input + config; hashing
        // the bytes is sufficient and avoids reaching into the C++ side.
        self.column_bytes.as_host().array_hash(state, precision);
        state.write_u32(self.bits);
    }
}

impl ArrayEq for OnPairData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.bits == other.bits
            && self
                .column_bytes
                .as_host()
                .array_eq(other.column_bytes.as_host(), precision)
    }
}

/// Zero-sized VTable marker for the OnPair encoding.
#[derive(Clone, Debug)]
pub struct OnPair;

impl OnPair {
    /// Build an [`OnPairArray`] from an in-memory [`Column`] and its
    /// previously-serialised bytes.
    pub fn try_new(
        dtype: DType,
        column: Column,
        column_bytes: BufferHandle,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> VortexResult<OnPairArray> {
        validate_outer(&dtype, &uncompressed_lengths, column.len())?;
        let len = column.len();
        let data = OnPairData::from_column(column, column_bytes);
        let slots: ArraySlots = smallvec![
            Some(uncompressed_lengths),
            validity_to_child(&validity, len),
        ];
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(OnPair, dtype, len, data).with_slots(slots))
        })
    }

    /// Internal lazy constructor used by [`OnPair::deserialize`].
    pub(crate) unsafe fn new_unchecked_lazy(
        dtype: DType,
        column_bytes: BufferHandle,
        len: usize,
        bits: u32,
        dict_size: usize,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> OnPairArray {
        let data = OnPairData::from_bytes(column_bytes, len, bits, dict_size);
        let slots: ArraySlots = smallvec![
            Some(uncompressed_lengths),
            validity_to_child(&validity, len),
        ];
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(OnPair, dtype, len, data).with_slots(slots))
        }
    }
}

fn validate_outer(dtype: &DType, uncompressed_lengths: &ArrayRef, len: usize) -> VortexResult<()> {
    vortex_ensure!(
        matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
        "OnPair arrays must be Binary or Utf8, found {dtype}"
    );
    vortex_ensure!(
        uncompressed_lengths.len() == len,
        InvalidArgument: "uncompressed_lengths must have same len as OnPair array"
    );
    vortex_ensure!(
        uncompressed_lengths.dtype().is_int() && !uncompressed_lengths.dtype().is_nullable(),
        InvalidArgument: "uncompressed_lengths must be non-nullable integer, found {}",
        uncompressed_lengths.dtype()
    );
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
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "OnPair arrays must be Binary or Utf8, found {dtype}"
        );
        let uncompressed_lengths = slots[UNCOMPRESSED_LENGTHS_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_err!("OnPairArray uncompressed_lengths slot missing"))?;
        if uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have same len as OnPair array");
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
            0 => array.column_bytes_handle().clone(),
            _ => vortex_panic!("OnPairArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("onpair_column".to_string()),
            _ => vortex_panic!("OnPairArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            OnPairMetadata {
                uncompressed_lengths_ptype: uncompressed_lengths_from_slots(array.slots())
                    .dtype()
                    .as_ptype()
                    .into(),
                bits: array.bits(),
                dict_size: array.dict_size() as u64,
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
        let uncompressed_lengths = children.get(
            0,
            &DType::Primitive(
                metadata.get_uncompressed_lengths_ptype()?,
                Nullability::NonNullable,
            ),
            len,
        )?;
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            Validity::Array(children.get(1, &Validity::DTYPE, len)?)
        } else {
            vortex_bail!(InvalidArgument: "Expected 1 or 2 children, got {}", children.len());
        };

        let dict_size = usize::try_from(metadata.dict_size)
            .map_err(|_| vortex_err!("dict_size {} too large for usize", metadata.dict_size))?;
        let data = OnPairData::from_bytes(buffers[0].clone(), len, metadata.bits, dict_size);
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

fn uncompressed_lengths_from_slots(slots: &[Option<ArrayRef>]) -> &ArrayRef {
    slots[UNCOMPRESSED_LENGTHS_SLOT]
        .as_ref()
        .vortex_expect("OnPairArray uncompressed_lengths slot")
}

/// Convenience extension trait, mirroring `FSSTArrayExt`. Only carries methods
/// that need slot lookups; the rest are accessed via the `ArrayView` →
/// `OnPairData` `Deref` chain.
pub trait OnPairArrayExt: TypedArrayRef<OnPair> {
    fn uncompressed_lengths(&self) -> &ArrayRef {
        uncompressed_lengths_from_slots(self.as_ref().slots())
    }

    fn array_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[VALIDITY_SLOT].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: TypedArrayRef<OnPair>> OnPairArrayExt for T {}
