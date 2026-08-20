// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::OnceLock;

use num_traits::AsPrimitive;
use onpair::CompactDictionary;
use onpair::CompactDictionaryView;
use onpair::Dictionary;
use onpair::DictionaryStorage;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::OffsetBuilderPType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_varbin_builder;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::canonical::OnPairDecodePlan;
use crate::canonical::canonicalize_onpair;
use crate::canonical::onpair_decode_bytes;
use crate::decode::collect_widened;
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
    #[prost(uint32, tag = "3")]
    pub dict_size: u32,
    /// Length of the `codes` slot child. A sliced array may retain codes that
    /// fall outside its visible row range.
    #[prost(uint64, tag = "4")]
    pub codes_len: u64,
    /// PType of the `dict_offsets` slot child (defaults to U32, may be
    /// narrowed to U16/U8 by the cascading compressor when values fit).
    #[prost(enumeration = "PType", tag = "5")]
    pub dict_offsets_ptype: i32,
    /// PType of the `codes` slot child.
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
    /// Dictionary-offset child, with length `dict_size + 1`. The cascading
    /// compressor may re-encode this child independently; the materialised
    /// offsets used by the runtime dictionary cache are derived from it on
    /// first use.
    #[slot(0)]
    pub dict_offsets: ArrayRef,
    /// Primitive integer token codes. Downstream integer compression may
    /// narrow or bit-pack this child independently of the OnPair metadata.
    #[slot(1)]
    pub codes: ArrayRef,
    /// Primitive integer row offsets into `codes`, length `num_rows + 1`. The
    /// cascading compressor may re-encode this child independently.
    #[slot(2)]
    pub codes_offsets: ArrayRef,
    /// Integer decoded-length child, length `num_rows`. Used to size the
    /// canonical output buffer.
    #[slot(3)]
    pub uncompressed_lengths: ArrayRef,
    /// Optional validity child for the outer string column.
    #[slot(4)]
    pub validity: Option<ArrayRef>,
}

/// Immutable storage for a materialised OnPair dictionary.
///
/// The two buffers remain owned by Vortex. Implementing [`DictionaryStorage`]
/// this way lets `onpair::CompactDictionary` retain them without copying
/// either the dictionary bytes or the widened offsets.
#[derive(Clone, Debug)]
struct OnPairDictionaryStorage {
    bytes: ByteBuffer,
    offsets: Buffer<u32>,
}

impl DictionaryStorage<u32> for OnPairDictionaryStorage {
    #[inline]
    fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    #[inline]
    fn offsets(&self) -> &[u32] {
        self.offsets.as_slice()
    }
}

/// Non-child data for an OnPair-encoded array.
///
/// The serialized dictionary bytes stay in buffer 0, while the dictionary
/// offsets remain a recursive child so the existing on-disk format is
/// unchanged. Once the offsets are materialised, `dictionary` retains an
/// upstream storage-backed [`CompactDictionary`] around the same immutable
/// buffers. Constructing an array from serialized parts does not materialise or
/// validate the dictionary: Vortex only reconstructs the recursive offset
/// child at that point. Converting that child to contiguous `u32` offsets and
/// validating the dictionary are deferred until the first decode/search
/// operation that needs dictionary token access.
#[derive(Clone)]
pub struct OnPairData {
    /// The dictionary blob (buffer 0).
    ///
    /// INVARIANT: this buffer is an OnPair compact dictionary byte buffer,
    /// including its trailing read padding.
    dict_bytes: BufferHandle,
    /// The storage-backed dictionary, memoized after successful initialization.
    /// Initialization decompresses the child and safety-validates the dictionary;
    /// once cached, later operations do neither again. The dictionary owns
    /// clones of the immutable Vortex buffer handles, so this does not copy the
    /// bytes. A failed validation is not cached, and concurrent first users may
    /// duplicate initialization work before one value wins the `OnceLock`.
    ///
    /// INVARIANT: once populated, the offsets passed
    /// [`CompactDictionary::validate_safety`] against `dict_bytes`.
    /// The `Arc` cell is shared only between arrays with identical dictionary
    /// bytes and logically identical offsets (slice / filter / cast keep both).
    dictionary: Arc<OnceLock<CompactDictionary<OnPairDictionaryStorage>>>,
}

impl OnPairData {
    /// Build [`OnPairData`] from the dictionary blob.
    pub fn new(dict_bytes: BufferHandle) -> Self {
        Self {
            dict_bytes,
            dictionary: Arc::new(OnceLock::new()),
        }
    }

    /// Build [`OnPairData`] with the dictionary already materialised, so the
    /// first decode skips both widening and validation.
    ///
    /// `offsets` must be the widened values of the `dict_offsets` child the
    /// caller attaches to the array; validation proves only that they are
    /// structurally safe against `dict_bytes`.
    pub(crate) fn try_new_with_dictionary(
        dict_bytes: BufferHandle,
        offsets: Buffer<u32>,
    ) -> VortexResult<Self> {
        let dictionary = build_dictionary(dict_bytes.as_host().clone(), offsets)?;
        Ok(Self {
            dict_bytes,
            dictionary: Arc::new(OnceLock::from(dictionary)),
        })
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

/// Safety-validate `(bytes, offsets)` and seal them into a storage-backed
/// dictionary.
fn build_dictionary(
    bytes: ByteBuffer,
    offsets: Buffer<u32>,
) -> VortexResult<CompactDictionary<OnPairDictionaryStorage>> {
    CompactDictionary::validate_safety(OnPairDictionaryStorage { bytes, offsets })
        .map_err(|e| vortex_err!(InvalidArgument: "Unsafe OnPair dictionary: {e}"))
}

/// A safety-validated dictionary built from host-materialized parts, owned
/// independently of any array.
///
/// [`dict_view`] reaches the same parts through host-only accessors, which panics when
/// the array's buffers are device-resident. Callers that have already copied the
/// dictionary blob and its widened offsets to the host — the CUDA executor, which stages
/// the dictionary on the host regardless — build through this instead.
pub struct OnPairDictionary(CompactDictionary<OnPairDictionaryStorage>);

impl OnPairDictionary {
    /// Validates `(bytes, offsets)` and seals them into a dictionary.
    pub fn try_new(bytes: ByteBuffer, offsets: Buffer<u32>) -> VortexResult<Self> {
        Ok(Self(build_dictionary(bytes, offsets)?))
    }

    /// Borrows the dictionary as a view.
    pub fn as_view(&self) -> CompactDictionaryView<'_> {
        self.0.as_view()
    }
}

/// A safety-validated [`CompactDictionaryView`] over `array`'s dictionary.
///
/// The first successful initialization widens the `dict_offsets` child and
/// safety-validates the dictionary structure; the resulting storage-backed
/// dictionary is memoized in [`OnPairData`]. Once cached, subsequent calls —
/// including on arrays derived by slice / filter / cast, which share the cell —
/// pay neither cost again.
pub fn dict_view<'a>(
    array: ArrayView<'a, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<CompactDictionaryView<'a>> {
    let data = array.data();
    let dictionary = match data.dictionary.get() {
        Some(dictionary) => dictionary,
        None => {
            let widened = collect_widened::<u32>(array.dict_offsets(), ctx)?;
            let dictionary = build_dictionary(data.dict_bytes().clone(), widened)?;
            data.dictionary.get_or_init(|| dictionary)
        }
    };
    Ok(dictionary.as_view())
}

impl Display for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dict_bytes_len: {}", self.dict_bytes.len())
    }
}

impl Debug for OnPairData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnPairData")
            .field("dict_bytes_len", &self.dict_bytes.len())
            .finish()
    }
}

impl ArrayHash for OnPairData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.dict_bytes.as_host().array_hash(state, accuracy);
    }
}

impl ArrayEq for OnPairData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.dict_bytes
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
        Self::try_new_with_data(
            dtype,
            OnPairData::new(dict_bytes),
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity,
        )
    }

    /// Build an [`OnPairArray`] from already-materialised parts while reusing
    /// an existing [`OnPairData`].
    ///
    /// Reusing the data preserves the dictionary byte-buffer handle and the
    /// shared lazy dictionary cache. This is useful when recursive compression
    /// replaces the slot children without changing the logical dictionary.
    ///
    /// If `data` contains a memoized dictionary, `dict_offsets` must be
    /// logically equivalent to the offsets used to create that dictionary.
    /// The constructor intentionally does not validate the dictionary itself;
    /// dictionary validation remains lazy and is performed on first use.
    pub fn try_new_with_data(
        dtype: DType,
        data: OnPairData,
        dict_offsets: ArrayRef,
        codes: ArrayRef,
        codes_offsets: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> VortexResult<OnPairArray> {
        validate_parts(
            &dtype,
            &dict_offsets,
            &codes,
            &codes_offsets,
            &uncompressed_lengths,
        )?;
        Ok(unsafe {
            Self::new_unchecked(
                dtype,
                data,
                dict_offsets,
                codes,
                codes_offsets,
                uncompressed_lengths,
                validity,
            )
        })
    }

    /// Build an [`OnPairArray`] without validation, carrying `data` — and with
    /// it the memoized dictionary — from an existing array.
    ///
    /// # Safety
    /// The parts must satisfy the same invariants [`try_new`](Self::try_new)
    /// checks. If `data`'s dictionary cell is populated (or shared with a live
    /// array), `dict_offsets` must hold the same logical offsets the dictionary
    /// was built from.
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        data: OnPairData,
        dict_offsets: ArrayRef,
        codes: ArrayRef,
        codes_offsets: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> OnPairArray {
        let len = uncompressed_lengths.len();
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
) -> VortexResult<()> {
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
    if codes_offsets.len() != uncompressed_lengths.len() + 1 {
        vortex_bail!(InvalidArgument:
            "codes_offsets.len ({}) != uncompressed_lengths.len + 1 ({})",
            codes_offsets.len(),
            uncompressed_lengths.len() + 1
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
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let s = OnPairSlotsView::from_slots(slots);
        validate_parts(
            dtype,
            s.dict_offsets,
            s.codes,
            s.codes_offsets,
            s.uncompressed_lengths,
        )?;
        if s.uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have same len as outer array");
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
        // The replacement blob may differ from the one the memoized dictionary
        // was validated against, so drop the (shared) cell rather than
        // carry a claim we can no longer prove.
        data.dictionary = Arc::new(OnceLock::new());
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
        let codes_len = array.codes().len() as u64;
        Ok(Some(
            OnPairMetadata {
                uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
                dict_size,
                codes_len,
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

        // Slot children do not persist their own lengths, so metadata records
        // the dictionary and code-stream sizes needed to deserialize them.
        let dict_offsets_len = metadata.dict_size as usize + 1;
        let codes_len = usize::try_from(metadata.codes_len)
            .map_err(|_| vortex_err!("codes_len {} overflows usize", metadata.codes_len))?;
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
            codes_len,
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

        let data = OnPairData::new(buffers[0].clone());
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
        if let Some(result) =
            match_each_varbin_builder!(builder, |builder| append_to_varbin(array, builder, ctx))
        {
            return result;
        }

        // The two arms here are every builder a `Utf8`/`Binary` dtype has: all four
        // `VarBinBuilder` widths above, and `VarBinViewBuilder` below. There is deliberately no
        // canonicalize-then-append fallback — it would decode to a `VarBinView` only for
        // `VarBinView::append_to_builder` to reject the same remainder.
        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            vortex_bail!("append_to_builder for OnPair requires a variable-binary builder")
        };

        // Decode the whole code stream into a new buffer, which the builder adopts as a data
        // buffer with views built over it in place.
        let validity = array
            .array()
            .validity()?
            .execute_mask(array.array().len(), ctx)?;
        let (out_bytes, lengths) = onpair_decode_bytes(array, ctx)?;
        match_each_integer_ptype!(lengths.ptype(), |P| {
            builder.append_buffer_with_lengths(
                out_bytes.freeze(),
                lengths.as_slice::<P>(),
                &validity,
            )
        });
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

/// Decodes the code stream straight into `builder`'s byte storage.
///
/// The offsets are the running sum of the uncompressed lengths the array already stores, so the
/// only work beyond the bulk `try_decode_into` is one prefix sum over them.
fn append_to_varbin<O: OffsetBuilderPType>(
    array: ArrayView<'_, OnPair>,
    builder: &mut VarBinBuilder<O>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let plan = OnPairDecodePlan::new(array, ctx)?;
    let validity = array
        .array()
        .validity()?
        .execute_mask(array.array().len(), ctx)?;
    // Built once, outside the ptype match: `append_decoded` takes it as `&mut dyn FnMut`, so
    // creating the closure inside each arm would stamp out a shim per length type for no gain.
    let mut decode = |out: &mut [MaybeUninit<u8>]| plan.decode_into(out);
    match_each_integer_ptype!(plan.lengths.ptype(), |P| {
        // SAFETY: `decode_into` initializes exactly the prefix whose length it returns. It needs
        // no slack: it derives its bound from the slice it is handed and writes each value exactly.
        unsafe {
            builder.append_decoded(
                plan.total_size,
                0,
                plan.lengths.as_slice::<P>(),
                &validity,
                &mut decode,
            )
        }
    })
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
