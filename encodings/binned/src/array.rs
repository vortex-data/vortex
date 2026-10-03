// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.binned` array encoding.
//!
//! Unlike `vortex.pco`, null rows keep their positions (filled with the previous valid value, so
//! they cost almost nothing), so row `i` is always value `i` and a scalar read decodes one
//! block. The decode tables are built once per array, on first use, and shared by every read.

use std::any::Any;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use prost::Message;
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
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
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
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::modes::Decoder;
use crate::modes::Encoded;
use crate::modes::Numeric;
use crate::modes::compress;
use crate::serde::BinnedMetadata;
use crate::stream::Config;

/// A [`Binned`]-encoded Vortex array.
pub type BinnedArray = Array<Binned>;

/// Binned array encoding marker.
#[derive(Clone, Debug)]
pub struct Binned;

/// A typed [`Decoder`], erased so one cache slot serves every ptype.
type CachedDecoder = Arc<dyn Any + Send + Sync>;

/// The lazily built decoder, shared by slices of the same encoded data.
#[derive(Default)]
struct DecoderCache(OnceLock<CachedDecoder>);

impl std::fmt::Debug for DecoderCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.get().is_some() { "DecoderCache(built)" } else { "DecoderCache" })
    }
}

/// Encoding-specific data for a [`BinnedArray`].
#[derive(Clone, Debug)]
pub struct BinnedData {
    ptype: PType,
    metadata: BinnedMetadata,
    buffers: Vec<ByteBuffer>,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
    decoder: Arc<DecoderCache>,
}

impl Display for BinnedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ptype: {}, nrows: {}, slice: {}..{}",
            self.ptype, self.unsliced_n_rows, self.slice_start, self.slice_stop
        )
    }
}

impl ArrayHash for BinnedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.unsliced_n_rows.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
        self.metadata.encode_to_vec().hash(state);
        for buffer in &self.buffers {
            buffer.array_hash(state, accuracy);
        }
    }
}

impl ArrayEq for BinnedData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.unsliced_n_rows == other.unsliced_n_rows
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
            && self.metadata == other.metadata
            && self.buffers.len() == other.buffers.len()
            && self
                .buffers
                .iter()
                .zip(&other.buffers)
                .all(|(a, b)| a.array_eq(b, accuracy))
    }
}

/// Replaces null rows with the previous valid value (the first valid value for leading nulls),
/// which the codec stores almost for free while keeping every row in place.
fn fill_nulls<T: NativePType>(values: &[T], validity: &Validity, len: usize, ctx: &mut ExecutionCtx) -> VortexResult<Vec<T>> {
    let mut out = values.to_vec();
    if validity.definitely_no_nulls() {
        return Ok(out);
    }
    let mask = validity.execute_mask(len, ctx)?;
    let Some(first_valid) = (0..len).find(|&i| mask.value(i)) else {
        // All null: any constant will do.
        out.fill(T::default());
        return Ok(out);
    };
    let mut last = values[first_valid];
    for (i, v) in out.iter_mut().enumerate() {
        if mask.value(i) {
            last = *v;
        } else {
            *v = last;
        }
    }
    Ok(out)
}

impl BinnedData {
    /// Compresses the values of a primitive array; validity is kept by the caller.
    pub fn from_primitive(
        parray: ArrayView<'_, Primitive>,
        config: &Config,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let ptype = parray.ptype();
        let len = parray.len();
        let validity = parray.validity()?;
        let (metadata, buffers) = match_each_native_ptype!(ptype, |T| {
            let values = fill_nulls(parray.as_slice::<T>(), &validity, len, ctx)?;
            compress::<T>(&values, config)?.to_parts()
        });
        Ok(Self {
            ptype,
            metadata,
            buffers,
            unsliced_n_rows: len,
            slice_start: 0,
            slice_stop: len,
            decoder: Arc::default(),
        })
    }

    /// Validates dtype, validity and slice invariants. Stream contents are validated when the
    /// decoder is first built.
    pub fn validate(&self, dtype: &DType, len: usize, validity: &Validity) -> VortexResult<()> {
        vortex_ensure!(
            dtype.as_ptype() == self.ptype,
            "expected ptype {}, got {}",
            self.ptype,
            dtype.as_ptype()
        );
        vortex_ensure!(
            dtype.nullability() == validity.nullability(),
            "expected nullability {}, got {}",
            validity.nullability(),
            dtype.nullability()
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop && self.slice_stop <= self.unsliced_n_rows,
            "invalid slice range {}..{} for {} rows",
            self.slice_start,
            self.slice_stop,
            self.unsliced_n_rows
        );
        vortex_ensure!(
            self.slice_stop - self.slice_start == len,
            "expected len {len}, got {}",
            self.slice_stop - self.slice_start
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_n_rows,
                "expected validity len {}, got {validity_len}",
                self.unsliced_n_rows
            );
        }
        Ok(())
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.slice_stop == self.slice_start
    }

    /// The encoded size in bytes: metadata plus buffers.
    pub fn nbytes(&self) -> usize {
        self.metadata.encoded_len() + self.buffers.iter().map(ByteBuffer::len).sum::<usize>()
    }

    fn sliced(&self, start: usize, stop: usize) -> Self {
        Self {
            slice_start: self.slice_start + start,
            slice_stop: self.slice_start + stop,
            ..self.clone()
        }
    }

    /// The decoder for this array's encoded data, built on first use.
    fn decoder<T: Numeric>(&self) -> VortexResult<Arc<Decoder<T>>> {
        if let Some(cached) = self.decoder.0.get() {
            return Arc::clone(cached)
                .downcast::<Decoder<T>>()
                .map_err(|_| vortex_err!("cached binned decoder has the wrong type"));
        }
        let encoded = Encoded::<T>::from_parts(self.unsliced_n_rows, &self.metadata, &self.buffers)?;
        let decoder: CachedDecoder = Arc::new(Decoder::new(Arc::new(encoded))?);
        // A concurrent first use may have won the race; either decoder is equivalent.
        let cached = self.decoder.0.get_or_init(|| decoder);
        Arc::clone(cached)
            .downcast::<Decoder<T>>()
            .map_err(|_| vortex_err!("cached binned decoder has the wrong type"))
    }

    fn decompress(&self, validity: &Validity) -> VortexResult<PrimitiveArray> {
        let values = match_each_native_ptype!(self.ptype, |T| {
            let decoder = self.decoder::<T>()?;
            Buffer::<T>::from(decoder.decode_range(self.slice_start, self.slice_stop))
                .into_byte_buffer()
        });
        Ok(PrimitiveArray::from_byte_buffer(
            values,
            self.ptype,
            validity.slice(self.slice_start..self.slice_stop)?,
        ))
    }
}

impl VTable for Binned {
    type TypedArrayData = BinnedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.binned");
        *ID
    }

    fn validate(
        &self,
        data: &BinnedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity = child_to_validity(
            BinnedSlotsView::from_slots(slots).validity,
            dtype.nullability(),
        );
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.buffers.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        BufferHandle::new_host(array.buffers[idx].clone())
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        let stream = ["primary", "secondary", "lookbacks"].get(idx / 2)?;
        let part = if idx % 2 == 0 { "index" } else { "data" };
        Some(format!("{stream}_{part}"))
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        vortex_ensure!(buffers.len() == data.buffers.len(), "wrong number of buffers");
        data.buffers = buffers
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;
        data.decoder = Arc::default();
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_ensure!(
            array.slice_start == 0 && array.slice_stop == array.unsliced_n_rows,
            "sliced binned arrays must be re-encoded before serialization"
        );
        Ok(Some(array.metadata.encode_to_vec()))
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
        let metadata = BinnedMetadata::decode(metadata)?;
        let validity = match children.len() {
            0 => Validity::from(dtype.nullability()),
            1 => Validity::Array(children.get(0, &Validity::DTYPE, len)?),
            n => vortex_bail!("BinnedArray expected 0 or 1 child, got {n}"),
        };
        let buffers = buffers
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;
        let data = BinnedData {
            ptype: dtype.as_ptype(),
            metadata,
            buffers,
            unsliced_n_rows: len,
            slice_start: 0,
            slice_stop: len,
            decoder: Arc::default(),
        };
        let slots = BinnedSlots {
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        BinnedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let validity = array.unsliced_validity();
        Ok(ExecutionResult::done(
            array.data().decompress(&validity)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

impl Binned {
    /// Constructs a binned array after validating its data and validity invariants.
    pub fn try_new(dtype: DType, data: BinnedData, validity: Validity) -> VortexResult<BinnedArray> {
        let len = data.len();
        data.validate(&dtype, len, &validity)?;
        let slots = BinnedSlots {
            validity: validity_to_child(&validity, data.unsliced_n_rows),
        }
        .into_slots();
        Array::try_from_parts(ArrayParts::new(Binned, dtype, len, data).with_slots(slots))
    }

    /// Compresses a primitive array.
    pub fn from_primitive(
        parray: ArrayView<'_, Primitive>,
        config: &Config,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BinnedArray> {
        let dtype = parray.dtype().clone();
        let validity = parray.validity()?;
        let data = BinnedData::from_primitive(parray, config, ctx)?;
        Self::try_new(dtype, data, validity)
    }
}

#[array_slots(Binned)]
pub struct BinnedSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity: Option<ArrayRef>,
}

/// Additional typed accessors for binned arrays.
pub trait BinnedArrayExt: BinnedArraySlotsExt {
    /// Reconstruct the unsliced [`Validity`] from the validity slot.
    fn unsliced_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[BinnedSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}
impl<T: TypedArrayRef<Binned>> BinnedArrayExt for T {}

impl ValidityVTable<Binned> for Binned {
    fn validity(array: ArrayView<'_, Binned>) -> VortexResult<Validity> {
        array
            .unsliced_validity()
            .slice(array.slice_start..array.slice_stop)
    }
}

impl OperationsVTable<Binned> for Binned {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Binned>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let row = array.slice_start + index;
        Ok(match_each_native_ptype!(array.ptype, |T| {
            Scalar::primitive(
                array.decoder::<T>()?.get(row),
                array.dtype().nullability(),
            )
        }))
    }
}

pub(crate) fn slice(array: ArrayView<'_, Binned>, start: usize, stop: usize) -> VortexResult<BinnedArray> {
    Binned::try_new(
        array.dtype().clone(),
        array.data().sliced(start, stop),
        array.unsliced_validity(),
    )
}
