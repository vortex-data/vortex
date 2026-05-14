// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::TypedArrayRef;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::PcoDictMetadata;

/// A [`PcoDict`]-encoded Vortex array.
///
/// Implements pco's `Dict` mode: every input value is represented as an
/// index into a small dictionary of unique values, and the decoded array
/// satisfies `out[i] = dict[indices[i]]`. The dictionary is stored in a
/// single contiguous buffer of raw native bytes (`dict_len *
/// size_of::<T>()` bytes) and the indices live in a `Primitive<L_idx>`
/// child whose width is the narrowest of `u8`/`u16`/`u32` that can
/// represent every index.
///
/// Only integer primitives are supported in this phase; float dicts are
/// deferred to a follow-up because of NaN bit-equality issues.
pub type PcoDictArray = Array<PcoDict>;

/// Slot holding the bitpacked dictionary indices.
pub(crate) const INDICES_SLOT: usize = 0;
const NUM_SLOTS: usize = 1;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["indices"];
const NUM_BUFFERS: usize = 1;
const DICT_BUFFER_NAME: &str = "dict";

/// Marker type implementing [`VTable`] for the [`PcoDict`] mode array.
///
/// PcoDict carries the dictionary itself, the dictionary's cardinality, and
/// the index byte-width at construction time. The element dtype is the
/// parent array's logical dtype.
#[derive(Clone, Debug)]
pub struct PcoDict;

/// Per-array data for [`PcoDictArray`]. Carries the raw dictionary bytes,
/// the dictionary cardinality, and the index byte-width.
#[derive(Clone, Debug)]
pub struct PcoDictData {
    /// Dictionary entries as raw native-LE bytes of the parent ptype.
    dict: ByteBuffer,
    /// Number of distinct dictionary entries (`dict.len() / ptype.byte_width()`).
    dict_len: u32,
    /// Byte-width of each entry in the `indices` child (`1`, `2`, or `4`).
    idx_width: u32,
}

impl PcoDictData {
    /// Construct dict data from already-validated parts.
    pub(crate) fn new(dict: ByteBuffer, dict_len: u32, idx_width: u32) -> Self {
        Self {
            dict,
            dict_len,
            idx_width,
        }
    }

    /// Returns the dictionary bytes (raw little-endian native).
    pub fn dict_bytes(&self) -> &ByteBuffer {
        &self.dict
    }

    /// Returns the number of dictionary entries.
    pub fn dict_len(&self) -> u32 {
        self.dict_len
    }

    /// Returns the byte-width of each value in the indices child.
    pub fn idx_width(&self) -> u32 {
        self.idx_width
    }
}

impl Display for PcoDictData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dict_len: {}, idx_width: {}",
            self.dict_len, self.idx_width
        )
    }
}

impl ArrayHash for PcoDictData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.dict_len.hash(state);
        self.idx_width.hash(state);
        self.dict.as_slice().hash(state);
    }
}

impl ArrayEq for PcoDictData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.dict_len == other.dict_len
            && self.idx_width == other.idx_width
            && self.dict.as_slice() == other.dict.as_slice()
    }
}

impl VTable for PcoDict {
    type TypedArrayData = PcoDictData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.pco_dict");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let indices = slots[INDICES_SLOT]
            .as_ref()
            .vortex_expect("PcoDictArray indices slot");
        validate_parts(data, dtype, indices, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        NUM_BUFFERS
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.data().dict.clone()),
            _ => vortex_panic!("PcoDictArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some(DICT_BUFFER_NAME.to_string()),
            _ => vortex_panic!("PcoDictArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            PcoDictMetadata {
                dict_len: array.data().dict_len,
                idx_width: array.data().idx_width,
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
        let metadata = PcoDictMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        if buffers.len() != NUM_BUFFERS {
            vortex_bail!("Expected {NUM_BUFFERS} buffers, got {}", buffers.len());
        }

        let ptype = PType::try_from(dtype)?;
        ensure_integer_dtype(ptype)?;
        let idx_ptype = idx_ptype_from_width(metadata.idx_width)?;

        let indices_dtype = DType::Primitive(idx_ptype, dtype.nullability());
        let indices = children.get(INDICES_SLOT, &indices_dtype, len)?;

        let dict = buffers[0].clone().try_to_host_sync()?;
        let data = PcoDictData::new(dict, metadata.dict_len, metadata.idx_width);
        validate_parts(&data, dtype, &indices, len)?;
        let slots = smallvec![Some(indices)];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let ptype = PType::try_from(array.dtype())?;
        let dict = array.data().dict.clone();
        let idx_width = array.data().idx_width;
        let indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(ExecutionResult::done(
            decode_primitive(indices, &dict, ptype, idx_width)?.into_array(),
        ))
    }
}

/// Map an index byte-width (`1`, `2`, `4`) to its `PType`.
fn idx_ptype_from_width(idx_width: u32) -> VortexResult<PType> {
    match idx_width {
        1 => Ok(PType::U8),
        2 => Ok(PType::U16),
        4 => Ok(PType::U32),
        other => vortex_bail!("PcoDict idx_width must be 1, 2, or 4, got {other}"),
    }
}

/// Choose the narrowest index ptype that can address `dict_len` entries.
fn choose_idx_ptype(dict_len: usize) -> VortexResult<PType> {
    if dict_len <= u8::MAX as usize + 1 {
        Ok(PType::U8)
    } else if dict_len <= u16::MAX as usize + 1 {
        Ok(PType::U16)
    } else if dict_len <= u32::MAX as usize {
        Ok(PType::U32)
    } else {
        vortex_bail!("PcoDict dictionary cardinality {dict_len} exceeds u32::MAX")
    }
}

/// Ensure the parent dtype is one of the eight integer primitives PcoDict
/// supports.
fn ensure_integer_dtype(ptype: PType) -> VortexResult<()> {
    if !ptype.is_int() {
        vortex_bail!(
            "PcoDictArray only supports integer primitives, got {}",
            ptype
        );
    }
    Ok(())
}

/// Validate that the `indices` child has the expected primitive dtype and
/// length and that `data` is internally consistent with `dtype`.
fn validate_parts(
    data: &PcoDictData,
    dtype: &DType,
    indices: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    ensure_integer_dtype(ptype)?;
    let idx_ptype = idx_ptype_from_width(data.idx_width)?;
    let expected_indices_dtype = DType::Primitive(idx_ptype, dtype.nullability());
    vortex_ensure!(
        indices.dtype() == &expected_indices_dtype,
        "PcoDictArray indices dtype {} does not match expected {}",
        indices.dtype(),
        expected_indices_dtype,
    );
    vortex_ensure!(
        indices.len() == len,
        "PcoDictArray indices len {} does not match array len {len}",
        indices.len(),
    );
    let expected_dict_bytes = (data.dict_len as usize) * ptype.byte_width();
    vortex_ensure!(
        data.dict.len() == expected_dict_bytes,
        "PcoDictArray dict buffer is {} bytes, expected {expected_dict_bytes} \
         ({} entries of {} bytes)",
        data.dict.len(),
        data.dict_len,
        ptype.byte_width(),
    );
    Ok(())
}

/// Extension methods on any typed reference to a [`PcoDictArray`].
pub trait PcoDictArrayExt: TypedArrayRef<PcoDict> {
    /// The dictionary bytes (`dict_len * size_of::<T>()` bytes).
    fn dict_bytes(&self) -> &ByteBuffer {
        PcoDictData::dict_bytes(self)
    }

    /// Number of distinct dictionary entries.
    fn dict_len(&self) -> u32 {
        PcoDictData::dict_len(self)
    }

    /// Byte-width of each value in the `indices` child (`1`, `2`, or `4`).
    fn idx_width(&self) -> u32 {
        PcoDictData::idx_width(self)
    }

    /// The bitpacked indices child.
    fn indices(&self) -> &ArrayRef {
        self.as_ref().slots()[INDICES_SLOT]
            .as_ref()
            .vortex_expect("PcoDictArray indices slot")
    }

    /// Parent [`PType`] of the array.
    fn ptype(&self) -> PType {
        PType::try_from(self.as_ref().dtype()).vortex_expect("PcoDictArray dtype")
    }
}

impl<T: TypedArrayRef<PcoDict>> PcoDictArrayExt for T {}

impl PcoDict {
    /// Construct a [`PcoDictArray`] from a validated dictionary buffer and an
    /// `indices` child.
    ///
    /// The dtype of the returned array is `dtype`; the indices child must be
    /// `Primitive<L_idx>` for `L_idx ∈ {u8, u16, u32}` matching `idx_width`,
    /// and its nullability must match the parent's. Validity flows from the
    /// indices child via [`ValidityVTableFromChild`].
    pub fn try_new(
        dtype: DType,
        dict: ByteBuffer,
        dict_len: u32,
        idx_width: u32,
        indices: ArrayRef,
    ) -> VortexResult<PcoDictArray> {
        let len = indices.len();
        let data = PcoDictData::new(dict, dict_len, idx_width);
        validate_parts(&data, &dtype, &indices, len)?;
        let slots = smallvec![Some(indices)];
        // SAFETY: validate_parts above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(PcoDict, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode an integer primitive array as PcoDict.
    ///
    /// Builds the dictionary by stable first-occurrence order. The index
    /// width is chosen automatically: `u8` if `dict.len() <= 256`, `u16` if
    /// `<= 65_536`, else `u32`. Returns an error if the dictionary would
    /// exceed `u32::MAX` distinct entries (in practice, Dict mode is only
    /// useful at very low cardinality).
    ///
    /// Nulls in the input do not enter the dictionary. The encoder produces
    /// indices whose validity matches input validity; null positions carry
    /// an arbitrary-but-valid index value (`0` when the dictionary is
    /// non-empty, `0` for empty input too — the validity bitmap is what
    /// matters).
    ///
    /// Floats are rejected with a clear error.
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PcoDictArray> {
        let ptype = PrimitiveArrayExt::ptype(&parray);
        ensure_integer_dtype(ptype)?;

        let dtype = parray.dtype().clone();
        let len = parray.array().len();
        let parray = parray.into_owned();
        let validity = PrimitiveArrayExt::validity(&parray);
        let mask = validity.execute_mask(len, ctx)?;

        match_each_integer_ptype!(ptype, |T| {
            encode_typed::<T>(parray.into_buffer::<T>(), validity, &mask, dtype)
        })
    }
}

/// Type-specific encode: walks `values`, building a first-occurrence dict
/// keyed by the bit-equal native representation, then assigns indices.
fn encode_typed<T>(
    values: Buffer<T>,
    validity: Validity,
    mask: &Mask,
    dtype: DType,
) -> VortexResult<PcoDictArray>
where
    T: NativePType + Hash + Eq + DictKey,
{
    let slice = values.as_slice();

    // First pass: build the dictionary. Skip nulls.
    let mut dict_vec: Vec<T> = Vec::new();
    let mut dict_map: HashMap<T::Key, u32> = HashMap::new();
    for (i, v) in slice.iter().enumerate() {
        if !mask.value(i) {
            continue;
        }
        let key = T::dict_key(*v);
        if let Entry::Vacant(e) = dict_map.entry(key) {
            let idx = u32::try_from(dict_vec.len())
                .map_err(|_| vortex_error::vortex_err!("PcoDict dictionary exceeds u32::MAX"))?;
            e.insert(idx);
            dict_vec.push(*v);
        }
    }
    let dict_len = dict_vec.len();
    let idx_ptype = choose_idx_ptype(dict_len)?;

    // Second pass: emit indices.
    let indices_array = match idx_ptype {
        PType::U8 => build_indices::<T, u8>(slice, &dict_map, mask, validity)?,
        PType::U16 => build_indices::<T, u16>(slice, &dict_map, mask, validity)?,
        PType::U32 => build_indices::<T, u32>(slice, &dict_map, mask, validity)?,
        other => vortex_bail!("unexpected idx ptype {other}"),
    };

    let dict_bytes = Buffer::<T>::from(dict_vec).into_byte_buffer();
    let dict_len_u32 = u32::try_from(dict_len)
        .map_err(|_| vortex_error::vortex_err!("PcoDict dictionary exceeds u32::MAX"))?;
    let idx_width =
        u32::try_from(idx_ptype.byte_width()).vortex_expect("idx_ptype byte_width fits in u32");

    PcoDict::try_new(dtype, dict_bytes, dict_len_u32, idx_width, indices_array)
}

/// Build the indices child of native type `I` from a first-occurrence map.
fn build_indices<T, I>(
    values: &[T],
    dict_map: &HashMap<T::Key, u32>,
    mask: &Mask,
    validity: Validity,
) -> VortexResult<ArrayRef>
where
    T: NativePType + Hash + Eq + DictKey,
    I: NativePType + TryFrom<u32>,
{
    let n = values.len();
    let mut out = BufferMut::<I>::with_capacity(n);
    let zero = I::try_from(0u32)
        .ok()
        .vortex_expect("0 fits in unsigned index ptype");
    for (i, v) in values.iter().enumerate() {
        if !mask.value(i) {
            out.push(zero);
            continue;
        }
        let key = T::dict_key(*v);
        let idx_u32 = *dict_map
            .get(&key)
            .vortex_expect("every valid value should be in the dict");
        let idx = I::try_from(idx_u32)
            .ok()
            .vortex_expect("dict index fits in chosen idx ptype");
        out.push(idx);
    }
    Ok(PrimitiveArray::new(out.freeze(), validity).into_array())
}

/// Hashable bit-equal key for each supported integer dtype. We use the
/// native value itself, since integer `Eq`/`Hash` are already bit-equal.
trait DictKey: Copy {
    type Key: Hash + Eq + Copy;
    fn dict_key(self) -> Self::Key;
}

macro_rules! impl_dict_key {
    ($t:ty) => {
        impl DictKey for $t {
            type Key = $t;
            #[inline]
            fn dict_key(self) -> Self::Key {
                self
            }
        }
    };
}

impl_dict_key!(u8);
impl_dict_key!(u16);
impl_dict_key!(u32);
impl_dict_key!(u64);
impl_dict_key!(i8);
impl_dict_key!(i16);
impl_dict_key!(i32);
impl_dict_key!(i64);

/// Recompose a primitive array from `indices` and the dictionary bytes.
fn decode_primitive(
    indices: PrimitiveArray,
    dict: &ByteBuffer,
    parent_ptype: PType,
    idx_width: u32,
) -> VortexResult<PrimitiveArray> {
    let validity = PrimitiveArrayExt::validity(&indices);
    let n = indices.len();

    match_each_integer_ptype!(parent_ptype, |T| {
        let dict_buf = Buffer::<T>::from_byte_buffer(dict.clone());
        let dict_slice = dict_buf.as_slice();
        let mut out = BufferMut::<T>::with_capacity(n);
        match idx_width {
            1 => fill_decoded::<T, u8>(&mut out, indices, dict_slice),
            2 => fill_decoded::<T, u16>(&mut out, indices, dict_slice),
            4 => fill_decoded::<T, u32>(&mut out, indices, dict_slice),
            other => vortex_bail!("PcoDict idx_width must be 1, 2, or 4, got {other}"),
        }
        Ok(PrimitiveArray::new(out.freeze(), validity))
    })
}

fn fill_decoded<T, I>(out: &mut BufferMut<T>, indices: PrimitiveArray, dict: &[T])
where
    T: NativePType,
    I: NativePType + IdxToUsize,
{
    let idx_buf = indices.into_buffer::<I>();
    for &idx in idx_buf.as_slice() {
        let idx_usize = I::idx_to_usize(idx);
        // Null positions still hold a valid in-range index (`0` by encode).
        // Out-of-range indices on hostile inputs would corrupt decode, but
        // validate_parts ensures the parts agree and decode trusts that.
        let v = dict[idx_usize];
        out.push(v);
    }
}

/// Loss-free coercion from one of the supported index widths to `usize`. All
/// of `u8`/`u16`/`u32` always fit in `usize` on the platforms Vortex
/// supports.
trait IdxToUsize: Copy {
    fn idx_to_usize(self) -> usize;
}

impl IdxToUsize for u8 {
    #[inline]
    fn idx_to_usize(self) -> usize {
        self as usize
    }
}
impl IdxToUsize for u16 {
    #[inline]
    fn idx_to_usize(self) -> usize {
        self as usize
    }
}
impl IdxToUsize for u32 {
    #[inline]
    fn idx_to_usize(self) -> usize {
        self as usize
    }
}

impl OperationsVTable<PcoDict> for PcoDict {
    fn scalar_at(
        array: ArrayView<'_, PcoDict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let ptype = array.ptype();
        let nullability = array.dtype().nullability();
        let idx_scalar = array.indices().execute_scalar(index, ctx)?;
        if idx_scalar.is_null() {
            return Scalar::try_new(array.dtype().clone(), None);
        }
        let idx_u64 = idx_scalar
            .as_primitive()
            .as_::<u64>()
            .vortex_expect("PcoDict indices scalar must coerce to u64");
        let idx_usize = usize::try_from(idx_u64)
            .vortex_expect("PcoDict index fits in usize on supported platforms");
        let dict = &array.data().dict;
        match_each_integer_ptype!(ptype, |T| {
            let dict_buf = Buffer::<T>::from_byte_buffer(dict.clone());
            let value = dict_buf.as_slice()[idx_usize];
            Ok(Scalar::primitive(value, nullability))
        })
    }
}

impl ValidityChild<PcoDict> for PcoDict {
    fn validity_child(array: ArrayView<'_, PcoDict>) -> ArrayRef {
        array.indices().clone()
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    fn round_trip_per_type<T>(values: Vec<T>, expected_dict_len: u32, expected_idx_width: u32)
    where
        T: NativePType + std::fmt::Debug,
    {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = PcoDict::encode(parray.as_view(), &mut ctx).expect("encode");
        assert_eq!(encoded.dtype(), parray.dtype());
        assert_eq!(encoded.len(), parray.len());
        assert_eq!(encoded.dict_len(), expected_dict_len);
        assert_eq!(encoded.idx_width(), expected_idx_width);
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)
            .expect("decode");
        assert_arrays_eq!(decoded, parray);
    }

    #[rstest]
    #[case::u8(vec![0u8, 1, 2, 3, 4, 5, 6, 7])]
    #[case::u16(vec![0u16, 1, 2, 3, 4, 5, 6, 7])]
    #[case::u32(vec![0u32, 1, 2, 3, 4, 5, 6, 7])]
    #[case::u64(vec![0u64, 1, 2, 3, 4, 5, 6, 7])]
    #[case::i8(vec![0i8, 1, 2, -3, 4, -5, 6, 7])]
    #[case::i16(vec![0i16, 1, 2, -3, 4, -5, 6, 7])]
    #[case::i32(vec![0i32, 1, 2, -3, 4, -5, 6, 7])]
    #[case::i64(vec![0i64, 1, 2, -3, 4, -5, 6, 7])]
    fn round_trip_cycled_8_uniques<T>(#[case] uniques: Vec<T>) -> VortexResult<()>
    where
        T: NativePType + std::fmt::Debug,
    {
        // Cycle eight unique values into a 64-element input. dict_len == 8,
        // idx_width == 1.
        let cycled: Vec<T> = (0..64).map(|i| uniques[i % uniques.len()]).collect();
        round_trip_per_type::<T>(cycled, 8, 1);
        Ok(())
    }

    #[rstest]
    #[case::u8_width(200usize, 1u32)]
    #[case::u16_width(1_000usize, 2u32)]
    #[case::u32_width(100_000usize, 4u32)]
    fn idx_width_selection(
        #[case] n_unique: usize,
        #[case] expected_width: u32,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Distinct u32 values from 0..n_unique; cycle if we want repetition,
        // but cardinality alone is what drives idx_width selection.
        let n_unique_u32 = u32::try_from(n_unique).vortex_expect("test sizes fit in u32");
        let values: Vec<u32> = (0..n_unique_u32).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = PcoDict::encode(parray.as_view(), &mut ctx)?;
        assert_eq!(encoded.dict_len() as usize, n_unique);
        assert_eq!(encoded.idx_width(), expected_width);
        Ok(())
    }

    #[test]
    fn identity_case_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<i64> = (0i64..100).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = PcoDict::encode(parray.as_view(), &mut ctx)?;
        assert_eq!(encoded.dict_len() as usize, values.len());
        let indices = encoded
            .indices()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u8>();
        let expected_indices: Vec<u8> = (0..100u8).collect();
        assert_eq!(indices.as_slice(), &expected_indices[..]);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let uniques: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let values: Vec<i32> = (0..50).map(|i| uniques[i % uniques.len()]).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = PcoDict::encode(parray.as_view(), &mut ctx)?;
        let sliced = encoded.into_array().slice(10..30)?;
        let expected = PrimitiveArray::from_iter(values[10..30].iter().copied());
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn nullable_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let input = PrimitiveArray::new(
            buffer![10i64, 20, 30, 40, 50, 60],
            Validity::from_iter([true, false, true, false, true, true]),
        );
        let encoded = PcoDict::encode(input.as_view(), &mut ctx)?;

        // Dict skips null positions: positions 1 and 3 (values 20 and 40)
        // are null, so the dict contains 10, 30, 50, 60 (in first-occurrence
        // order over valid positions).
        assert_eq!(encoded.dict_len(), 4);

        let s1 = encoded.clone().into_array().execute_scalar(1, &mut ctx)?;
        let s3 = encoded.clone().into_array().execute_scalar(3, &mut ctx)?;
        assert!(s1.is_null());
        assert!(s3.is_null());

        let s0 = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(s0.as_primitive().typed_value::<i64>(), Some(10));
        let s2 = encoded.clone().into_array().execute_scalar(2, &mut ctx)?;
        assert_eq!(s2.as_primitive().typed_value::<i64>(), Some(30));

        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input);
        Ok(())
    }

    #[rstest]
    #[case::f16(PrimitiveArray::from_iter([
        vortex_array::dtype::half::f16::from_f32(1.0),
        vortex_array::dtype::half::f16::from_f32(2.0),
    ]))]
    #[case::f32(PrimitiveArray::from_iter([1.0_f32, 2.0]))]
    #[case::f64(PrimitiveArray::from_iter([1.0_f64, 2.0]))]
    fn rejects_float_input(#[case] parray: PrimitiveArray) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let err = PcoDict::encode(parray.as_view(), &mut ctx);
        assert!(err.is_err(), "expected error for float input, got {err:?}");
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let uniques: Vec<u64> = (0..64).map(|i| 1_000_000u64 + 7 * i).collect();
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let values: Vec<u64> = (0..256)
            .map(|_| uniques[rng.random_range(0..uniques.len())])
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = PcoDict::encode(parray.as_view(), &mut ctx)?;
        let arr = encoded.into_array();

        let decoded = arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u64>();

        let mut idx_rng = SmallRng::seed_from_u64(0xD00D);
        let indices: Vec<usize> = (0..32).map(|_| idx_rng.random_range(0..256)).collect();
        for &i in &indices {
            let scalar = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(scalar, Scalar::from(decoded.as_slice()[i]));
        }
        Ok(())
    }

    #[test]
    fn empty_input_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(Vec::<i64>::new());
        let encoded = PcoDict::encode(parray.as_view(), &mut ctx)?;
        assert_eq!(encoded.len(), 0);
        assert_eq!(encoded.dict_len(), 0);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }
}
