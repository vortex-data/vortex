// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A map encoding that stores each distinct key set once.
//!
//! Observability rows of the same metric or span kind carry the same keys in the same order, so a
//! canonical map repeats every key on every row. [`KeySetMap`] stores the distinct key sets once,
//! one key-set id per row, and only the values per row. Optionally, a row whose values also equal
//! the previous row's reuses its value range, so repeated label maps cost one id and one offset.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::ops::Range;

use prost::Message;

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
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::arrays::dict::TakeReduceAdaptor;
use vortex_array::arrays::filter::FilterReduce;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::decode::codes_u32;
use crate::flat::FlatMap;
use crate::flat::Strings;
use crate::flat::build_listview;
use crate::flat::build_map;
use crate::flat::mask_indices;
use crate::flat::to_usize_vec;
use crate::labels::values_to_utf8;
use crate::rowcmp::RowCmp;

/// A [`KeySetMap`]-encoded Vortex array.
pub type KeySetMapArray = Array<KeySetMap>;

/// Map encoding with one copy of each distinct key set. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct KeySetMap;

#[array_slots(KeySetMap)]
pub struct KeySetMapSlots {
    /// The key set of each row, `u32`, null for a null row.
    #[slot(0)]
    pub keyset_ids: ArrayRef,
    /// The distinct key sets, a list of keys.
    #[slot(1)]
    pub keysets: ArrayRef,
    /// The start of each row's values in `values`, `u64`.
    #[slot(2)]
    pub value_offsets: ArrayRef,
    /// The values, in the order of their row's key set.
    #[slot(3)]
    pub values: ArrayRef,
}

/// [`KeySetMap`] has no metadata beyond its dtype and children.
#[derive(Clone, Debug, Default)]
pub struct KeySetMapData;

impl Display for KeySetMapData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArrayEq for KeySetMapData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl ArrayHash for KeySetMapData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

#[derive(Clone, prost::Message)]
struct KeySetMapMetadata {
    #[prost(uint64, tag = "1")]
    keysets_len: u64,
    #[prost(uint64, tag = "2")]
    values_len: u64,
}

fn ids_dtype(nullability: Nullability) -> DType {
    DType::Primitive(PType::U32, nullability)
}

fn offsets_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

fn keysets_dtype(map_dtype: &MapDType) -> DType {
    DType::List(map_dtype.key_dtype().into(), Nullability::NonNullable)
}

fn make_parts(
    dtype: DType,
    len: usize,
    keyset_ids: ArrayRef,
    keysets: ArrayRef,
    value_offsets: ArrayRef,
    values: ArrayRef,
) -> ArrayParts<KeySetMap> {
    ArrayParts::new(KeySetMap, dtype, len, KeySetMapData).with_slots(smallvec![
        Some(keyset_ids),
        Some(keysets),
        Some(value_offsets),
        Some(values),
    ])
}

impl KeySetMap {
    /// Constructs a key-set map from its parts.
    ///
    /// # Errors
    ///
    /// Returns an error if the children's dtypes or lengths do not match `dtype`.
    pub fn try_new(
        dtype: DType,
        keyset_ids: ArrayRef,
        keysets: ArrayRef,
        value_offsets: ArrayRef,
        values: ArrayRef,
    ) -> VortexResult<KeySetMapArray> {
        let len = keyset_ids.len();
        Array::try_from_parts(make_parts(
            dtype,
            len,
            keyset_ids,
            keysets,
            value_offsets,
            values,
        ))
    }
}

impl VTable for KeySetMap {
    type TypedArrayData = KeySetMapData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.keyset_map");
        *ID
    }

    fn validate(
        &self,
        _data: &KeySetMapData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Map(map_dtype, nullability) = dtype else {
            vortex_bail!("KeySetMap requires a map dtype, got {dtype}");
        };
        vortex_ensure!(slots.len() == KeySetMapSlots::COUNT, "KeySetMap expects 4 slots");
        let s = KeySetMapSlotsView::from_slots(slots);
        vortex_ensure!(s.keyset_ids.dtype() == &ids_dtype(*nullability), "bad keyset_ids dtype");
        vortex_ensure!(s.keysets.dtype() == &keysets_dtype(map_dtype), "bad keysets dtype");
        vortex_ensure!(s.value_offsets.dtype() == &offsets_dtype(), "bad value_offsets dtype");
        vortex_ensure!(s.values.dtype() == &map_dtype.value_dtype(), "bad values dtype");
        vortex_ensure!(
            s.keyset_ids.len() == len && s.value_offsets.len() == len,
            "KeySetMap row children must have length {len}"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("KeySetMapArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            KeySetMapMetadata {
                keysets_len: array.keysets().len() as u64,
                values_len: array.values().len() as u64,
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
        let DType::Map(map_dtype, nullability) = dtype else {
            vortex_bail!("Expected map dtype, got {dtype}");
        };
        vortex_ensure!(children.len() == KeySetMapSlots::COUNT, "KeySetMap expects 4 children");
        let metadata = KeySetMapMetadata::decode(metadata)
            .map_err(|e| vortex_err!("invalid KeySetMap metadata: {e}"))?;
        let keysets_len = usize::try_from(metadata.keysets_len)?;
        let values_len = usize::try_from(metadata.values_len)?;
        let keyset_ids = children.get(0, &ids_dtype(*nullability), len)?;
        let keysets = children.get(1, &keysets_dtype(map_dtype), keysets_len)?;
        let value_offsets = children.get(2, &offsets_dtype(), len)?;
        let values = children.get(3, &map_dtype.value_dtype(), values_len)?;
        Ok(make_parts(
            dtype.clone(),
            len,
            keyset_ids,
            keysets,
            value_offsets,
            values,
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        KeySetMapSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(decode(&array, ctx)?.into_array()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<KeySetMap> for KeySetMap {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, KeySetMap>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let row = array.array().slice(index..index + 1)?;
        row.execute::<MapArray>(ctx)?.into_array().execute_scalar(0, ctx)
    }
}

impl ValidityChild<KeySetMap> for KeySetMap {
    fn validity_child(array: ArrayView<'_, KeySetMap>) -> ArrayRef {
        array.keyset_ids().clone()
    }
}

static RULES: ParentRuleSet<KeySetMap> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(KeySetMap)),
    ParentRuleSet::lift(&SliceReduceAdaptor(KeySetMap)),
    ParentRuleSet::lift(&TakeReduceAdaptor(KeySetMap)),
]);

/// Applies one row selection to the row-aligned children.
fn select_rows(
    array: ArrayView<'_, KeySetMap>,
    f: impl Fn(&ArrayRef) -> VortexResult<ArrayRef>,
) -> VortexResult<ArrayRef> {
    let keyset_ids = f(array.keyset_ids())?;
    let value_offsets = f(array.value_offsets())?;
    // A null take index nulls the id, while its offset is never read.
    let value_offsets = if value_offsets.dtype().is_nullable() {
        value_offsets.fill_null(Scalar::from(0u64))?
    } else {
        value_offsets
    };
    let dtype = array.dtype().with_nullability(keyset_ids.dtype().nullability());
    Ok(KeySetMap::try_new(
        dtype,
        keyset_ids,
        array.keysets().clone(),
        value_offsets,
        array.values().clone(),
    )?
    .into_array())
}

impl SliceReduce for KeySetMap {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        select_rows(array, |c| c.slice(range.clone())).map(Some)
    }
}

impl TakeReduce for KeySetMap {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        select_rows(array, |c| c.take(indices.clone())).map(Some)
    }
}

impl FilterReduce for KeySetMap {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let indices = mask_indices(mask);
        select_rows(array, |c| c.take(indices.clone())).map(Some)
    }
}

/// Options for [`keyset_encode`].
#[derive(Clone, Copy, Debug)]
pub struct KeySetOptions {
    /// Let a row whose values equal the previous row's reuse its value range.
    pub dedup_values: bool,
}

/// Encodes a map with UTF-8 keys as a [`KeySetMapArray`].
///
/// # Errors
///
/// Returns an error if `map` is not a map with UTF-8 keys.
pub fn keyset_encode(
    map: &ArrayRef,
    options: KeySetOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<KeySetMapArray> {
    let flat = FlatMap::new(map, ctx)?;
    vortex_ensure!(
        matches!(flat.map_dtype.key_dtype(), DType::Utf8(_)),
        "keyset_encode requires UTF-8 keys, got {}",
        map.dtype()
    );
    let keys = Strings::new(&flat.keys);
    let values_cmp = options
        .dedup_values
        .then(|| RowCmp::new(&flat.values, ctx))
        .transpose()?;

    let mut ids_of: HashMap<Vec<&[u8]>, u32> = HashMap::new();
    let mut keyset_entries: Vec<u64> = Vec::new();
    let mut keyset_offsets: Vec<u64> = Vec::new();
    let mut keyset_sizes: Vec<u64> = Vec::new();
    let mut ids = Vec::with_capacity(flat.len);
    let mut valid = BitBufferMut::with_capacity(flat.len);
    let mut value_entries: Vec<u64> = Vec::new();
    let mut value_offsets = Vec::with_capacity(flat.len);
    let mut prev: Option<(Range<usize>, u32, u64)> = None;

    for row in 0..flat.len {
        if !flat.row_valid.value(row) {
            ids.push(0);
            valid.append(false);
            value_offsets.push(0);
            continue;
        }
        let range = flat.range(row);
        let same_keys = prev.as_ref().is_some_and(|(p, ..)| {
            p.len() == range.len() && p.clone().zip(range.clone()).all(|(a, b)| keys.get(a) == keys.get(b))
        });
        let id = match (&prev, same_keys) {
            (Some((_, id, _)), true) => *id,
            _ => {
                let signature: Vec<&[u8]> = range.clone().map(|j| keys.get(j)).collect();
                let next = u32::try_from(keyset_offsets.len())?;
                *ids_of.entry(signature).or_insert_with(|| {
                    keyset_offsets.push(keyset_entries.len() as u64);
                    keyset_sizes.push(range.len() as u64);
                    keyset_entries.extend(range.clone().map(|j| j as u64));
                    next
                })
            }
        };
        let same_values = same_keys
            && values_cmp.as_ref().is_some_and(|cmp| {
                let (p, ..) = prev.as_ref().unwrap_or_else(|| unreachable!());
                p.clone().zip(range.clone()).all(|(a, b)| cmp.equal(a, b))
            });
        let offset = match (&prev, same_values) {
            (Some((.., offset)), true) => *offset,
            _ => {
                let offset = value_entries.len() as u64;
                value_entries.extend(range.clone().map(|j| j as u64));
                offset
            }
        };
        ids.push(id);
        valid.append(true);
        value_offsets.push(offset);
        prev = Some((range, id, offset));
    }

    let take = |array: ArrayRef, idx: Vec<u64>, ctx: &mut ExecutionCtx| -> VortexResult<ArrayRef> {
        let idx = PrimitiveArray::new(Buffer::from(idx), Validity::NonNullable).into_array();
        Ok(array.take(idx)?.execute::<Canonical>(ctx)?.compact(ctx)?.into_array())
    };
    let keyset_keys = take(flat.keys.clone().into_array(), keyset_entries, ctx)?;
    let keysets = build_listview(keyset_keys, keyset_offsets, keyset_sizes, Validity::NonNullable)?;
    let values = take(flat.values.clone(), value_entries, ctx)?;
    let nullability = map.dtype().nullability();
    let ids = PrimitiveArray::new(
        Buffer::from(ids),
        Validity::from_bit_buffer(valid.freeze(), nullability),
    );
    let offsets = PrimitiveArray::new(Buffer::from(value_offsets), Validity::NonNullable);
    KeySetMap::try_new(
        map.dtype().clone(),
        ids.into_array(),
        keysets.into_array(),
        offsets.into_array(),
        values,
    )
}

/// The canonical parts of a key-set map.
struct Parts {
    map_dtype: MapDType,
    ids: Vec<u32>,
    valid: Mask,
    validity: Validity,
    value_offsets: Vec<usize>,
    keysets: ListViewArray,
    keyset_offsets: Vec<usize>,
    keyset_sizes: Vec<usize>,
    values: ArrayRef,
}

impl Parts {
    fn new(array: &KeySetMapArray, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let map_dtype = array
            .dtype()
            .as_map_opt()
            .cloned()
            .unwrap_or_else(|| unreachable!());
        let ids_array = array.keyset_ids();
        let validity = ids_array.validity()?;
        let valid = validity.execute_mask(array.len(), ctx)?;
        let keysets = array.keysets().clone().execute::<ListViewArray>(ctx)?;
        Ok(Self {
            map_dtype,
            ids: codes_u32(ids_array, ctx)?,
            valid,
            validity,
            value_offsets: to_usize_vec(array.value_offsets(), ctx)?,
            keyset_offsets: to_usize_vec(keysets.offsets(), ctx)?,
            keyset_sizes: to_usize_vec(keysets.sizes(), ctx)?,
            keysets,
            values: array.values().clone(),
        })
    }

    fn size(&self, row: usize) -> usize {
        if self.valid.value(row) {
            self.keyset_sizes[self.ids[row] as usize]
        } else {
            0
        }
    }

    fn row_offsets_sizes(&self) -> (Vec<u64>, Vec<u64>) {
        (0..self.ids.len())
            .map(|row| (self.value_offsets[row] as u64, self.size(row) as u64))
            .unzip()
    }
}

/// Decodes a key-set map into a canonical map. Rows sharing a value range share their entries.
pub fn decode(array: &KeySetMapArray, ctx: &mut ExecutionCtx) -> VortexResult<MapArray> {
    let p = Parts::new(array, ctx)?;
    let mut key_index = vec![0u64; p.values.len()];
    // Rows sharing a value range repeat their predecessor's offset and key set. An empty row can
    // share an offset with the next row's start, so the key set must match too.
    let mut last = None;
    for row in 0..p.ids.len() {
        let offset = p.value_offsets[row];
        if !p.valid.value(row) || last == Some((offset, p.ids[row])) {
            continue;
        }
        last = Some((offset, p.ids[row]));
        let start = p.keyset_offsets[p.ids[row] as usize];
        for t in 0..p.size(row) {
            key_index[offset + t] = (start + t) as u64;
        }
    }
    let key_index = PrimitiveArray::new(Buffer::from(key_index), Validity::NonNullable);
    let keys = p
        .keysets
        .elements()
        .take(key_index.into_array())?
        .execute::<Canonical>(ctx)?
        .into_array();
    let values = p.values.clone().execute::<Canonical>(ctx)?.into_array();
    let (offsets, sizes) = p.row_offsets_sizes();
    build_map(&p.map_dtype, keys, values, offsets, sizes, p.validity)
}

/// The keys of each row as a `List<Utf8>`, sharing the key-set storage.
pub fn label_names(array: &KeySetMapArray, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
    let p = Parts::new(array, ctx)?;
    let (offsets, sizes): (Vec<u64>, Vec<u64>) = (0..p.ids.len())
        .map(|row| match p.valid.value(row) {
            true => (p.keyset_offsets[p.ids[row] as usize] as u64, p.size(row) as u64),
            false => (0, 0),
        })
        .unzip();
    build_listview(p.keysets.elements().clone(), offsets, sizes, p.validity)
}

/// The distinct keys over all non-null rows, sorted.
pub fn distinct_label_names(
    array: &KeySetMapArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<String>> {
    let p = Parts::new(array, ctx)?;
    let used: HashSet<u32> = (0..p.ids.len())
        .filter(|&row| p.valid.value(row))
        .map(|row| p.ids[row])
        .collect();
    let keys = p.keysets.elements().clone().execute::<VarBinViewArray>(ctx)?;
    let strings = Strings::new(&keys);
    let mut names: Vec<String> = used
        .into_iter()
        .flat_map(|id| {
            let start = p.keyset_offsets[id as usize];
            (start..start + p.keyset_sizes[id as usize]).map(|j| strings.get(j))
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|k| String::from_utf8_lossy(k).into_owned())
        .collect();
    names.sort();
    Ok(names)
}

/// The first position of `key` in every key set.
fn key_positions(p: &Parts, key: &str, ctx: &mut ExecutionCtx) -> VortexResult<Vec<Option<usize>>> {
    let keys = p.keysets.elements().clone().execute::<VarBinViewArray>(ctx)?;
    let strings = Strings::new(&keys);
    Ok((0..p.keyset_offsets.len())
        .map(|id| {
            let start = p.keyset_offsets[id];
            (0..p.keyset_sizes[id]).find(|&t| strings.get(start + t) == key.as_bytes())
        })
        .collect())
}

/// The value of `key` in each row formatted as a string, null when absent or null.
pub fn get_label_utf8(
    array: &KeySetMapArray,
    key: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let p = Parts::new(array, ctx)?;
    let positions = key_positions(&p, key, ctx)?;
    let mut indices = Vec::with_capacity(p.ids.len());
    let mut valid = BitBufferMut::with_capacity(p.ids.len());
    for row in 0..p.ids.len() {
        let found = p.valid.value(row).then(|| positions[p.ids[row] as usize]).flatten();
        indices.push(found.map_or(0, |t| (p.value_offsets[row] + t) as u64));
        valid.append(found.is_some());
    }
    let indices = PrimitiveArray::new(
        Buffer::from(indices),
        Validity::from_bit_buffer(valid.freeze(), Nullability::Nullable),
    );
    let values = p.values.take(indices.into_array())?;
    values_to_utf8(&values.execute::<Canonical>(ctx)?.into_array(), ctx)
}

/// Restricts each row to the entries whose key is in `keys`, keeping the key-set layout.
pub fn project(
    array: &KeySetMapArray,
    keys: &[&str],
    ctx: &mut ExecutionCtx,
) -> VortexResult<KeySetMapArray> {
    let p = Parts::new(array, ctx)?;
    let elements = p.keysets.elements().clone().execute::<VarBinViewArray>(ctx)?;
    let strings = Strings::new(&elements);
    let wanted: HashSet<&[u8]> = keys.iter().map(|k| k.as_bytes()).collect();

    // Kept positions of every key set; projected key sets keep their ids.
    let mut kept_entries = Vec::new();
    let mut new_offsets = Vec::with_capacity(p.keyset_offsets.len());
    let mut new_sizes = Vec::with_capacity(p.keyset_offsets.len());
    let mut kept_positions: Vec<Vec<usize>> = Vec::with_capacity(p.keyset_offsets.len());
    for id in 0..p.keyset_offsets.len() {
        let start = p.keyset_offsets[id];
        let positions: Vec<usize> = (0..p.keyset_sizes[id])
            .filter(|&t| wanted.contains(strings.get(start + t)))
            .collect();
        new_offsets.push(kept_entries.len() as u64);
        new_sizes.push(positions.len() as u64);
        kept_entries.extend(positions.iter().map(|&t| (start + t) as u64));
        kept_positions.push(positions);
    }

    let mut value_entries = Vec::new();
    let mut value_offsets = Vec::with_capacity(p.ids.len());
    // Rows sharing values share offset and key set; an empty row can share only the offset.
    let mut last: Option<(usize, u32, u64)> = None;
    for row in 0..p.ids.len() {
        if !p.valid.value(row) {
            value_offsets.push(0);
            continue;
        }
        let offset = p.value_offsets[row];
        if let Some((prev, id, new)) = last
            && prev == offset
            && id == p.ids[row]
        {
            value_offsets.push(new);
            continue;
        }
        let new = value_entries.len() as u64;
        value_entries.extend(
            kept_positions[p.ids[row] as usize]
                .iter()
                .map(|&t| (offset + t) as u64),
        );
        value_offsets.push(new);
        last = Some((offset, p.ids[row], new));
    }

    let take = |array: &ArrayRef, idx: Vec<u64>| -> VortexResult<ArrayRef> {
        array.take(PrimitiveArray::new(Buffer::from(idx), Validity::NonNullable).into_array())
    };
    let keysets = build_listview(
        take(p.keysets.elements(), kept_entries)?,
        new_offsets,
        new_sizes,
        Validity::NonNullable,
    )?;
    KeySetMap::try_new(
        array.dtype().clone(),
        array.keyset_ids().clone(),
        keysets.into_array(),
        PrimitiveArray::new(Buffer::from(value_offsets), Validity::NonNullable).into_array(),
        take(&p.values, value_entries)?,
    )
}
