// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading one label of one row without decoding anything else.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArrayExt;
use vortex_runend::RunEndArraySlotsExt;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ShreddedMapArray;
use crate::array::ShreddedMapArrayExt;
use crate::array::ShreddedMapArraySlotsExt;
use crate::flat::to_usize_vec;
use crate::labels::values_to_utf8;

fn scalar_usize(array: &ArrayRef, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    Ok(array
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_::<usize>()
        .unwrap_or_default())
}

/// The position of `row` among a sparse array's present rows, if present.
///
/// Reads the row's chunk bounds from the chunk offsets, decodes only that chunk's patch indices
/// (at most 1024) and searches them in memory.
fn sparse_rank(
    sparse: &vortex_sparse::SparseArray,
    row: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<usize>> {
    let patches = sparse.patches();
    let indices = patches.indices();
    let (start, end) = match patches.chunk_offsets() {
        Some(offsets) if patches.offset() == 0 => {
            let chunk = row / PATCH_CHUNK_SIZE;
            let start = scalar_usize(offsets, chunk, ctx)?;
            let end = if chunk + 1 < offsets.len() {
                scalar_usize(offsets, chunk + 1, ctx)?
            } else {
                indices.len()
            };
            (start, end)
        }
        _ => (0, indices.len()),
    };
    let chunk = indices
        .slice(start..end)?
        .execute::<PrimitiveArray>(ctx)?
        .into_array();
    let target = row + patches.offset();
    Ok(to_usize_vec(&chunk, ctx)?
        .binary_search(&target)
        .ok()
        .map(|i| start + i))
}

/// The value at `row` of a shredded column, peeling sparse and dictionary layers.
fn column_value(
    column: &ArrayRef,
    row: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    let mut array = column.clone();
    let mut pos = row;
    loop {
        if let Some(sparse) = array.as_opt::<Sparse>()
            && sparse.fill_scalar().is_null()
        {
            let Some(rank) = sparse_rank(&sparse.into_owned(), pos, ctx)? else {
                return Ok(None);
            };
            pos = rank;
            array = sparse.patches().values().clone();
            continue;
        }
        if let Some(dict) = array.as_opt::<Dict>() {
            let code = dict.codes().execute_scalar(pos, ctx)?;
            let Some(code) = code.as_primitive().as_::<usize>() else {
                return Ok(None);
            };
            pos = code;
            array = dict.values().clone();
            continue;
        }
        let value = array.execute_scalar(pos, ctx)?;
        return Ok((!value.is_null()).then_some(value));
    }
}

fn scalar_to_string(value: &Scalar, ctx: &mut ExecutionCtx) -> VortexResult<Option<String>> {
    if let Some(s) = value.as_utf8_opt() {
        return Ok(s.value().map(|v| v.to_string()));
    }
    let one = vortex_array::arrays::ConstantArray::new(value.clone(), 1).into_array();
    let text = values_to_utf8(&one, ctx)?.execute_scalar(0, ctx)?;
    Ok(text.as_utf8().value().map(|v| v.to_string()))
}

/// The value of `key` in the map at `row` of a canonical or compressed map, read through scalar
/// access to that one row.
pub fn map_label_at(
    map: &ArrayRef,
    key: &str,
    row: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<String>> {
    let scalar = map.execute_scalar(row, ctx)?;
    let map = scalar.as_map();
    for (k, v) in map.entries() {
        if k.as_utf8().value().is_some_and(|k| k.as_str() == key) {
            return if v.is_null() {
                Ok(None)
            } else {
                scalar_to_string(&v, ctx)
            };
        }
    }
    Ok(None)
}

/// The value of `key` at `row` formatted as a string, reading only that key's column (or the
/// row's residual entries for a key without a column).
pub fn label_at(
    array: &ShreddedMapArray,
    key: &str,
    row: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<String>> {
    let column = array
        .data()
        .columns()
        .binary_search_by(|c| c.key.as_ref().cmp(key))
        .ok();
    match column {
        Some(k) => match column_value(&array.column_array(k), row, ctx)? {
            Some(value) => scalar_to_string(&value, ctx),
            // A shredded key absent from its column may still have a null-valued residual entry,
            // which reads as null too.
            None => Ok(None),
        },
        None => map_label_at(array.residual(), key, row, ctx),
    }
}

/// Elements decoded per block on first touch.
const BLOCK: usize = 1024;

/// An integer array decoded one block at a time, on first touch. Nulls read as `None`.
struct Ints {
    array: ArrayRef,
    blocks: Vec<Option<Box<[usize]>>>,
}

impl Ints {
    /// `searched` arrays are binary searched, reading many elements per lookup, so they keep
    /// decoded blocks. Others read one element per lookup, directly when the encoding decodes one
    /// element in constant time.
    fn new(array: ArrayRef, searched: bool) -> Self {
        let direct = !searched
            && ["vortex.primitive", "fastlanes.bitpacked", "vortex.constant"]
                .contains(&array.encoding_id().as_ref());
        let blocks = if direct {
            Vec::new()
        } else {
            (0..array.len().div_ceil(BLOCK)).map(|_| None).collect()
        };
        Self { array, blocks }
    }

    fn get(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Option<usize>> {
        if self.blocks.is_empty() {
            let value = self.array.execute_scalar(index, ctx)?;
            return Ok(value.as_primitive().as_::<usize>());
        }
        let block = index / BLOCK;
        if self.blocks[block].is_none() {
            let start = block * BLOCK;
            let end = (start + BLOCK).min(self.array.len());
            let slice = self.array.slice(start..end)?;
            let valid = slice.validity()?.execute_mask(end - start, ctx)?;
            let mut values = to_usize_vec(&slice, ctx)?;
            for (i, v) in values.iter_mut().enumerate() {
                if !valid.value(i) {
                    *v = usize::MAX;
                }
            }
            self.blocks[block] = Some(values.into_boxed_slice());
        }
        let value = self.blocks[block]
            .as_ref()
            .map_or(usize::MAX, |b| b[index % BLOCK]);
        Ok((value != usize::MAX).then_some(value))
    }

    /// Bytes held by decoded blocks.
    fn nbytes(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.len() * size_of::<usize>())
            .sum()
    }
}

/// Any array decoded to canonical form one block at a time, on first touch.
struct Blocks {
    array: ArrayRef,
    blocks: Vec<Option<ArrayRef>>,
}

/// Arrays up to this length and compressed size, such as most dictionaries, decode whole on
/// first touch, since string encodings like FSST or OnPair decode a slice little faster than the
/// whole array.
const WHOLE_LEN: usize = 1 << 16;
const WHOLE_BYTES: u64 = 256 << 10;

impl Blocks {
    fn new(array: ArrayRef) -> Self {
        let n = if array.len() <= WHOLE_LEN && array.nbytes() <= WHOLE_BYTES {
            1
        } else {
            array.len().div_ceil(BLOCK)
        };
        let blocks = (0..n).map(|_| None).collect();
        Self { array, blocks }
    }

    fn get(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        if self.blocks.len() == 1 {
            let decoded = match &mut self.blocks[0] {
                Some(decoded) => decoded,
                slot @ None => slot.insert(
                    self.array
                        .clone()
                        .execute::<vortex_array::Canonical>(ctx)?
                        .into_array(),
                ),
            };
            return decoded.execute_scalar(index, ctx);
        }
        let block = index / BLOCK;
        let decoded = match &mut self.blocks[block] {
            Some(decoded) => decoded,
            slot @ None => {
                let start = block * BLOCK;
                let end = (start + BLOCK).min(self.array.len());
                let decoded = self
                    .array
                    .slice(start..end)?
                    .execute::<vortex_array::Canonical>(ctx)?
                    .into_array();
                slot.insert(decoded)
            }
        };
        decoded.execute_scalar(index % BLOCK, ctx)
    }

    fn nbytes(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(|b| usize::try_from(b.nbytes()).unwrap_or(usize::MAX))
            .sum()
    }
}

/// How a [`Probe`] reads its array: through a sparse, dictionary or run-end layer, or from
/// decoded blocks at the leaf.
enum Layer {
    /// A null-filled sparse array. `chunk_offsets[c]` is the first patch of 1024-row chunk `c`.
    Sparse {
        chunk_offsets: Option<Vec<usize>>,
        indices: Ints,
        offset: usize,
        values: Box<Probe>,
    },
    Dict {
        codes: Box<Probe>,
        values: Box<Probe>,
    },
    RunEnd {
        ends: Vec<usize>,
        offset: usize,
        values: Box<Probe>,
    },
    Ints(Ints),
    Values(Blocks),
}

/// A row reader over one possibly compressed array that keeps what it decodes between reads.
///
/// Each layer keeps only its own small index structures (decoded run ends, sparse chunk
/// offsets) and the blocks of its children that reads have touched, so a lookup after the
/// first few costs a binary search in memory and a block-local read.
pub struct Probe {
    layer: Layer,
}

impl Probe {
    /// A probe that reads scalars.
    pub fn new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Self::build(array, false, ctx)
    }

    /// A probe that reads non-negative integers, such as offsets or codes.
    pub fn new_index(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Self::build(array, true, ctx)
    }

    fn build(array: &ArrayRef, index: bool, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(sparse) = array.as_opt::<Sparse>()
            && sparse.fill_scalar().is_null()
        {
            let patches = sparse.patches();
            let chunk_offsets = patches
                .chunk_offsets()
                .as_ref()
                .map(|o| to_usize_vec(o, ctx))
                .transpose()?;
            let layer = Layer::Sparse {
                chunk_offsets,
                indices: Ints::new(patches.indices().clone(), true),
                offset: patches.offset(),
                values: Box::new(Self::build(patches.values(), index, ctx)?),
            };
            return Ok(Self { layer });
        }
        if let Some(dict) = array.as_opt::<Dict>() {
            let layer = Layer::Dict {
                codes: Box::new(Self::build(dict.codes(), true, ctx)?),
                values: Box::new(Self::build(dict.values(), index, ctx)?),
            };
            return Ok(Self { layer });
        }
        if let Some(run_end) = array.as_opt::<RunEnd>() {
            let layer = Layer::RunEnd {
                ends: to_usize_vec(run_end.ends(), ctx)?,
                offset: run_end.offset(),
                values: Box::new(Self::build(run_end.values(), index, ctx)?),
            };
            return Ok(Self { layer });
        }
        let layer = if index && array.dtype().is_int() {
            Layer::Ints(Ints::new(array.clone(), false))
        } else {
            Layer::Values(Blocks::new(array.clone()))
        };
        Ok(Self { layer })
    }

    /// The position of a row in the next layer down, or `None` if the row is null here.
    fn descend(&mut self, row: usize, ctx: &mut ExecutionCtx) -> VortexResult<Option<usize>> {
        Ok(match &mut self.layer {
            Layer::Sparse {
                chunk_offsets,
                indices,
                offset,
                ..
            } => {
                let target = row + *offset;
                let (mut lo, mut hi) = match chunk_offsets {
                    Some(offsets) if *offset == 0 => {
                        let chunk = row / PATCH_CHUNK_SIZE;
                        let end = offsets
                            .get(chunk + 1)
                            .copied()
                            .unwrap_or(indices.array.len());
                        (offsets[chunk], end)
                    }
                    _ => (0, indices.array.len()),
                };
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let index = indices.get(mid, ctx)?.unwrap_or_default();
                    match index.cmp(&target) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => return Ok(Some(mid)),
                    }
                }
                None
            }
            Layer::Dict { codes, .. } => codes.index(row, ctx)?,
            Layer::RunEnd { ends, offset, .. } => {
                let target = row + *offset;
                Some(ends.partition_point(|&end| end <= target))
            }
            Layer::Ints(_) | Layer::Values(_) => Some(row),
        })
    }

    fn child(&mut self) -> Option<&mut Probe> {
        match &mut self.layer {
            Layer::Sparse { values, .. }
            | Layer::Dict { values, .. }
            | Layer::RunEnd { values, .. } => Some(values),
            Layer::Ints(_) | Layer::Values(_) => None,
        }
    }

    /// The integer at `row`, or `None` if it is null. Only for probes built with
    /// [`Self::new_index`].
    pub fn index(&mut self, row: usize, ctx: &mut ExecutionCtx) -> VortexResult<Option<usize>> {
        let Some(pos) = self.descend(row, ctx)? else {
            return Ok(None);
        };
        match &mut self.layer {
            Layer::Ints(ints) => ints.get(pos, ctx),
            Layer::Values(blocks) => {
                let value = blocks.get(pos, ctx)?;
                Ok(value.as_primitive_opt().and_then(|p| p.as_::<usize>()))
            }
            _ => self.child().map_or(Ok(None), |child| child.index(pos, ctx)),
        }
    }

    /// The scalar at `row`, or `None` if it is null.
    pub fn scalar(&mut self, row: usize, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>> {
        let Some(pos) = self.descend(row, ctx)? else {
            return Ok(None);
        };
        match &mut self.layer {
            Layer::Values(blocks) => {
                let value = blocks.get(pos, ctx)?;
                Ok((!value.is_null()).then_some(value))
            }
            Layer::Ints(_) => unreachable!("scalar probes never build integer leaves"),
            _ => self
                .child()
                .map_or(Ok(None), |child| child.scalar(pos, ctx)),
        }
    }

    /// Bytes this probe holds in decoded state.
    pub fn nbytes(&self) -> usize {
        match &self.layer {
            Layer::Sparse {
                chunk_offsets,
                indices,
                values,
                ..
            } => {
                chunk_offsets
                    .as_ref()
                    .map_or(0, |o| o.len() * size_of::<usize>())
                    + indices.nbytes()
                    + values.nbytes()
            }
            Layer::Dict { codes, values } => codes.nbytes() + values.nbytes(),
            Layer::RunEnd { ends, values, .. } => ends.len() * size_of::<usize>() + values.nbytes(),
            Layer::Ints(ints) => ints.nbytes(),
            Layer::Values(blocks) => blocks.nbytes(),
        }
    }
}

/// How a [`MapProbe`] finds a key among a row's entries.
enum KeyMatch {
    /// Dictionary-encoded keys: the key is resolved to its codes once, then entries compare
    /// integer codes.
    Codes {
        codes: Probe,
        lookup: HashMap<String, Vec<usize>>,
    },
    Strings(Probe),
}

/// A row reader over a possibly compressed map that keeps what it decodes between reads.
pub struct MapProbe {
    validity: RepeatedArrayProbe,
    offsets: Probe,
    sizes: Probe,
    keys: KeyMatch,
    values: Probe,
}

impl MapProbe {
    pub fn new(map: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let validity = map.repeated_probe();
        let map = map.clone().execute::<MapArray>(ctx)?;
        let entries = match map.entries().as_opt::<ListView>() {
            Some(entries) => entries.into_owned(),
            None => map.entries().clone().execute::<ListViewArray>(ctx)?,
        };
        let elements = entries.elements().clone().execute::<StructArray>(ctx)?;
        let key_array = elements.unmasked_field(0).clone();
        let keys = match key_array.as_opt::<Dict>() {
            Some(dict) => {
                let dictionary = dict.values().clone().execute::<VarBinViewArray>(ctx)?;
                let mut lookup: HashMap<String, Vec<usize>> = HashMap::new();
                for code in 0..dictionary.len() {
                    if let Some(key) = dictionary.execute_scalar(code, ctx)?.as_utf8().value() {
                        lookup.entry(key.to_string()).or_default().push(code);
                    }
                }
                KeyMatch::Codes {
                    codes: Probe::new_index(dict.codes(), ctx)?,
                    lookup,
                }
            }
            None => KeyMatch::Strings(Probe::new(&key_array, ctx)?),
        };
        Ok(Self {
            validity,
            offsets: Probe::new_index(entries.offsets(), ctx)?,
            sizes: Probe::new_index(entries.sizes(), ctx)?,
            keys,
            values: Probe::new(elements.unmasked_field(1), ctx)?,
        })
    }

    /// The value of the first entry of `key` in the map at `row`, or `None` if the key is
    /// absent, null, or the row is null.
    pub fn get(
        &mut self,
        key: &str,
        row: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if let KeyMatch::Codes { lookup, .. } = &self.keys
            && !lookup.contains_key(key)
        {
            return Ok(None);
        }
        if !self.validity.execute_is_valid(row, ctx)? {
            return Ok(None);
        }
        let offset = self.offsets.index(row, ctx)?.unwrap_or_default();
        let size = self.sizes.index(row, ctx)?.unwrap_or_default();
        for entry in offset..offset + size {
            let found = match &mut self.keys {
                KeyMatch::Codes { codes, lookup } => codes
                    .index(entry, ctx)?
                    .is_some_and(|code| lookup.get(key).is_some_and(|c| c.contains(&code))),
                KeyMatch::Strings(keys) => keys
                    .scalar(entry, ctx)?
                    .is_some_and(|k| k.as_utf8().value().is_some_and(|k| k.as_str() == key)),
            };
            if found {
                return self.values.scalar(entry, ctx);
            }
        }
        Ok(None)
    }

    /// [`Self::get`] formatted as a label string.
    pub fn get_label(
        &mut self,
        key: &str,
        row: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<String>> {
        match self.get(key, row, ctx)? {
            Some(value) => scalar_to_string(&value, ctx),
            None => Ok(None),
        }
    }

    pub fn nbytes(&self) -> usize {
        let keys = match &self.keys {
            KeyMatch::Codes { codes, lookup } => {
                codes.nbytes() + lookup.keys().map(|k| k.len() + 32).sum::<usize>()
            }
            KeyMatch::Strings(keys) => keys.nbytes(),
        };
        self.offsets.nbytes() + self.sizes.nbytes() + keys + self.values.nbytes()
    }
}

/// A label reader over a shredded map that builds a [`Probe`] per column on its first read and
/// a [`MapProbe`] over the residual on the first read of an unshredded key.
pub struct ShreddedProbe {
    array: ShreddedMapArray,
    columns: Vec<Option<Probe>>,
    residual: Option<MapProbe>,
}

impl ShreddedProbe {
    pub fn new(array: &ShreddedMapArray) -> Self {
        Self {
            array: array.clone(),
            columns: (0..array.data().columns().len()).map(|_| None).collect(),
            residual: None,
        }
    }

    /// The value of `key` at `row` formatted as a label string.
    pub fn get_label(
        &mut self,
        key: &str,
        row: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<String>> {
        let column = self
            .array
            .data()
            .columns()
            .binary_search_by(|c| c.key.as_ref().cmp(key))
            .ok();
        let value = match column {
            Some(k) => {
                let probe = match &mut self.columns[k] {
                    Some(probe) => probe,
                    slot @ None => slot.insert(Probe::new(&self.array.column_array(k), ctx)?),
                };
                probe.scalar(row, ctx)?
            }
            None => {
                let probe = match &mut self.residual {
                    Some(probe) => probe,
                    slot @ None => slot.insert(MapProbe::new(self.array.residual(), ctx)?),
                };
                probe.get(key, row, ctx)?
            }
        };
        match value {
            Some(value) => scalar_to_string(&value, ctx),
            None => Ok(None),
        }
    }

    /// Bytes held in decoded state across all probes built so far.
    pub fn nbytes(&self) -> usize {
        self.columns
            .iter()
            .flatten()
            .map(Probe::nbytes)
            .sum::<usize>()
            + self.residual.as_ref().map_or(0, MapProbe::nbytes)
    }
}
