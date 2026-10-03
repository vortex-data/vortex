// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Label operations on canonical maps (the baseline) and on shredded maps.
//!
//! All operations follow map semantics where a key's value is that of its first entry in the row.

use std::collections::BTreeSet;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArrayExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ShreddedColumn;
use crate::ShreddedMap;
use crate::ShreddedMapArray;
use crate::array::ShreddedMapArraySlotsExt;
use crate::decode::ShreddedParts;
use crate::decode::decode_keys;
use crate::decode::decode_parts;
use crate::flat::FlatMap;
use crate::flat::Strings;
use crate::flat::build_map;
use crate::flat::to_usize_vec;
use crate::labels::values_to_utf8;

/// Values decoded per block when gathering a few dictionary values.
const GATHER_BLOCK: usize = 1024;

/// A dictionary whose values hold only the entries its codes reference.
///
/// Decoding a compressed string dictionary (FSST, OnPair) and gathering from it decodes every
/// value, even when a filtered or taken column references a handful. This decodes only the
/// 1024-value blocks holding referenced entries, and keeps the dictionary as is when those
/// blocks cover at least half of it.
fn narrow_dict(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let Some(dict) = array.as_opt::<Dict>() else {
        return Ok(array.clone());
    };
    let values = dict.values();
    if values.is_canonical() || values.len() <= GATHER_BLOCK {
        return Ok(array.clone());
    }
    let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let code_values = crate::decode::codes_u32(&codes.clone().into_array(), ctx)?;
    let mut used = vec![false; values.len()];
    for &code in &code_values {
        if let Some(u) = used.get_mut(code as usize) {
            *u = true;
        }
    }
    let blocks: Vec<usize> = used
        .chunks(GATHER_BLOCK)
        .enumerate()
        .filter(|(_, block)| block.iter().any(|&u| u))
        .map(|(b, _)| b)
        .collect();
    if blocks.len() * GATHER_BLOCK * 2 >= values.len() {
        return Ok(array.clone());
    }
    let mut remap = vec![0u32; values.len()];
    let mut next = 0u32;
    let mut chunks = Vec::with_capacity(blocks.len());
    for b in blocks {
        let start = b * GATHER_BLOCK;
        let end = (start + GATHER_BLOCK).min(values.len());
        let local: Vec<u32> = (start..end)
            .filter(|&c| used[c])
            .map(|c| {
                remap[c] = next;
                next += 1;
                (c - start) as u32
            })
            .collect();
        let block = values.slice(start..end)?.execute::<Canonical>(ctx)?.into_array();
        let local = PrimitiveArray::new(Buffer::from(local), Validity::NonNullable).into_array();
        chunks.push(block.take(local)?);
    }
    let gathered = ChunkedArray::try_new(chunks, values.dtype().clone())?
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array();
    let remapped: Buffer<u32> = code_values.iter().map(|&c| remap[c as usize]).collect();
    let codes = PrimitiveArray::new(remapped, codes.validity()?).into_array();
    Ok(DictArray::try_new(codes, gathered)?.into_array())
}

/// `values.take(indices)`, gathering through a dictionary's codes and narrowing its values.
fn take_narrow(values: &ArrayRef, indices: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    match values.as_opt::<Dict>() {
        Some(dict) => {
            let codes = dict.codes().take(indices)?;
            narrow_dict(&DictArray::try_new(codes, dict.values().clone())?.into_array(), ctx)
        }
        None => values.take(indices),
    }
}

/// The rows of a shredded column selected by `mask`, keeping sparse and dictionary layers and
/// narrowing dictionaries to the values the selected rows reference.
fn filter_column(column: &ArrayRef, mask: &Mask, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if let Some(sparse) = column.as_opt::<Sparse>()
        && sparse.fill_scalar().is_null()
    {
        let fill = sparse.fill_scalar().clone();
        return Ok(match sparse.patches().filter(mask, ctx)? {
            Some(patches) => {
                let patches = patches.map_values(|values| narrow_dict(&values, ctx))?;
                Sparse::try_new_from_patches(patches, fill)?.into_array()
            }
            None => ConstantArray::new(fill, mask.true_count()).into_array(),
        });
    }
    if let Some(dict) = column.as_opt::<Dict>() {
        let codes = dict.codes().filter(mask.clone())?;
        return narrow_dict(&DictArray::try_new(codes, dict.values().clone())?.into_array(), ctx);
    }
    column.filter(mask.clone())
}

/// Label operations over the canonical [`MapArray`] representation.
pub mod map {
    use super::*;

    /// The keys of each row as a `List<Utf8>`. Zero-copy over the map entries.
    pub fn label_names(map: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
        let flat = FlatMap::new(map, ctx)?;
        let offsets: Vec<u64> = flat.offsets.iter().map(|&o| o as u64).collect();
        let sizes: Vec<u64> = flat.sizes.iter().map(|&s| s as u64).collect();
        let offsets = PrimitiveArray::new(Buffer::from(offsets), Validity::NonNullable);
        let sizes = PrimitiveArray::new(Buffer::from(sizes), Validity::NonNullable);
        ListViewArray::try_new(
            flat.keys.into_array(),
            offsets.into_array(),
            sizes.into_array(),
            flat.validity,
        )
    }

    /// The distinct keys over all non-null rows, sorted.
    pub fn distinct_label_names(
        map: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<String>> {
        let flat = FlatMap::new(map, ctx)?;
        let keys = Strings::new(&flat.keys);
        let mut seen: HashSet<&[u8]> = HashSet::new();
        for row in 0..flat.len {
            if flat.repeats_prev(row) {
                continue;
            }
            for j in flat.range(row) {
                seen.insert(keys.get(j));
            }
        }
        Ok(seen
            .into_iter()
            .map(|k| String::from_utf8_lossy(k).into_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect())
    }

    /// The value of `key` in each row formatted as a string, null when absent or null.
    ///
    /// Keys are matched with a vectorized comparison, which on dictionary-encoded keys compares
    /// the dictionary once and then integer codes, so no key string is decoded per entry. Only
    /// the matched values are taken and formatted.
    pub fn get_label_utf8(
        map: &ArrayRef,
        key: &str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let map = map.clone().execute::<MapArray>(ctx)?;
        let len = map.len();
        let entries = map.entries().clone().execute::<ListViewArray>(ctx)?;
        let elements = entries.elements().clone().execute::<StructArray>(ctx)?;
        let keys = elements.unmasked_field(0).clone();
        let needle = ConstantArray::new(
            Scalar::utf8(key, keys.dtype().nullability()),
            keys.len(),
        )
        .into_array();
        let matches = keys
            .binary(needle, Operator::Eq)?
            .null_as_false()
            .execute(ctx)?;
        if matches.all_false() {
            return Ok(ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), len)
                .into_array());
        }
        let offsets = to_usize_vec(entries.offsets(), ctx)?;
        let sizes = to_usize_vec(entries.sizes(), ctx)?;
        let row_valid = map.map_validity().execute_mask(len, ctx)?;
        let mut indices = Vec::with_capacity(len);
        let mut valid = vortex_buffer::BitBufferMut::with_capacity(len);
        for row in 0..len {
            let found = if row_valid.value(row) {
                (offsets[row]..offsets[row] + sizes[row]).find(|&j| matches.value(j))
            } else {
                None
            };
            indices.push(found.unwrap_or(0) as u64);
            valid.append(found.is_some());
        }
        let indices = PrimitiveArray::new(
            Buffer::from(indices),
            Validity::from_bit_buffer(valid.freeze(), Nullability::Nullable),
        );
        let values = take_narrow(elements.unmasked_field(1), indices.into_array(), ctx)?;
        values_to_utf8(&values, ctx)
    }

    /// The map with every value formatted as a string.
    pub fn to_utf8_map(map: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<MapArray> {
        let flat = FlatMap::new(map, ctx)?;
        let values = values_to_utf8(&flat.values, ctx)?;
        let map_dtype = utf8_map_dtype(&flat.map_dtype)?;
        let values = values.cast(map_dtype.value_dtype())?;
        let offsets = flat.offsets.iter().map(|&o| o as u64).collect();
        let sizes = flat.sizes.iter().map(|&s| s as u64).collect();
        build_map(
            &map_dtype,
            flat.keys.into_array(),
            values,
            offsets,
            sizes,
            flat.validity,
        )
    }

    /// The map restricted to the entries whose key is in `keys`.
    pub fn project(map: &ArrayRef, keys: &[&str], ctx: &mut ExecutionCtx) -> VortexResult<MapArray> {
        let flat = FlatMap::new(map, ctx)?;
        let strings = Strings::new(&flat.keys);
        let wanted: HashSet<&[u8]> = keys.iter().map(|k| k.as_bytes()).collect();
        let mut kept = Vec::new();
        let mut offsets = Vec::with_capacity(flat.len);
        let mut sizes = Vec::with_capacity(flat.len);
        for row in 0..flat.len {
            if flat.repeats_prev(row) {
                offsets.push(offsets[row - 1]);
                sizes.push(sizes[row - 1]);
                continue;
            }
            let start = kept.len();
            for j in flat.range(row) {
                if wanted.contains(strings.get(j)) {
                    kept.push(j as u64);
                }
            }
            offsets.push(start as u64);
            sizes.push((kept.len() - start) as u64);
        }
        let kept = PrimitiveArray::new(Buffer::from(kept), Validity::NonNullable).into_array();
        let new_keys = flat
            .keys
            .into_array()
            .take(kept.clone())?
            .execute::<Canonical>(ctx)?
            .into_array();
        let new_values = flat.values.take(kept)?.execute::<Canonical>(ctx)?.into_array();
        build_map(
            &flat.map_dtype,
            new_keys,
            new_values,
            offsets,
            sizes,
            flat.validity,
        )
    }
}


fn utf8_map_dtype(map_dtype: &MapDType) -> VortexResult<MapDType> {
    MapDType::try_new(
        DType::Utf8(Nullability::NonNullable),
        DType::Utf8(map_dtype.value_dtype().nullability()),
        map_dtype.keys_sorted(),
    )
}

/// Label operations over a [`ShreddedMapArray`].
pub mod shredded {
    use super::*;

    fn parts(array: &ShreddedMapArray) -> ShreddedParts<'_> {
        ShreddedParts::from_view(array.as_view())
    }

    pub(super) fn find_column(array: &ShreddedMapArray, key: &str) -> Option<usize> {
        array
            .data()
            .columns()
            .binary_search_by(|c| c.key.as_ref().cmp(key))
            .ok()
    }

    /// Decompresses into a canonical map.
    pub fn to_map(array: &ShreddedMapArray, ctx: &mut ExecutionCtx) -> VortexResult<MapArray> {
        decode_parts(&parts(array), ctx)
    }

    /// The keys of each row as a `List<Utf8>`.
    pub fn label_names(
        array: &ShreddedMapArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        decode_keys(&parts(array), ctx)
    }

    /// The distinct keys over all non-null rows, sorted.
    pub fn distinct_label_names(
        array: &ShreddedMapArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<String>> {
        let mut names: BTreeSet<String> = map::distinct_label_names(array.residual(), ctx)?
            .into_iter()
            .collect();
        for (column, values) in array.data().columns().iter().zip(array.columns().iter()) {
            if values.valid_count(ctx)? > 0 {
                names.insert(column.key.to_string());
            }
        }
        Ok(names.into_iter().collect())
    }

    /// The value of `key` in each row formatted as a string, null when absent or null.
    pub fn get_label_utf8(
        array: &ShreddedMapArray,
        key: &str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match find_column(array, key) {
            // The shredder only moves a key's first entry, and only when it is non-null, so a null
            // column row means the first entry is absent or null.
            Some(c) => values_to_utf8(&array.columns()[c], ctx),
            None => map::get_label_utf8(array.residual(), key, ctx),
        }
    }

    /// Formats every value as a string, keeping the shredded layout.
    pub fn to_utf8(
        array: &ShreddedMapArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ShreddedMapArray> {
        let residual = map::to_utf8_map(array.residual(), ctx)?.into_array();
        let value_dtype = residual
            .dtype()
            .as_map_opt()
            .map(|m| m.value_dtype())
            .unwrap_or_else(|| unreachable!());
        let columns = array
            .data()
            .columns()
            .iter()
            .map(|c| ShreddedColumn {
                key: c.key.clone(),
                variant: None,
            })
            .collect();
        let column_arrays = array
            .columns()
            .iter()
            .map(|c| values_to_utf8(c, ctx)?.cast(value_dtype.as_nullable()))
            .collect::<VortexResult<Vec<_>>>()?;
        ShreddedMap::try_new_with_repeats(
            residual,
            array.repeats().cloned(),
            columns,
            column_arrays,
        )
    }

    /// Decompresses into a map with every value formatted as a string.
    pub fn to_utf8_map(
        array: &ShreddedMapArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<MapArray> {
        to_map(&to_utf8(array, ctx)?, ctx)
    }

    /// Restricts the map to the entries whose key is in `keys`, keeping the shredded layout.
    pub fn project(
        array: &ShreddedMapArray,
        keys: &[&str],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ShreddedMapArray> {
        // Shredded keys can still have residual entries: null values and later duplicates.
        let residual = map::project(array.residual(), keys, ctx)?;
        let mut selected: Vec<usize> = keys.iter().filter_map(|k| find_column(array, k)).collect();
        selected.sort_unstable();
        selected.dedup();
        let columns = selected
            .iter()
            .map(|&c| array.data().columns()[c].clone())
            .collect();
        let column_arrays = selected
            .iter()
            .map(|&c| array.columns()[c].clone())
            .collect();
        // Equal rows stay equal after projection, so the repeats hint carries over.
        ShreddedMap::try_new_with_repeats(
            residual.into_array(),
            array.repeats().cloned(),
            columns,
            column_arrays,
        )
    }
}

/// Label operations over any encoding [`encode`](crate::encode) produces: a
/// `Dict(codes, ShreddedMap)`, a bare [`ShreddedMapArray`] or a canonical map.
///
/// For a dictionary, each operation runs once per distinct label map and expands through the
/// codes. Expanding a list-view only gathers `(offset, size)` pairs, so rows that share a label map
/// share its entries.
pub mod encoded {
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_array::arrays::listview::ListViewArraySlotsExt;
    use vortex_array::arrays::map::MapArrayExt;
    use vortex_array::arrays::map::MapArraySlotsExt;
    use vortex_buffer::BitBufferMut;

    use super::*;
    use crate::decode::codes_u32;
    use crate::flat::to_usize_vec;
    use crate::keyset;
    use crate::keyset::KeySetMap;
    use crate::keyset::KeySetMapArray;

    enum View {
        Dict {
            codes: ArrayRef,
            values: ShreddedMapArray,
        },
        Shredded(ShreddedMapArray),
        KeySet(KeySetMapArray),
        Map(ArrayRef),
    }

    fn view(array: &ArrayRef) -> View {
        if let Some(dict) = array.as_opt::<Dict>()
            && let Ok(values) = dict.values().clone().try_downcast::<ShreddedMap>()
        {
            return View::Dict {
                codes: dict.codes().clone(),
                values,
            };
        }
        match array.clone().try_downcast::<ShreddedMap>() {
            Ok(shredded) => View::Shredded(shredded),
            Err(array) => match array.try_downcast::<KeySetMap>() {
                Ok(keyset) => View::KeySet(keyset),
                Err(array) => View::Map(array),
            },
        }
    }

    /// Repeats the rows of `list` selected by `codes`, sharing their elements.
    fn expand_listview(
        list: &ListViewArray,
        codes: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListViewArray> {
        let codes = codes_u32(codes, ctx)?;
        let offsets = to_usize_vec(list.offsets(), ctx)?;
        let sizes = to_usize_vec(list.sizes(), ctx)?;
        let valid = list.listview_validity().execute_mask(list.len(), ctx)?;
        let mut new_offsets = Vec::with_capacity(codes.len());
        let mut new_sizes = Vec::with_capacity(codes.len());
        let mut new_valid = BitBufferMut::with_capacity(codes.len());
        for &c in &codes {
            let c = c as usize;
            new_offsets.push(offsets[c] as u64);
            new_sizes.push(sizes[c] as u64);
            new_valid.append(valid.value(c));
        }
        let offsets = PrimitiveArray::new(Buffer::from(new_offsets), Validity::NonNullable);
        let sizes = PrimitiveArray::new(Buffer::from(new_sizes), Validity::NonNullable);
        ListViewArray::try_new(
            list.elements().clone(),
            offsets.into_array(),
            sizes.into_array(),
            Validity::from_bit_buffer(new_valid.freeze(), list.nullability()),
        )
    }

    fn expand_map(
        map: &MapArray,
        codes: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<MapArray> {
        let entries = map.entries().clone().execute::<ListViewArray>(ctx)?;
        MapArray::try_new(
            map.map_dtype().clone(),
            expand_listview(&entries, codes, ctx)?,
        )
    }

    /// Decompresses into a canonical map.
    pub fn to_map(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<MapArray> {
        match view(array) {
            View::Dict { codes, values } => {
                expand_map(&shredded::to_map(&values, ctx)?, &codes, ctx)
            }
            View::Shredded(s) => shredded::to_map(&s, ctx),
            View::KeySet(k) => keyset::decode(&k, ctx),
            View::Map(m) => m.execute::<MapArray>(ctx),
        }
    }

    /// The keys of each row as a `List<Utf8>`.
    pub fn label_names(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
        match view(array) {
            View::Dict { codes, values } => {
                expand_listview(&shredded::label_names(&values, ctx)?, &codes, ctx)
            }
            View::Shredded(s) => shredded::label_names(&s, ctx),
            View::KeySet(k) => keyset::label_names(&k, ctx),
            View::Map(m) => map::label_names(&m, ctx),
        }
    }

    /// The distinct keys over all non-null rows, sorted.
    pub fn distinct_label_names(
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<String>> {
        match view(array) {
            View::Dict { values, .. } | View::Shredded(values) => {
                shredded::distinct_label_names(&values, ctx)
            }
            View::KeySet(k) => keyset::distinct_label_names(&k, ctx),
            View::Map(m) => map::distinct_label_names(&m, ctx),
        }
    }

    /// The value of `key` in each row formatted as a string, null when absent or null.
    pub fn get_label_utf8(
        array: &ArrayRef,
        key: &str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match view(array) {
            View::Dict { codes, values } => Ok(DictArray::try_new(
                codes,
                shredded::get_label_utf8(&values, key, ctx)?,
            )?
            .into_array()),
            View::Shredded(s) => shredded::get_label_utf8(&s, key, ctx),
            View::KeySet(k) => keyset::get_label_utf8(&k, key, ctx),
            View::Map(m) => map::get_label_utf8(&m, key, ctx),
        }
    }

    /// Decompresses into a map with every value formatted as a string.
    pub fn to_utf8_map(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<MapArray> {
        match view(array) {
            View::Dict { codes, values } => {
                expand_map(&shredded::to_utf8_map(&values, ctx)?, &codes, ctx)
            }
            View::Shredded(s) => shredded::to_utf8_map(&s, ctx),
            View::KeySet(k) => map::to_utf8_map(&keyset::decode(&k, ctx)?.into_array(), ctx),
            View::Map(m) => map::to_utf8_map(&m, ctx),
        }
    }

    /// Restricts the map to the entries whose key is in `keys`, keeping the encoding.
    pub fn project(
        array: &ArrayRef,
        keys: &[&str],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(match view(array) {
            View::Dict { codes, values } => {
                DictArray::try_new(codes, shredded::project(&values, keys, ctx)?.into_array())?
                    .into_array()
            }
            View::Shredded(s) => shredded::project(&s, keys, ctx)?.into_array(),
            View::KeySet(k) => keyset::project(&k, keys, ctx)?.into_array(),
            View::Map(m) => map::project(&m, keys, ctx)?.into_array(),
        })
    }
}

/// Filter-then-project queries: the rows whose label equals a value, then a few labels of those
/// rows, one `Utf8` array per requested key.
pub mod query {
    use super::*;

    /// The rows where `label` (a `Utf8` label array) equals `value`.
    pub fn label_eq(label: &ArrayRef, value: &str, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        let needle =
            ConstantArray::new(Scalar::utf8(value, Nullability::Nullable), label.len()).into_array();
        label.binary(needle, Operator::Eq)?.null_as_false().execute(ctx)
    }

    /// `SELECT keys WHERE map[filter_key] = value` over a shredded map. Only the filter key's
    /// column and the projected keys' columns are read; the residual is filtered once and only
    /// when a projected key lives there.
    pub fn shredded(
        array: &ShreddedMapArray,
        filter_key: &str,
        value: &str,
        keys: &[&str],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<VarBinViewArray>> {
        let label = shredded::get_label_utf8(array, filter_key, ctx)?;
        let mask = label_eq(&label, value, ctx)?;
        let mut residual = None;
        keys.iter()
            .map(|key| {
                let label = match shredded::find_column(array, key) {
                    Some(c) => values_to_utf8(&filter_column(&array.columns()[c], &mask, ctx)?, ctx)?,
                    None => {
                        let residual = match &residual {
                            Some(residual) => residual,
                            None => residual.insert(array.residual().filter(mask.clone())?),
                        };
                        map::get_label_utf8(residual, key, ctx)?
                    }
                };
                label.execute::<VarBinViewArray>(ctx)
            })
            .collect()
    }

    /// `SELECT keys WHERE map[filter_key] = value` over a map in any encoding, in one pass.
    ///
    /// Rows are scanned once. In each row, entries' keys are compared until the filter key is
    /// found, and only matching rows look at their other entries, so non-matching rows stop
    /// early as in a row-oriented scan. Dictionary keys and values compare their dictionaries
    /// once and then integer codes; plain strings compare inline view headers before bytes.
    /// Only the projected values of matching rows are gathered and decoded.
    pub fn map(
        map: &ArrayRef,
        filter_key: &str,
        value: &str,
        keys: &[&str],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<VarBinViewArray>> {
        let map = map.clone().execute::<MapArray>(ctx)?;
        let len = map.len();
        let entries = map.entries().clone().execute::<ListViewArray>(ctx)?;
        let elements = entries.elements().clone().execute::<StructArray>(ctx)?;
        let mut wanted = vec![filter_key];
        for key in keys {
            if !wanted.contains(key) {
                wanted.push(key);
            }
        }
        let key_array = elements.unmasked_field(0).clone();
        let key_strings;
        let key_source = match key_array.as_opt::<Dict>() {
            Some(dict) => Source::codes(dict, |s| {
                wanted.iter().position(|w| w.as_bytes() == s).map_or(0, |i| (i + 1) as u8)
            }, ctx)?,
            None => {
                key_strings = key_array.execute::<VarBinViewArray>(ctx)?;
                Source::views(&key_strings, &wanted, ctx)?
            }
        };
        let values = elements.unmasked_field(1);
        let value_strings;
        let value_source = match values.as_opt::<Dict>() {
            Some(dict) if matches!(dict.values().dtype(), DType::Utf8(_)) => {
                Source::codes(dict, |s| u8::from(s == value.as_bytes()), ctx)?
            }
            _ => {
                value_strings = values_to_utf8(values, ctx)?.execute::<VarBinViewArray>(ctx)?;
                Source::views(&value_strings, &[value], ctx)?
            }
        };
        let offsets = to_usize_vec(entries.offsets(), ctx)?;
        let sizes = to_usize_vec(entries.sizes(), ctx)?;
        let row_valid = map.map_validity().execute_mask(len, ctx)?;
        let key_slots: Vec<usize> = keys
            .iter()
            .map(|k| wanted.iter().position(|w| w == k).unwrap_or_default() + 1)
            .collect();
        let mut indices: Vec<Vec<u64>> = keys.iter().map(|_| Vec::new()).collect();
        let mut valid: Vec<vortex_buffer::BitBufferMut> = keys
            .iter()
            .map(|_| vortex_buffer::BitBufferMut::with_capacity(0))
            .collect();
        let mut first = vec![usize::MAX; wanted.len() + 1];
        for row in 0..len {
            if !row_valid.value(row) {
                continue;
            }
            let range = offsets[row]..offsets[row] + sizes[row];
            let Some(f) = range.clone().find(|&j| key_source.is(j, 1)) else {
                continue;
            };
            if !value_source.is(f, 1) {
                continue;
            }
            first.fill(usize::MAX);
            for j in range {
                let s = key_source.slot(j) as usize;
                if s != 0 && first[s] == usize::MAX {
                    first[s] = j;
                }
            }
            for (k, &s) in key_slots.iter().enumerate() {
                let j = first[s];
                indices[k].push(if j == usize::MAX { 0 } else { j as u64 });
                valid[k].append(j != usize::MAX);
            }
        }
        indices
            .into_iter()
            .zip(valid)
            .map(|(indices, valid)| {
                let indices = PrimitiveArray::new(
                    Buffer::from(indices),
                    Validity::from_bit_buffer(valid.freeze(), Nullability::Nullable),
                );
                let values = take_narrow(values, indices.into_array(), ctx)?;
                values_to_utf8(&values, ctx)?.execute::<VarBinViewArray>(ctx)
            })
            .collect()
    }

    /// Classifies entries of a string array against a few wanted strings on demand: `1 + i`
    /// for `wanted[i]`, 0 for anything else or null.
    enum Source<'a> {
        /// Dictionary-encoded: each distinct value is classified once, entries by their code.
        Codes { codes: Vec<u32>, table: Vec<u8> },
        /// Plain strings: a view's low 64 bits hold its length and first four bytes, ruling out
        /// almost every other string without touching string data.
        Views {
            strings: Strings<'a>,
            valid: Mask,
            headers: Vec<u64>,
            wanted: Vec<&'a [u8]>,
        },
    }

    impl<'a> Source<'a> {
        fn codes(
            dict: vortex_array::ArrayView<'_, Dict>,
            classify: impl Fn(&[u8]) -> u8,
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<Self> {
            let dictionary = dict.values().clone().execute::<VarBinViewArray>(ctx)?;
            let dict_valid = dictionary.as_ref().validity()?.execute_mask(dictionary.len(), ctx)?;
            let strings = Strings::new(&dictionary);
            let table = (0..dictionary.len())
                .map(|c| if dict_valid.value(c) { classify(strings.get(c)) } else { 0 })
                .collect();
            let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
            let valid = codes.validity()?.execute_mask(codes.len(), ctx)?;
            let mut codes = crate::decode::codes_u32(&codes.into_array(), ctx)?;
            if !valid.all_true() {
                for (i, c) in codes.iter_mut().enumerate() {
                    if !valid.value(i) {
                        *c = u32::MAX;
                    }
                }
            }
            Ok(Self::Codes { codes, table })
        }

        fn views(
            strings: &'a VarBinViewArray,
            wanted: &[&'a str],
            ctx: &mut ExecutionCtx,
        ) -> VortexResult<Self> {
            Ok(Self::Views {
                strings: Strings::new(strings),
                valid: strings.as_ref().validity()?.execute_mask(strings.len(), ctx)?,
                headers: wanted
                    .iter()
                    .map(|w| BinaryView::make_view(w.as_bytes(), 0, 0).as_u128() as u64)
                    .collect(),
                wanted: wanted.iter().map(|w| w.as_bytes()).collect(),
            })
        }

        /// Whether the entry is `wanted[slot - 1]`.
        #[inline]
        fn is(&self, entry: usize, slot: u8) -> bool {
            match self {
                Self::Codes { codes, table } => table.get(codes[entry] as usize) == Some(&slot),
                Self::Views {
                    strings,
                    valid,
                    headers,
                    wanted,
                } => {
                    let i = slot as usize - 1;
                    strings.views[entry].as_u128() as u64 == headers[i]
                        && valid.value(entry)
                        && strings.get(entry) == wanted[i]
                }
            }
        }

        #[inline]
        fn slot(&self, entry: usize) -> u8 {
            match self {
                Self::Codes { codes, table } => table.get(codes[entry] as usize).copied().unwrap_or(0),
                Self::Views {
                    strings,
                    valid,
                    headers,
                    wanted,
                } => {
                    let header = strings.views[entry].as_u128() as u64;
                    headers
                        .iter()
                        .position(|&h| h == header)
                        .filter(|&i| valid.value(entry) && strings.get(entry) == wanted[i])
                        .map_or(0, |i| (i + 1) as u8)
                }
            }
        }
    }
}
