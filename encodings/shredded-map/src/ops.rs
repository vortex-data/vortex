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
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
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
use crate::labels::values_to_utf8;

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
    pub fn get_label_utf8(
        map: &ArrayRef,
        key: &str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let flat = FlatMap::new(map, ctx)?;
        let keys = Strings::new(&flat.keys);
        let needle = key.as_bytes();
        let mut indices = Vec::with_capacity(flat.len);
        let mut valid = vortex_buffer::BitBufferMut::with_capacity(flat.len);
        let mut found = None;
        for row in 0..flat.len {
            if !flat.repeats_prev(row) {
                found = flat.range(row).find(|&j| keys.get(j) == needle);
            }
            indices.push(found.unwrap_or(0) as u64);
            valid.append(found.is_some());
        }
        let indices = PrimitiveArray::new(
            Buffer::from(indices),
            Validity::from_bit_buffer(valid.freeze(), Nullability::Nullable),
        );
        let values = flat
            .values
            .take(indices.into_array())?
            .execute::<Canonical>(ctx)?
            .into_array();
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

    fn find_column(array: &ShreddedMapArray, key: &str) -> Option<usize> {
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
