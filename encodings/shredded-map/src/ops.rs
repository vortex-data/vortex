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
        for row in 0..flat.len {
            let found = flat.range(row).find(|&j| keys.get(j) == needle);
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
        ShreddedMap::try_new(residual, columns, column_arrays)
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
        ShreddedMap::try_new(residual.into_array(), columns, column_arrays)
    }
}
