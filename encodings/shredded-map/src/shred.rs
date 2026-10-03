// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting a canonical map into shredded key columns and a residual map.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ShreddedColumn;
use crate::ShreddedMap;
use crate::ShreddedMapArray;
use crate::flat::FlatMap;
use crate::flat::Strings;
use crate::flat::build_map;

/// Controls which keys [`shred`] moves into dedicated columns.
#[derive(Clone, Debug)]
pub struct ShredOptions {
    /// Minimum fraction of rows that must hold a non-null value for a key to be shredded.
    pub min_frequency: f64,
    /// Maximum number of shredded columns. The most frequent keys win.
    pub max_columns: usize,
    /// Store a column as a single union variant when all of its values select that variant.
    pub typed: bool,
}

impl Default for ShredOptions {
    fn default() -> Self {
        Self {
            min_frequency: 0.05,
            max_columns: 64,
            typed: true,
        }
    }
}

/// Materializes `array`, copying out only the string bytes it references, including inside
/// union children.
fn compact(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    match array.execute::<Canonical>(ctx)? {
        Canonical::Union(union) => {
            let children = union
                .iter_children()
                .map(|c| compact(c.clone(), ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            let type_ids = compact(union.type_ids().clone(), ctx)?;
            Ok(UnionArray::try_new(type_ids, union.variants().clone(), children)?.into_array())
        }
        canonical => Ok(canonical.compact(ctx)?.into_array()),
    }
}

/// Whether entry `j` is the first entry with its key in the row starting at `start`.
#[inline]
fn is_first(keys: &Strings<'_>, start: usize, j: usize) -> bool {
    j == start || keys.get(j) != keys.get(j - 1)
}

/// Shreds a map with UTF-8 keys and `keys_sorted = true` into a [`ShreddedMapArray`].
///
/// # Errors
///
/// Returns an error if `map` is not such a map.
pub fn shred(
    map: &ArrayRef,
    options: &ShredOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ShreddedMapArray> {
    let flat = FlatMap::new(map, ctx)?;
    vortex_ensure!(
        matches!(flat.map_dtype.key_dtype(), DType::Utf8(_)) && flat.map_dtype.keys_sorted(),
        "shredding requires a keys_sorted map with UTF-8 keys, got {}",
        map.dtype()
    );
    let len = flat.len;
    let keys = Strings::new(&flat.keys);
    let value_valid = flat
        .values
        .validity()?
        .execute_mask(flat.values.len(), ctx)?;

    // Count, per key, the rows whose first entry with that key has a non-null value.
    let mut counts: HashMap<&[u8], usize> = HashMap::new();
    for row in 0..len {
        let range = flat.range(row);
        for j in range.clone() {
            if is_first(&keys, range.start, j) && value_valid.value(j) {
                *counts.entry(keys.get(j)).or_default() += 1;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let threshold = options.min_frequency * len as f64;
    let mut chosen: Vec<(&[u8], usize)> = counts
        .into_iter()
        .filter(|&(_, count)| count > 0 && count as f64 >= threshold)
        .collect();
    chosen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    chosen.truncate(options.max_columns);
    chosen.sort_by(|a, b| a.0.cmp(b.0));
    let column_of: HashMap<&[u8], usize> = chosen
        .iter()
        .enumerate()
        .map(|(i, &(key, _))| (key, i))
        .collect();

    // Route every entry to a column or to the residual.
    let n_columns = chosen.len();
    vortex_ensure!(
        n_columns <= u16::MAX as usize,
        "at most {} shredded columns are supported",
        u16::MAX
    );
    vortex_ensure!(
        u32::try_from(flat.values.len()).is_ok(),
        "shredding supports at most u32::MAX entries"
    );
    let mut column_index = vec![vec![0u32; len]; n_columns];
    let mut column_valid: Vec<BitBufferMut> = (0..n_columns)
        .map(|_| BitBufferMut::new_unset(len))
        .collect();
    let mut column_entries: Vec<Vec<u64>> = vec![Vec::new(); n_columns];
    let mut kept = Vec::with_capacity(flat.keys.len());
    let mut offsets = Vec::with_capacity(len);
    let mut sizes = Vec::with_capacity(len);
    for row in 0..len {
        let start_kept = kept.len();
        let range = flat.range(row);
        for j in range.clone() {
            let column = (n_columns > 0 && is_first(&keys, range.start, j) && value_valid.value(j))
                .then(|| column_of.get(keys.get(j)))
                .flatten();
            match column {
                Some(&c) => {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        column_index[c][row] = j as u32;
                    }
                    column_valid[c].set(row);
                    column_entries[c].push(j as u64);
                }
                None => kept.push(j as u64),
            }
        }
        offsets.push(start_kept as u64);
        sizes.push((kept.len() - start_kept) as u64);
    }

    let union_values = match flat.values.dtype() {
        DType::Union(..) if options.typed => {
            Some(flat.values.clone().execute::<UnionArray>(ctx)?)
        }
        _ => None,
    };
    let union_type_ids = union_values
        .as_ref()
        .map(|u| u.type_ids().clone().execute::<PrimitiveArray>(ctx))
        .transpose()?;
    let mut child_valid: Vec<Option<Mask>> = union_values
        .as_ref()
        .map(|u| vec![None; u.variants().len()])
        .unwrap_or_default();

    let mut columns = Vec::with_capacity(n_columns);
    let mut column_arrays = Vec::with_capacity(n_columns);
    for (c, ((key, _), (index, valid))) in chosen
        .iter()
        .zip(column_index.into_iter().zip(column_valid))
        .enumerate()
    {
        let indices = PrimitiveArray::new(
            Buffer::from(index),
            Validity::from_bit_buffer(valid.freeze(), vortex_array::dtype::Nullability::Nullable),
        )
        .into_array();

        let mut variant = None;
        if let (Some(union), Some(type_ids)) = (&union_values, &union_type_ids) {
            let tags = type_ids.as_slice::<u8>();
            let entries = &column_entries[c];
            let tag = tags[entries[0] as usize];
            if entries.iter().all(|&j| tags[j as usize] == tag)
                && let Some(child) = union.variants().tag_to_child_index(tag)
            {
                let child_array = union.child(child).cloned().unwrap_or_else(|| unreachable!());
                if child_valid[child].is_none() {
                    child_valid[child] =
                        Some(child_array.validity()?.execute_mask(child_array.len(), ctx)?);
                }
                let child_mask = child_valid[child].as_ref().unwrap_or_else(|| unreachable!());
                if entries.iter().all(|&j| child_mask.value(j as usize)) {
                    variant = Some(child);
                }
            }
        }

        let source = match variant {
            Some(child) => union_values
                .as_ref()
                .and_then(|u| u.child(child).cloned())
                .unwrap_or_else(|| unreachable!()),
            None => flat.values.clone(),
        };
        // Compact so the column does not pin the source's string buffers.
        column_arrays.push(compact(
            source
                .take(indices)?,
            ctx,
        )?
        );
        columns.push(ShreddedColumn {
            key: Arc::from(std::str::from_utf8(key).unwrap_or_else(|_| unreachable!())),
            variant,
        });
    }

    let kept = PrimitiveArray::new(Buffer::from(kept), Validity::NonNullable).into_array();
    let residual_keys = flat
        .keys
        .clone()
        .into_array()
        .take(kept.clone())?
        .execute::<Canonical>(ctx)?
        .compact(ctx)?
        .into_array();
    let residual_values = compact(flat.values.take(kept)?, ctx)?;
    let residual = build_map(
        &flat.map_dtype,
        residual_keys,
        residual_values,
        offsets,
        sizes,
        flat.validity.clone(),
    )?;

    ShreddedMap::try_new(residual.into_array(), columns, column_arrays)
}
