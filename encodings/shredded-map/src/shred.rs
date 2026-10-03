// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting a canonical map into shredded key columns and a residual map.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::scalar::Scalar;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::patches::Patches;
use vortex_sparse::Sparse;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builders::dict::dict_encode;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
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
use crate::decode::codes_u32;
use crate::flat::Strings;
use crate::rowcmp::RowCmp;
use crate::flat::build_map;

/// Controls which keys [`shred`] moves into dedicated columns.
#[derive(Clone, Debug)]
pub struct ShredOptions {
    /// Minimum fraction of rows that must hold a non-null value for a key to be shredded.
    pub min_frequency: f64,
    /// Maximum number of sparse columns, for keys on fewer than `sparse_below` of the rows. The
    /// most frequent keys win. Keys on at least `sparse_below` of the rows are always shredded.
    pub max_sparse_columns: usize,
    /// Store a column as a single union variant when all of its values select that variant.
    pub typed: bool,
    /// Store each column as per-row codes into its distinct values instead of one value per row.
    /// With rows that repeat their labels this makes shredding, compression and decoding cheap.
    pub dictionary: bool,
    /// [`encode`] deduplicates rows when at most this fraction of rows starts a new run of equal
    /// label maps.
    pub max_distinct_rows: f64,
    /// With [`dictionary`](Self::dictionary), a column present on fewer than this fraction of
    /// rows stores only its present rows, as a sparse array.
    pub sparse_below: f64,
}

impl Default for ShredOptions {
    fn default() -> Self {
        Self {
            min_frequency: 0.01,
            max_sparse_columns: 64,
            typed: true,
            dictionary: true,
            max_distinct_rows: 0.5,
            sparse_below: 0.8,
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

/// Turns per-row codes into a column's distinct entries into per-row codes into its distinct
/// values. Value types the dictionary encoder does not support keep the entry codes.
fn dedup_values(
    codes: Vec<u32>,
    values: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<u32>, ArrayRef)> {
    let encodable = matches!(
        values.dtype(),
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    );
    let (codes, values) = match encodable.then(|| dict_encode(&values, ctx)) {
        Some(Ok(dict)) => {
            let remap = codes_u32(dict.codes(), ctx)?;
            let codes: Vec<u32> = codes.iter().map(|&c| remap[c as usize]).collect();
            (codes, dict.values().clone())
        }
        _ => (codes, values),
    };
    Ok((codes, values))
}

/// Builds a dictionary column from per-row codes. A column present on fewer than
/// `sparse_below` of the rows stores only its present rows, as a sparse array over a null fill.
fn dictionary_column(
    codes: Vec<u32>,
    valid: BitBuffer,
    values: ArrayRef,
    sparse_below: f64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = codes.len();
    let (codes, values) = dedup_values(codes, values, ctx)?;
    let values = into_nullable(values, ctx)?;
    let present = valid.true_count();
    #[allow(clippy::cast_precision_loss)]
    // Sparse arrays over union values are not supported by every kernel yet.
    let sparse_supported = !matches!(values.dtype(), DType::Union(..));
    if sparse_supported && (present as f64) < sparse_below * len as f64 {
        let rows: Vec<u64> = valid.set_indices().map(|i| i as u64).collect();
        let present_codes: Vec<u32> = valid.set_indices().map(|i| codes[i]).collect();
        let fill = Scalar::null(values.dtype().as_nullable());
        let values = DictArray::try_new(
            PrimitiveArray::new(Buffer::from(present_codes), Validity::NonNullable).into_array(),
            values,
        )?
        .into_array();
        // Chunk offsets let a point lookup jump to its 1024-row chunk instead of binary
        // searching every patch index.
        let n_chunks = len.div_ceil(PATCH_CHUNK_SIZE);
        let mut chunk_offsets = Vec::with_capacity(n_chunks);
        let mut next = 0usize;
        for chunk in 0..n_chunks {
            let chunk_start = (chunk * PATCH_CHUNK_SIZE) as u64;
            while next < rows.len() && rows[next] < chunk_start {
                next += 1;
            }
            chunk_offsets.push(next as u64);
        }
        let rows = PrimitiveArray::new(Buffer::from(rows), Validity::NonNullable).into_array();
        let chunk_offsets =
            PrimitiveArray::new(Buffer::from(chunk_offsets), Validity::NonNullable).into_array();
        let patches = Patches::new(len, 0, rows, values, Some(chunk_offsets))?;
        return Ok(Sparse::try_new_from_patches(patches, fill)?.into_array());
    }
    let codes = PrimitiveArray::new(
        Buffer::from(codes),
        Validity::from_bit_buffer(valid, Nullability::Nullable),
    );
    Ok(DictArray::try_new(codes.into_array(), values)?.into_array())
}

/// Makes `array` nullable without changing its values.
fn into_nullable(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if array.dtype().is_nullable() {
        return Ok(array);
    }
    if let DType::Union(..) = array.dtype() {
        let union = array.execute::<UnionArray>(ctx)?;
        let type_ids = union
            .type_ids()
            .cast(DType::Primitive(PType::U8, Nullability::Nullable))?;
        let children: Vec<ArrayRef> = union.iter_children().cloned().collect();
        return Ok(UnionArray::try_new(type_ids, union.variants().clone(), children)?.into_array());
    }
    array.cast(array.dtype().as_nullable())
}

/// Whether entry `j` is the first entry with its key in the row starting at `start`.
#[inline]
fn is_first(keys: &Strings<'_>, start: usize, j: usize) -> bool {
    j == start || keys.get(j) != keys.get(j - 1)
}

/// Marks the rows whose entries equal the previous row's, by shared range or by content.
fn repeated_rows(flat: &FlatMap, ctx: &mut ExecutionCtx) -> VortexResult<Vec<bool>> {
    let keys = Strings::new(&flat.keys);
    let values_cmp = RowCmp::new(&flat.values, ctx)?;
    Ok((0..flat.len)
        .map(|row| {
            flat.repeats_prev(row)
                || (row > 0 && flat.row_valid.value(row) && flat.row_valid.value(row - 1) && {
                    let (a, b) = (flat.range(row), flat.range(row - 1));
                    a.len() == b.len()
                        && a.zip(b).all(|(i, j)| {
                            keys.get(i) == keys.get(j) && values_cmp.equal(i, j)
                        })
                })
        })
        .collect())
}

/// Assigns every row the id of its distinct label map, returning the ids and the first row of
/// each distinct map. String values are matched across the whole array; other value types only
/// against the previous row.
fn row_dictionary(flat: &FlatMap, ctx: &mut ExecutionCtx) -> VortexResult<(Vec<u32>, Vec<usize>)> {
    let mut codes = Vec::with_capacity(flat.len);
    let mut first_rows: Vec<usize> = Vec::new();
    if matches!(flat.values.dtype(), DType::Utf8(_)) {
        let keys = Strings::new(&flat.keys);
        let values = flat.values.clone().execute::<VarBinViewArray>(ctx)?;
        let value_valid = values.validity()?.execute_mask(values.len(), ctx)?;
        let strings = Strings::new(&values);
        let mut ids: HashMap<Option<Vec<(&[u8], Option<&[u8]>)>>, u32> = HashMap::new();
        for row in 0..flat.len {
            if flat.repeats_prev(row) {
                codes.push(codes[row - 1]);
                continue;
            }
            let signature = flat.row_valid.value(row).then(|| {
                flat.range(row)
                    .map(|j| (keys.get(j), value_valid.value(j).then(|| strings.get(j))))
                    .collect::<Vec<_>>()
            });
            let next = u32::try_from(first_rows.len())?;
            let id = *ids.entry(signature).or_insert_with(|| {
                first_rows.push(row);
                next
            });
            codes.push(id);
        }
        return Ok((codes, first_rows));
    }
    for (row, repeat) in repeated_rows(flat, ctx)?.into_iter().enumerate() {
        if !repeat {
            first_rows.push(row);
        }
        #[allow(clippy::cast_possible_truncation)]
        codes.push((first_rows.len() - 1) as u32);
    }
    Ok((codes, first_rows))
}

/// Encodes a label map as `Dict(codes, ShreddedMap)`: runs of rows with equal label maps become
/// one code each, and only the distinct label maps are shredded.
///
/// This mirrors a time-series database's series index, where every sample references the
/// labels of its series. Operations run on the distinct label maps and expand through the codes.
/// When more than [`ShredOptions::max_distinct_rows`] of the rows start a new run, this returns
/// the [`ShreddedMapArray`] alone.
///
/// # Errors
///
/// Returns an error if `map` is not a `keys_sorted` map with UTF-8 keys.
pub fn encode(
    map: &ArrayRef,
    options: &ShredOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let flat = FlatMap::new(map, ctx)?;
    let (codes, first_rows) = row_dictionary(&flat, ctx)?;
    let distinct = first_rows.len();
    #[allow(clippy::cast_precision_loss)]
    if distinct as f64 > options.max_distinct_rows * flat.len as f64 {
        return Ok(shred(map, options, ctx)?.into_array());
    }

    let offsets = first_rows.iter().map(|&r| flat.offsets[r] as u64).collect();
    let sizes = first_rows.iter().map(|&r| flat.sizes[r] as u64).collect();
    let mut valid = BitBufferMut::with_capacity(distinct);
    for &row in &first_rows {
        valid.append(flat.row_valid.value(row));
    }
    let unique = build_map(
        &flat.map_dtype,
        flat.keys.clone().into_array(),
        flat.values.clone(),
        offsets,
        sizes,
        Validity::from_bit_buffer(valid.freeze(), map.dtype().nullability()),
    )?;
    let shredded = shred(&unique.into_array(), options, ctx)?;
    let codes = PrimitiveArray::new(Buffer::from(codes), Validity::NonNullable).into_array();
    Ok(DictArray::try_new(codes, shredded.into_array())?.into_array())
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
    // A row whose entries equal the previous row's, by range or by content, is routed like it
    // and shares its residual entries.
    let repeats = repeated_rows(&flat, ctx)?;

    // Rows that share the previous row's entries are counted once, weighted by the run length.
    let mut counts: HashMap<&[u8], usize> = HashMap::new();
    let mut row = 0;
    while row < len {
        let mut end = row + 1;
        while end < len && repeats[end] {
            end += 1;
        }
        let range = flat.range(row);
        for j in range.clone() {
            if is_first(&keys, range.start, j) && value_valid.value(j) {
                *counts.entry(keys.get(j)).or_default() += end - row;
            }
        }
        row = end;
    }

    #[allow(clippy::cast_precision_loss)]
    let threshold = options.min_frequency * len as f64;
    // Every key on at least `sparse_below` of the rows becomes a dense column. Of the rest, only
    // the most frequent `max_sparse_columns` keys become sparse columns, since each column adds
    // decoding work for every row.
    #[allow(clippy::cast_precision_loss)]
    let dense_threshold = options.sparse_below * len as f64;
    let (mut chosen, mut sparse): (Vec<(&[u8], usize)>, Vec<(&[u8], usize)>) = counts
        .into_iter()
        .filter(|&(_, count)| count > 0 && count as f64 >= threshold)
        .partition(|&(_, count)| count as f64 >= dense_threshold);
    sparse.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    sparse.truncate(options.max_sparse_columns);
    chosen.extend(sparse);
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
        // A row sharing the previous row's entries shares its routing too, which keeps the
        // residual shared and the columns run-length friendly.
        if repeats[row] {
            for c in 0..n_columns {
                if column_valid[c].value(row - 1) {
                    column_index[c][row] = column_index[c][row - 1];
                    column_valid[c].set(row);
                }
            }
            offsets.push(offsets[row - 1]);
            sizes.push(sizes[row - 1]);
            continue;
        }
        let start_kept = kept.len();
        let range = flat.range(row);
        for j in range.clone() {
            let column = (n_columns > 0 && is_first(&keys, range.start, j) && value_valid.value(j))
                .then(|| column_of.get(keys.get(j)))
                .flatten();
            match column {
                Some(&c) => {
                    // The row's code is the position of its entry among the column's distinct
                    // entries; rows that repeat the previous row reuse its code.
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        column_index[c][row] = column_entries[c].len() as u32;
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
        let entries =
            PrimitiveArray::new(Buffer::from(column_entries[c].clone()), Validity::NonNullable)
                .into_array();
        // Compact so the column does not pin the source's string buffers.
        let values = compact(source.take(entries)?, ctx)?;
        column_arrays.push(if options.dictionary {
            dictionary_column(index, valid.freeze(), values, options.sparse_below, ctx)?
        } else {
            let indices = PrimitiveArray::new(
                Buffer::from(index),
                Validity::from_bit_buffer(valid.freeze(), Nullability::Nullable),
            )
            .into_array();
            compact(values.take(indices)?, ctx)?
        });
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

    let repeats = repeats
        .iter()
        .any(|&r| r)
        .then(|| BoolArray::from_iter(repeats.iter().copied()).into_array());
    ShreddedMap::try_new_with_repeats(residual.into_array(), repeats, columns, column_arrays)
}
