// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::patches::PatchesMetadata;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Reference model for a [`Patches`] over `i32` values: `None` means unpatched at that logical
/// position, `Some(scalar)` means patched to `scalar` (which may itself be null when nullable).
struct PatchesModel {
    entries: Vec<Option<Scalar>>,
    dtype: DType,
}

impl PatchesModel {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn patch_positions(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_some().then_some(i))
            .collect()
    }

    fn patch_values_array(&self) -> PrimitiveArray {
        if self.dtype.is_nullable() {
            PrimitiveArray::from_option_iter(
                self.entries
                    .iter()
                    .filter_map(|v| v.as_ref().map(|s| s.as_primitive().typed_value::<i32>())),
            )
        } else {
            PrimitiveArray::from_iter(self.entries.iter().filter_map(|v| {
                v.as_ref().map(|s| {
                    s.as_primitive()
                        .typed_value::<i32>()
                        .vortex_expect("non-null")
                })
            }))
        }
    }

    fn slice(&self, range: Range<usize>) -> Self {
        Self {
            entries: self.entries[range].to_vec(),
            dtype: self.dtype.clone(),
        }
    }

    fn filter(&self, mask: &Mask) -> Self {
        let entries = self
            .entries
            .iter()
            .enumerate()
            .filter(|&(i, _)| mask.value(i))
            .map(|(_, v)| v.clone())
            .collect();
        Self {
            entries,
            dtype: self.dtype.clone(),
        }
    }

    fn mask(&self, mask: &Mask) -> Self {
        let nullable_dtype = self.dtype.as_nullable();
        let entries = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if mask.value(i) {
                    None
                } else {
                    v.as_ref()
                        .map(|s| s.cast(&nullable_dtype).vortex_expect("widen to nullable"))
                }
            })
            .collect();
        Self {
            entries,
            dtype: nullable_dtype,
        }
    }

    fn take(&self, indices: &[Option<usize>], indices_nullable: bool, include_nulls: bool) -> Self {
        let out_dtype = self
            .dtype
            .union_nullability(Nullability::from(indices_nullable));
        let entries = indices
            .iter()
            .map(|idx| match idx {
                None => include_nulls.then(|| Scalar::null(out_dtype.clone())),
                Some(i) => self.entries.get(*i).and_then(|v| {
                    v.as_ref()
                        .map(|s| s.cast(&out_dtype).vortex_expect("cast to take nullability"))
                }),
            })
            .collect();
        Self {
            entries,
            dtype: out_dtype,
        }
    }
}

fn assert_patches_match_model(
    actual: Option<&Patches>,
    model: &PatchesModel,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let expected_positions = model.patch_positions();
    let Some(actual) = actual else {
        assert!(
            expected_positions.is_empty(),
            "expected {} patches, got None",
            expected_positions.len()
        );
        return Ok(());
    };

    assert!(!expected_positions.is_empty(), "expected None, got Some");
    assert_eq!(actual.array_len(), model.len());
    assert_eq!(actual.dtype(), &model.dtype);
    assert_eq!(actual.num_patches(), expected_positions.len());
    assert_eq!(actual.min_index(ctx)?, expected_positions[0]);
    assert_eq!(
        actual.max_index(ctx)?,
        *expected_positions.last().vortex_expect("non-empty")
    );

    let actual_indices = actual.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = actual.offset();
    let actual_logical_indices: Vec<usize> = (0..actual_indices.len())
        .map(|i| {
            usize::try_from(&actual_indices.execute_scalar(i, ctx).vortex_expect("idx"))
                .vortex_expect("usize")
                - offset
        })
        .collect();
    assert_eq!(actual_logical_indices, expected_positions);
    assert_arrays_eq!(actual.values(), model.patch_values_array(), ctx);

    // Probe search_index & get_patched at every patch position, adjacent gaps, boundaries,
    // and past the end of the array (including past the last chunk).
    let mut probe_points = BTreeSet::from([
        0usize,
        model.len().saturating_sub(1),
        model.len(),
        model.len() + 1,
        model.len() + PATCH_CHUNK_SIZE,
    ]);
    for &p in &expected_positions {
        probe_points.insert(p);
        if p > 0 {
            probe_points.insert(p - 1);
        }
        if p + 1 < model.len() {
            probe_points.insert(p + 1);
        }
    }
    for idx in probe_points {
        let expected_search = match expected_positions.binary_search(&idx) {
            Ok(pos) => SearchResult::Found(pos),
            Err(pos) => SearchResult::NotFound(pos),
        };
        assert_eq!(
            actual.search_index(idx, ctx)?,
            expected_search,
            "search_index({idx}) mismatch"
        );
        if idx < model.len() {
            assert_eq!(
                actual.get_patched(idx, ctx)?,
                model.entries[idx],
                "get_patched({idx}) mismatch"
            );
        }
    }

    Ok(())
}

fn make_unsigned_indices(ptype: PType, raw: &[usize]) -> ArrayRef {
    match ptype {
        PType::U8 => raw
            .iter()
            .map(|&v| u8::try_from(v).vortex_expect("u8"))
            .collect::<Buffer<u8>>()
            .into_array(),
        PType::U16 => raw
            .iter()
            .map(|&v| u16::try_from(v).vortex_expect("u16"))
            .collect::<Buffer<u16>>()
            .into_array(),
        PType::U32 => raw
            .iter()
            .map(|&v| u32::try_from(v).vortex_expect("u32"))
            .collect::<Buffer<u32>>()
            .into_array(),
        PType::U64 => raw
            .iter()
            .map(|&v| v as u64)
            .collect::<Buffer<u64>>()
            .into_array(),
        other => unreachable!("non-unsigned ptype {other}"),
    }
}

fn make_take_indices(ptype: PType, indices: &[Option<usize>], nullable: bool) -> PrimitiveArray {
    let validity = if nullable {
        Validity::from_iter(indices.iter().map(Option::is_some))
    } else {
        Validity::NonNullable
    };
    let vals = indices.iter().map(|opt| opt.unwrap_or(0));
    match ptype {
        PType::U8 => PrimitiveArray::new(
            vals.map(|v| u8::try_from(v).vortex_expect("u8"))
                .collect::<Buffer<u8>>(),
            validity,
        ),
        PType::U16 => PrimitiveArray::new(
            vals.map(|v| u16::try_from(v).vortex_expect("u16"))
                .collect::<Buffer<u16>>(),
            validity,
        ),
        PType::U32 => PrimitiveArray::new(
            vals.map(|v| u32::try_from(v).vortex_expect("u32"))
                .collect::<Buffer<u32>>(),
            validity,
        ),
        PType::U64 => {
            PrimitiveArray::new(vals.map(|v| v as u64).collect::<Buffer<u64>>(), validity)
        }
        PType::I32 => PrimitiveArray::new(
            vals.map(|v| i32::try_from(v).vortex_expect("i32"))
                .collect::<Buffer<i32>>(),
            validity,
        ),
        PType::I64 => PrimitiveArray::new(
            vals.map(|v| i64::try_from(v).vortex_expect("i64"))
                .collect::<Buffer<i64>>(),
            validity,
        ),
        other => unreachable!("unsupported take ptype {other}"),
    }
}

/// Draws a valid [`Patches`] (with `u8`/`u16`/`u32`/`u64` indices, optional non-zero offset,
/// optional `chunk_offsets` spanning multiple `PATCH_CHUNK_SIZE` chunks, nullable or non-nullable
/// values, and optional pre-slice) alongside its reference [`PatchesModel`].
fn draw_patches_and_model(
    tc: &TestCase,
    allow_multi_chunk: bool,
    ctx: &mut ExecutionCtx,
) -> (Patches, PatchesModel) {
    let use_chunks = allow_multi_chunk && tc.draw(gs::booleans());
    let array_len = if use_chunks {
        tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(3 * PATCH_CHUNK_SIZE + 64),
        )
    } else {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(96))
    };

    // Choose offset: 0 when using chunk_offsets (we obtain non-zero offset_within_chunk via slice
    // below), or arbitrary small offset when unchunked.
    let offset = if use_chunks || tc.draw(gs::booleans()) {
        0
    } else {
        tc.draw(gs::integers::<usize>().max_value(32))
    };

    // Draw sorted unique logical patch positions, biasing toward boundaries (0, len-1, chunk edges).
    let max_patches = array_len.min(if use_chunks { 24 } else { 32 });
    let num_drawn = tc.draw(gs::integers::<usize>().min_value(1).max_value(max_patches));
    let mut positions = BTreeSet::new();
    for _ in 0..num_drawn {
        let pos = if use_chunks && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
            // Focus on chunk boundaries like 1023, 1024, 2047, 2048.
            let chunk = tc.draw(gs::integers::<usize>().max_value(array_len / PATCH_CHUNK_SIZE));
            let delta = tc.draw(gs::integers::<usize>().max_value(2));
            (chunk * PATCH_CHUNK_SIZE)
                .saturating_add(delta)
                .saturating_sub(1)
                .min(array_len - 1)
        } else {
            tc.draw(gs::integers::<usize>().max_value(array_len - 1))
        };
        positions.insert(pos);
    }
    let positions: Vec<usize> = positions.into_iter().collect();
    let raw_indices: Vec<usize> = positions.iter().map(|&p| p + offset).collect();
    let max_raw = *raw_indices.last().vortex_expect("non-empty");

    // Pick the widest range of valid unsigned index ptypes that fit `max_raw`.
    let mut index_ptypes = vec![PType::U32, PType::U64];
    if max_raw <= usize::from(u16::MAX) {
        index_ptypes.push(PType::U16);
    }
    if max_raw <= usize::from(u8::MAX) {
        index_ptypes.push(PType::U8);
    }
    let index_ptype =
        index_ptypes[tc.draw(gs::integers::<usize>().max_value(index_ptypes.len() - 1))];
    let indices_array = make_unsigned_indices(index_ptype, &raw_indices);

    let nullable = tc.draw(gs::booleans());
    let dtype = DType::Primitive(PType::I32, Nullability::from(nullable));
    let mut entries = vec![None; array_len];
    let values_array = if nullable {
        let vals: Vec<Option<i32>> = positions
            .iter()
            .map(|&p| {
                let v = if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                    None
                } else {
                    Some(tc.draw(gs::integers::<i32>().min_value(-500).max_value(500)))
                };
                entries[p] = Some(match v {
                    Some(x) => Scalar::primitive(x, Nullability::Nullable),
                    None => Scalar::null(dtype.clone()),
                });
                v
            })
            .collect();
        PrimitiveArray::from_option_iter(vals).into_array()
    } else {
        let vals: Vec<i32> = positions
            .iter()
            .map(|&p| {
                let x = tc.draw(gs::integers::<i32>().min_value(-500).max_value(500));
                entries[p] = Some(Scalar::primitive(x, Nullability::NonNullable));
                x
            })
            .collect();
        PrimitiveArray::from_iter(vals).into_array()
    };

    let chunk_offsets = use_chunks.then(|| {
        let num_chunks = array_len.div_ceil(PATCH_CHUNK_SIZE);
        let offsets: Vec<u64> = (0..num_chunks)
            .map(|c| positions.partition_point(|&p| p < c * PATCH_CHUNK_SIZE) as u64)
            .collect();
        Buffer::from(offsets).into_array()
    });

    let patches = Patches::new(
        array_len,
        offset,
        indices_array,
        values_array,
        chunk_offsets,
        ctx,
    )
    .vortex_expect("valid generated patches");
    let model = PatchesModel { entries, dtype };

    // Optionally pre-slice so downstream operations also see non-zero offset & offset_within_chunk.
    if array_len > 1 && tc.draw(gs::booleans()) {
        let a = tc.draw(gs::integers::<usize>().max_value(array_len - 1));
        let b = tc.draw(
            gs::integers::<usize>()
                .min_value(a + 1)
                .max_value(array_len),
        );
        if let Some(sliced) = patches.slice(a..b, ctx).vortex_expect("slice") {
            return (sliced, model.slice(a..b));
        }
    }

    (patches, model)
}

fn draw_mask(tc: &TestCase, len: usize) -> Mask {
    match tc.draw(gs::integers::<u8>().max_value(5)) {
        0 => Mask::new_true(len),
        1 => Mask::new_false(len),
        2 => {
            // Sparse mask with cached indices.
            let count = tc.draw(gs::integers::<usize>().max_value(len.min(16)));
            let mut set = BTreeSet::new();
            for _ in 0..count {
                set.insert(tc.draw(gs::integers::<usize>().max_value(len - 1)));
            }
            Mask::from_indices(len, set)
        }
        3 => {
            // Sliced BitBuffer with non-zero bit offset, optionally with cached indices.
            let bit_offset = tc.draw(gs::integers::<usize>().min_value(1).max_value(11));
            let pattern: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(1).max_size(17));
            let bits = BitBuffer::collect_bool(len + bit_offset, |i| pattern[i % pattern.len()]);
            let mask = Mask::from_buffer(bits.slice(bit_offset..bit_offset + len));
            if tc.draw(gs::booleans())
                && let Some(values) = mask.values()
            {
                let _ = values.indices();
            }
            mask
        }
        _ => {
            let bits: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
            Mask::from_iter(bits)
        }
    }
}

/// Builds piecewise indices, encoding uniform lengths or multipliers as constants.
fn piecewise(starts: &[u64], lengths: &[u64], multipliers: &[u64]) -> ArrayRef {
    fn param(values: &[u64]) -> ArrayRef {
        match values.first() {
            Some(&first) if values.iter().all(|&v| v == first) => {
                ConstantArray::new(first, values.len()).into_array()
            }
            _ => Buffer::copy_from(values).into_array(),
        }
    }
    let len = usize::try_from(lengths.iter().sum::<u64>()).vortex_expect("length fits in usize");
    PiecewiseSequenceArray::try_new(
        Buffer::copy_from(starts).into_array(),
        param(lengths),
        param(multipliers),
        len,
    )
    .vortex_expect("valid piecewise indices")
    .into_array()
}

/// Differential model for `search_index`, `get_patched`, `min_index`/`max_index`, `map_values`
/// (including cast), `to_metadata`/`from_metadata` roundtrip, and chained `slice` operations across
/// chunked & unchunked patches and all unsigned index widths.
#[hegel::test(test_cases = 100)]
fn hegel_patches_search_slice_map_and_metadata(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let (mut patches, mut model) = draw_patches_and_model(&tc, true, &mut ctx);
    assert_patches_match_model(Some(&patches), &model, &mut ctx).vortex_expect("initial match");

    // Verify metadata roundtrip before slicing.
    let meta = patches
        .to_metadata(patches.array_len(), patches.dtype())
        .vortex_expect("to_metadata");
    let roundtripped = Patches::from_metadata(
        patches.array_len(),
        &meta,
        patches.indices().clone(),
        patches.values().clone(),
        patches.chunk_offsets().clone(),
        &mut ctx,
    )
    .vortex_expect("from_metadata");
    assert_eq!(
        roundtripped.offset_within_chunk(),
        patches.offset_within_chunk()
    );
    assert_patches_match_model(Some(&roundtripped), &model, &mut ctx)
        .vortex_expect("metadata roundtrip");

    // Verify map_values preserves indices/offsets and transforms values (including casting).
    let target_dtype = DType::Primitive(PType::I64, model.dtype.nullability());
    let mapped = patches
        .clone()
        .map_values(|v| v.cast(target_dtype.clone()))
        .vortex_expect("map_values cast");
    assert_eq!(mapped.dtype(), &target_dtype);
    assert_eq!(mapped.num_patches(), patches.num_patches());
    assert_arrays_eq!(mapped.indices(), patches.indices(), &mut ctx);
    assert_arrays_eq!(
        mapped.values(),
        patches.values().cast(target_dtype).vortex_expect("cast"),
        &mut ctx
    );

    // Chain up to 3 slices (including dropping leading chunks and empty slices).
    let steps = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
    for _ in 0..steps {
        let len = model.len();
        let start = tc.draw(gs::integers::<usize>().max_value(len));
        let end = tc.draw(gs::integers::<usize>().min_value(start).max_value(len));
        let next_model = model.slice(start..end);
        let next_patches = patches.slice(start..end, &mut ctx).vortex_expect("slice");
        assert_patches_match_model(next_patches.as_ref(), &next_model, &mut ctx)
            .vortex_expect("sliced match");

        let Some(next) = next_patches else {
            break;
        };
        let meta = next
            .to_metadata(next.array_len(), next.dtype())
            .vortex_expect("sliced to_metadata");
        let rt = Patches::from_metadata(
            next.array_len(),
            &meta,
            next.indices().clone(),
            next.values().clone(),
            next.chunk_offsets().clone(),
            &mut ctx,
        )
        .vortex_expect("sliced from_metadata");
        assert_eq!(rt.offset_within_chunk(), next.offset_within_chunk());
        assert_patches_match_model(Some(&rt), &next_model, &mut ctx)
            .vortex_expect("sliced metadata roundtrip");

        patches = next;
        model = next_model;
    }
}

/// Differential model for `filter` (all 3 internal algorithms: patch gallop, mask gallop, and
/// SIMD bitmap rank with sliced bit offsets) and `mask` (nullable widening + patch removal).
#[hegel::test(test_cases = 120)]
fn hegel_patches_filter_and_mask(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let make_patches_from_positions =
        |len: usize, offset: usize, positions: Vec<usize>, ctx: &mut ExecutionCtx| {
            let raw: Vec<usize> = positions.iter().map(|&p| p + offset).collect();
            let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
            let mut entries = vec![None; len];
            let vals: Vec<i32> = positions
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let val = i32::try_from(i).vortex_expect("fits");
                    entries[p] = Some(Scalar::primitive(val, Nullability::NonNullable));
                    val
                })
                .collect();
            let patches = Patches::new(
                len,
                offset,
                make_unsigned_indices(PType::U32, &raw),
                PrimitiveArray::from_iter(vals).into_array(),
                None,
                ctx,
            )
            .vortex_expect("valid patches");
            (patches, PatchesModel { entries, dtype })
        };

    // Cover every internal filter branch deterministically across draws:
    // 0: very sparse patches + cached mask indices (`expected_matches * 256 < len`,
    //    `patch_count <= true_count` -> `gallop_mask_indices_for_patches`),
    // 1: very sparse cached mask (`expected_matches * 256 < len`, `patch_count > true_count` ->
    //    `gallop_patches_for_mask_indices`),
    // 2: dense large arrays (`rank_patches_in_mask_bitmap` across 64-bit words + gallop paths),
    // 3: arbitrary generated patches (nullable, all index widths, pre-sliced).
    let (patches, model, mask) = match tc.draw(gs::integers::<u8>().max_value(3)) {
        0 => {
            let len = tc.draw(gs::integers::<usize>().min_value(800).max_value(1500));
            let offset = tc.draw(gs::integers::<usize>().max_value(16));
            let patch_count = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
            let mut pos = BTreeSet::new();
            for _ in 0..patch_count {
                pos.insert(tc.draw(gs::integers::<usize>().max_value(len - 1)));
            }
            let positions: Vec<usize> = pos.into_iter().collect();
            let mask_count = tc.draw(
                gs::integers::<usize>()
                    .min_value(positions.len())
                    .max_value(64),
            );
            let mut mpos = BTreeSet::new();
            for _ in 0..mask_count {
                mpos.insert(tc.draw(gs::integers::<usize>().max_value(len - 1)));
            }
            if tc.draw(gs::booleans()) {
                mpos.insert(positions[0]);
            }
            let (patches, model) = make_patches_from_positions(len, offset, positions, &mut ctx);
            (patches, model, Mask::from_indices(len, mpos))
        }
        1 => {
            let len = tc.draw(gs::integers::<usize>().min_value(800).max_value(1500));
            let offset = tc.draw(gs::integers::<usize>().max_value(16));
            let patch_count = tc.draw(gs::integers::<usize>().min_value(4).max_value(48));
            let mut pos = BTreeSet::new();
            for _ in 0..patch_count {
                pos.insert(tc.draw(gs::integers::<usize>().max_value(len - 1)));
            }
            let positions: Vec<usize> = pos.into_iter().collect();
            let mask_count = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
            let mut mpos = BTreeSet::new();
            for _ in 0..mask_count {
                mpos.insert(tc.draw(gs::integers::<usize>().max_value(len - 1)));
            }
            if tc.draw(gs::booleans()) {
                mpos.insert(*positions.last().vortex_expect("non-empty"));
            }
            let (patches, model) = make_patches_from_positions(len, offset, positions, &mut ctx);
            (patches, model, Mask::from_indices(len, mpos))
        }
        2 => {
            let len = tc.draw(gs::integers::<usize>().min_value(64).max_value(1000));
            let step = tc.draw(gs::integers::<usize>().min_value(1).max_value(97));
            let offset = tc.draw(gs::integers::<usize>().max_value(16));
            let positions: Vec<usize> = (0..len).step_by(step).collect();
            let (patches, model) = make_patches_from_positions(len, offset, positions, &mut ctx);
            let mask = draw_mask(&tc, len);
            (patches, model, mask)
        }
        _ => {
            let (patches, model) = draw_patches_and_model(&tc, false, &mut ctx);
            let mask = draw_mask(&tc, model.len());
            (patches, model, mask)
        }
    };

    let filtered = patches.filter(&mask, &mut ctx).vortex_expect("filter");
    assert_patches_match_model(filtered.as_ref(), &model.filter(&mask), &mut ctx)
        .vortex_expect("filter match");

    let masked = patches.mask(&mask, &mut ctx).vortex_expect("mask");
    assert_patches_match_model(masked.as_ref(), &model.mask(&mask), &mut ctx)
        .vortex_expect("mask match");
}

/// Differential model for `take`, `take_with_nulls`, `take_search`, `take_map`, and piecewise
/// `take_sorted_runs` across all take index widths, nullability, duplicates, dense/long runs,
/// and out-of-order runs.
#[hegel::test(test_cases = 120)]
fn hegel_patches_take_all_paths(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let dense_mode = tc.draw(gs::integers::<u8>().max_value(3)) == 0;
    let (patches, model) = if dense_mode {
        // Dense patches (step 1..=2) stress `ValueRuns::push` adjacent-run merging and
        // piece-galloping in `merge_sorted_runs`.
        let len = tc.draw(gs::integers::<usize>().min_value(32).max_value(220));
        let step = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
        let offset = tc.draw(gs::integers::<usize>().max_value(8));
        let positions: Vec<usize> = (0..len).step_by(step).collect();
        let raw: Vec<usize> = positions.iter().map(|&p| p + offset).collect();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut entries = vec![None; len];
        let vals: Vec<i32> = positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let val = i32::try_from(i).vortex_expect("fits");
                entries[p] = Some(Scalar::primitive(val, Nullability::NonNullable));
                val
            })
            .collect();
        let patches = Patches::new(
            len,
            offset,
            make_unsigned_indices(PType::U32, &raw),
            PrimitiveArray::from_iter(vals).into_array(),
            None,
            &mut ctx,
        )
        .vortex_expect("dense patches");
        (patches, PatchesModel { entries, dtype })
    } else {
        draw_patches_and_model(&tc, true, &mut ctx)
    };
    let len = model.len();

    // 1. Primitive take indices: verify take, take_with_nulls, take_search, and take_map all agree.
    let take_len = tc.draw(gs::integers::<usize>().max_value(32));
    let nullable_indices = tc.draw(gs::booleans());
    let mut max_idx = 0usize;
    let logical_indices: Vec<Option<usize>> = (0..take_len)
        .map(|_| {
            if nullable_indices && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                None
            } else {
                // Include occasional out-of-range indices (like len + 5) which must match nothing.
                let idx = if tc.draw(gs::integers::<u8>().max_value(7)) == 0 {
                    len + tc.draw(gs::integers::<usize>().max_value(16))
                } else {
                    tc.draw(gs::integers::<usize>().max_value(len - 1))
                };
                max_idx = max_idx.max(idx);
                Some(idx)
            }
        })
        .collect();

    let mut take_ptypes = vec![PType::U32, PType::U64, PType::I32, PType::I64];
    if max_idx <= usize::from(u16::MAX) {
        take_ptypes.push(PType::U16);
    }
    if max_idx <= usize::from(u8::MAX) {
        take_ptypes.push(PType::U8);
    }
    let take_ptype = take_ptypes[tc.draw(gs::integers::<usize>().max_value(take_ptypes.len() - 1))];
    let take_prim = make_take_indices(take_ptype, &logical_indices, nullable_indices);
    let take_arr = take_prim.clone().into_array();

    let expected_no_nulls = model.take(&logical_indices, nullable_indices, false);
    let expected_with_nulls = model.take(&logical_indices, nullable_indices, true);

    assert_patches_match_model(
        patches
            .take(&take_arr, &mut ctx)
            .vortex_expect("take")
            .as_ref(),
        &expected_no_nulls,
        &mut ctx,
    )
    .vortex_expect("take");
    assert_patches_match_model(
        patches
            .take_with_nulls(&take_arr, &mut ctx)
            .vortex_expect("take_with_nulls")
            .as_ref(),
        &expected_with_nulls,
        &mut ctx,
    )
    .vortex_expect("take_with_nulls");

    if take_len > 0 {
        for (include_nulls, expected) in [(false, &expected_no_nulls), (true, &expected_with_nulls)]
        {
            assert_patches_match_model(
                patches
                    .take_search(take_prim.clone(), include_nulls, &mut ctx)
                    .vortex_expect("take_search")
                    .as_ref(),
                expected,
                &mut ctx,
            )
            .vortex_expect("take_search");
            assert_patches_match_model(
                patches
                    .take_map(take_prim.clone(), include_nulls, &mut ctx)
                    .vortex_expect("take_map")
                    .as_ref(),
                expected,
                &mut ctx,
            )
            .vortex_expect("take_map");
        }
    }

    // 2. PiecewiseSequenceArray indices: scale gaps/lengths with `len` so sorted runs reach across
    // multi-chunk arrays, and include many-unit-piece / long-dense-run cases.
    let num_pieces = if dense_mode {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(28))
    } else {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(16))
    };
    let sorted_runs = tc.draw(gs::integers::<u8>().max_value(2)) != 0;
    let mut starts = Vec::with_capacity(num_pieces);
    let mut lengths = Vec::with_capacity(num_pieces);
    let mut multipliers = Vec::with_capacity(num_pieces);
    let mut cursor = 0usize;
    let stride = (len / num_pieces).max(1);
    let uniform_len = tc.draw(gs::booleans()).then(|| {
        tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(stride.clamp(1, 32)),
        )
    });
    for _ in 0..num_pieces {
        if sorted_runs {
            if cursor >= len {
                break;
            }
            let max_gap = (len - cursor).min(stride);
            let gap = tc.draw(gs::integers::<usize>().max_value(max_gap));
            let start = cursor + gap;
            if start > len {
                break;
            }
            let max_l = (len - start).min(stride.clamp(12, 48));
            let piece_len = uniform_len.map_or_else(
                || tc.draw(gs::integers::<usize>().max_value(max_l)),
                |u| u.min(max_l),
            );
            starts.push(start as u64);
            lengths.push(piece_len as u64);
            multipliers.push(1u64);
            cursor = start + piece_len;
        } else {
            let start = tc.draw(gs::integers::<usize>().max_value(len + 4));
            let piece_len = tc.draw(gs::integers::<usize>().max_value(6));
            let mult = tc.draw(gs::integers::<u64>().max_value(3));
            starts.push(start as u64);
            lengths.push(piece_len as u64);
            multipliers.push(mult);
        }
    }
    if !starts.is_empty() {
        let pw = piecewise(&starts, &lengths, &multipliers);
        let materialized: Vec<Option<usize>> = starts
            .iter()
            .zip(&lengths)
            .zip(&multipliers)
            .flat_map(|((&s, &l), &m)| {
                (0..l).map(move |k| Some(usize::try_from(s + k * m).vortex_expect("usize")))
            })
            .collect();
        let expected = model.take(&materialized, false, false);
        assert_patches_match_model(
            patches
                .take(&pw, &mut ctx)
                .vortex_expect("take piecewise")
                .as_ref(),
            &expected,
            &mut ctx,
        )
        .vortex_expect("take piecewise");
    }
}

// --- Targeted boundary, metadata-validation, narrow-index, and error/panic guards ---

#[rstest]
#[case::unchunked(false)]
#[case::chunked(true)]
fn test_from_metadata_without_offset_within_chunk(#[case] chunked: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let metadata = PatchesMetadata::new(
        3,
        0,
        PType::U32,
        chunked.then_some(3),
        chunked.then_some(PType::U32),
        None,
    );
    let patches = Patches::from_metadata(
        3072,
        &metadata,
        buffer![7u32, 1029, 2051].into_array(),
        buffer![11i32, 22, 33].into_array(),
        chunked.then(|| buffer![0u32, 1, 2].into_array()),
        &mut ctx,
    )?;

    assert_eq!(patches.offset_within_chunk(), chunked.then_some(0));
    assert_eq!(
        patches.search_index(1029, &mut ctx)?,
        SearchResult::Found(1)
    );
    Ok(())
}

#[rstest]
#[case::later_single_chunk(1500..1501)]
#[case::multiple_chunks(900..2050)]
fn test_from_metadata_short_slice(#[case] range: Range<usize>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let indices = PrimitiveArray::from_iter(0u32..3072).into_array();
    let patches = Patches::new(
        3072,
        0,
        indices.clone(),
        indices,
        Some(buffer![0u32, 1024, 2048].into_array()),
        &mut ctx,
    )?;
    let sliced = patches
        .slice(range, &mut ctx)?
        .vortex_expect("slice has patches");
    let metadata = sliced.to_metadata(sliced.array_len(), sliced.dtype())?;
    let read = Patches::from_metadata(
        sliced.array_len(),
        &metadata,
        sliced.indices().clone(),
        sliced.values().clone(),
        sliced.chunk_offsets().clone(),
        &mut ctx,
    )?;

    assert_eq!(read.offset_within_chunk(), sliced.offset_within_chunk());
    assert_eq!(read.search_index(0, &mut ctx)?, SearchResult::Found(0));
    assert_arrays_eq!(read.values(), sliced.values(), &mut ctx);
    Ok(())
}

#[test]
fn test_take_search_large_chunked() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patches = Patches::new(
        1500,
        0,
        PrimitiveArray::from_iter(0..1500u64).into_array(),
        PrimitiveArray::from_iter(0..1500i32).into_array(),
        Some(buffer![0u64, 1024u64].into_array()),
        &mut ctx,
    )?;

    let taken = patches
        .take_search(
            PrimitiveArray::new((0..1500u64).collect::<Buffer<_>>(), Validity::AllValid),
            false,
            &mut ctx,
        )?
        .vortex_expect("taken patches");

    assert_eq!(taken.array_len(), 1500);
    assert_arrays_eq!(
        taken.indices(),
        PrimitiveArray::from_iter(0..1500u64),
        &mut ctx
    );
    assert_arrays_eq!(
        taken.values(),
        PrimitiveArray::from_option_iter((0..1500i32).map(Some)),
        &mut ctx
    );
    Ok(())
}

#[rstest]
fn test_from_metadata_rejects_inconsistent_chunk_offset(
    #[values(0, 2, 3, 501, usize::MAX)] offset_within_chunk: usize,
) {
    let mut ctx = SESSION.create_execution_ctx();
    // The original first chunk had patches at 100 and 500; slicing at 500 removed one.
    let metadata = PatchesMetadata::new(
        2,
        500,
        PType::U32,
        Some(2),
        Some(PType::U32),
        Some(offset_within_chunk),
    );
    let error = Patches::from_metadata(
        1000,
        &metadata,
        buffer![500u32, 1200].into_array(),
        buffer![11i32, 22].into_array(),
        Some(buffer![0u32, 2].into_array()),
        &mut ctx,
    )
    .expect_err("inconsistent patch offset must be rejected");
    assert!(error.to_string().contains("Patch offset within chunk"));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "Patch indices must be strictly increasing")]
fn test_new_rejects_repeated_indices() {
    let mut ctx = SESSION.create_execution_ctx();
    drop(Patches::new(
        5,
        0,
        buffer![1u32, 1].into_array(),
        buffer![10i32, 20].into_array(),
        None,
        &mut ctx,
    ));
}

#[test]
fn test_new_rejects_first_index_below_offset() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for (indices, values) in [
        (
            buffer![2u32, 10].into_array(),
            buffer![1i32, 2].into_array(),
        ),
        (buffer![2u32].into_array(), buffer![1i32].into_array()),
    ] {
        let err = Patches::new(10, 5, indices, values, None, &mut ctx)
            .expect_err("index below offset must fail");
        assert!(
            err.to_string()
                .contains("Patch minimum index 2 is less than offset 5"),
            "unexpected error: {err}"
        );
    }
    Ok(())
}

#[test]
fn test_filter_and_mask_reject_wrong_length() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patches = Patches::new(
        10,
        0,
        buffer![2u64, 5, 8].into_array(),
        buffer![100i32, 200, 300].into_array(),
        None,
        &mut ctx,
    )?;
    let mask = Mask::from_iter([false, false, true, false, false]);
    for err in [
        patches
            .filter(&mask, &mut ctx)
            .expect_err("wrong filter mask length"),
        patches
            .mask(&mask, &mut ctx)
            .expect_err("wrong mask length"),
    ] {
        assert!(
            err.to_string()
                .contains("Filter mask length 5 does not match array length 10"),
            "unexpected error: {err}"
        );
    }
    Ok(())
}

/// u8 patch indices on an array longer than 256: needles or mask indices past `u8::MAX` must not
/// wrap or fail when searching or filtering.
#[rstest]
fn test_narrow_u8_indices_past_type_max(#[values(false, true)] chunked: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let chunk_offsets = chunked.then(|| buffer![0u64, 2].into_array());
    let patches = Patches::new(
        1100,
        0,
        buffer![10u8, 200].into_array(),
        buffer![1i32, 2].into_array(),
        chunk_offsets,
        &mut ctx,
    )?;

    let sliced = patches
        .slice(5..1100, &mut ctx)?
        .vortex_expect("slice keeps patch at 200");
    // 195 -> raw 200 (found); 253 + 5 = 258 overflows u8 and matches no patch.
    let taken = sliced
        .take_search(
            PrimitiveArray::new(buffer![195u64, 253], Validity::NonNullable),
            false,
            &mut ctx,
        )?
        .vortex_expect("patch at 200 is taken");
    assert_arrays_eq!(taken.indices(), PrimitiveArray::from_iter([0u64]), &mut ctx);
    assert_arrays_eq!(taken.values(), PrimitiveArray::from_iter([2i32]), &mut ctx);

    let dense_u8 = Patches::new(
        1000,
        0,
        (0..=u8::MAX).collect::<Buffer<u8>>().into_array(),
        (0..256i32).collect::<Buffer<i32>>().into_array(),
        None,
        &mut ctx,
    )?;
    let mask = Mask::from_indices(1000, [3, 255, 256, 900]);
    let filtered = dense_u8
        .filter(&mask, &mut ctx)?
        .vortex_expect("patches at 3 and 255 kept");
    assert_arrays_eq!(
        filtered.indices(),
        PrimitiveArray::from_iter([0u64, 1]),
        &mut ctx
    );
    assert_arrays_eq!(
        filtered.values(),
        PrimitiveArray::from_iter([3i32, 255]),
        &mut ctx
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum CorruptOffsetLookup {
    SearchIndex,
    GetPatched,
    Take,
    Slice,
}

/// Chunk offsets that can come from a corrupt file but that no writer produces.
#[rstest]
#[case::later_offset_below_the_first(buffer![2u64, 0, 0])]
#[case::empty(Buffer::<u64>::empty())]
#[case::fewer_than_the_chunks(buffer![0u64])]
fn corrupt_chunk_offsets_return_errors(
    #[case] chunk_offsets: Buffer<u64>,
    #[values(
        CorruptOffsetLookup::SearchIndex,
        CorruptOffsetLookup::GetPatched,
        CorruptOffsetLookup::Take,
        CorruptOffsetLookup::Slice
    )]
    lookup: CorruptOffsetLookup,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Ten patches for one taken index, so `take` searches by chunk rather than through a map.
    let patches = Patches::new(
        3000,
        0,
        buffer![5u64, 100, 1500, 1600, 1700, 1800, 1900, 2100, 2500, 2999].into_array(),
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array(),
        Some(chunk_offsets.into_array()),
        &mut ctx,
    )?;
    let result = match lookup {
        CorruptOffsetLookup::SearchIndex => patches.search_index(2100, &mut ctx).map(|_| ()),
        CorruptOffsetLookup::GetPatched => patches.get_patched(2100, &mut ctx).map(|_| ()),
        CorruptOffsetLookup::Take => patches
            .take(&buffer![2100u64].into_array(), &mut ctx)
            .map(|_| ()),
        CorruptOffsetLookup::Slice => patches.slice(2000..3000, &mut ctx).map(|_| ()),
    };
    assert!(
        result.is_err(),
        "{lookup:?} should reject corrupt chunk offsets"
    );
    Ok(())
}
