// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Randomized model-based tests.
//!
//! Each case builds a [`Model`] describing which rows carry a patch, constructs the equivalent
//! [`Patched`] array, and asserts that execution, point lookups, slicing and compaction agree
//! with the model. Seeds are fixed so failures reproduce.

use std::ops::Range;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::Patched;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::assert_arrays_eq;
use crate::dtype::Nullability::NonNullable;
use crate::optimizer::ArrayOptimizer;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// The reference semantics: a window of `len` rows starting at `base` over the sequence
/// `row * 3`, with a sorted set of patched rows carrying one value each.
#[derive(Clone, Debug)]
struct Model {
    base: usize,
    len: usize,
    /// Strictly increasing row positions carrying a patch, relative to the window.
    positions: Vec<usize>,
    values: Vec<u64>,
}

impl Model {
    /// Draw a length spanning several chunks and a random subset of patched rows.
    fn generate(rng: &mut StdRng) -> Self {
        let len = rng.random_range(1usize..4096);
        let num_patches = rng.random_range(1usize..=len.min(64));
        let mut positions: Vec<usize> = Vec::with_capacity(num_patches);
        while positions.len() < num_patches {
            let candidate = rng.random_range(0..len);
            if let Err(idx) = positions.binary_search(&candidate) {
                positions.insert(idx, candidate);
            }
        }
        let values = (0..positions.len()).map(|_| rng.random::<u64>()).collect();
        Self {
            base: 0,
            len,
            positions,
            values,
        }
    }

    fn base_value(&self, row: usize) -> u64 {
        ((self.base + row) * 3) as u64
    }

    fn value_at(&self, row: usize) -> u64 {
        match self.positions.binary_search(&row) {
            Ok(ordinal) => self.values[ordinal],
            Err(_) => self.base_value(row),
        }
    }

    fn expected(&self) -> PrimitiveArray {
        PrimitiveArray::from_iter((0..self.len).map(|row| self.value_at(row)))
    }

    /// The model restricted to `range`, with positions rebased onto the slice.
    fn slice(&self, range: Range<usize>) -> Self {
        let (positions, values) = self
            .positions
            .iter()
            .zip(&self.values)
            .filter(|&(&position, _)| range.contains(&position))
            .map(|(&position, &value)| (position - range.start, value))
            .unzip();
        Self {
            base: self.base + range.start,
            len: range.len(),
            positions,
            values,
        }
    }

    /// Build the equivalent array by way of a global-index [`Patches`].
    fn build(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let inner =
            PrimitiveArray::from_iter((0..self.len).map(|row| self.base_value(row))).into_array();
        let indices: Vec<u64> = self.positions.iter().map(|&p| p as u64).collect();
        let patches = Patches::new(
            self.len,
            0,
            PrimitiveArray::new(Buffer::from(indices), Validity::NonNullable).into_array(),
            PrimitiveArray::new(Buffer::from(self.values.clone()), Validity::NonNullable)
                .into_array(),
            None,
        )?;
        Ok(Patched::from_array_and_patches(inner, &patches, ctx)?.into_array())
    }

    /// Assert that execution and point lookups of `array` agree with this model.
    fn assert_matches(&self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        assert_eq!(array.len(), self.len, "length of {self:?}");
        let executed = array.clone().execute::<PrimitiveArray>(ctx)?;
        assert_arrays_eq!(self.expected(), executed, ctx);

        // Point lookups at every patched row and its neighbours, which are the rows where a
        // chunk or offset mistake would show.
        for &position in &self.positions {
            for row in [position.saturating_sub(1), position, position + 1] {
                if row >= self.len {
                    continue;
                }
                assert_eq!(
                    array.execute_scalar(row, ctx)?,
                    Scalar::primitive(self.value_at(row), NonNullable),
                    "scalar_at({row}) on {self:?}"
                );
            }
        }
        Ok(())
    }
}

#[test]
fn executes_like_the_model() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    for seed in 0..48u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let model = Model::generate(&mut rng);
        let array = model.build(&mut ctx)?;
        model.assert_matches(&array, &mut ctx)?;

        // The constructed children pass full validation.
        let patched = array.as_::<Patched>();
        Patched::try_new(
            patched.inner().clone(),
            patched.patch_indices().clone(),
            patched.patch_values().clone(),
            patched.chunk_offsets().clone(),
            patched.offset(),
        )?;
    }
    Ok(())
}

#[test]
fn slices_match_the_model() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    for seed in 0..48u64 {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5115);
        let model = Model::generate(&mut rng);
        let array = model.build(&mut ctx)?;

        for _ in 0..6 {
            let start = rng.random_range(0..model.len);
            let end = rng.random_range(start + 1..=model.len);
            let sliced = array.slice(start..end)?.optimize()?;
            let expected = model.slice(start..end);
            expected.assert_matches(&sliced, &mut ctx)?;

            // Slicing a slice keeps working on the shared children.
            let inner_start = rng.random_range(0..expected.len);
            let inner_end = rng.random_range(inner_start + 1..=expected.len);
            let inner = sliced.slice(inner_start..inner_end)?.optimize()?;
            let expected_inner = expected.slice(inner_start..inner_end);
            expected_inner.assert_matches(&inner, &mut ctx)?;

            // Compaction drops the dead patches without changing the logical array.
            if let Some(patched) = inner.as_opt::<Patched>() {
                let compacted = patched.compact()?;
                expected_inner.assert_matches(&compacted, &mut ctx)?;
                if let Some(compacted) = compacted.as_opt::<Patched>() {
                    let offsets = compacted.chunk_offsets().as_::<Primitive>();
                    let offsets = offsets.as_slice::<u32>();
                    assert_eq!(offsets[0], 0);
                    assert_eq!(
                        offsets[offsets.len() - 1] as usize,
                        compacted.patch_indices().len()
                    );
                    assert_eq!(
                        compacted.patch_indices().len(),
                        expected_inner.positions.len()
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn from_patches_keeps_the_grid_alignment() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    // A v1 patch set sliced to start at global row 1030: logical rows 5 and 1020 are patched.
    let inner = PrimitiveArray::from_iter(0u32..2000).into_array();
    let patches = Patches::new(
        2000,
        1030,
        buffer![1035u32, 2050].into_array(),
        buffer![u32::MAX, u32::MAX - 1].into_array(),
        None,
    )?;
    let patched = Patched::from_array_and_patches(inner, &patches, &mut ctx)?;

    assert_eq!(patched.offset(), 1030 % PATCH_CHUNK_SIZE);
    assert_eq!(patched.n_chunks(), 2);
    assert_eq!(
        patched.patch_indices().as_::<Primitive>().as_slice::<u16>(),
        &[11, 2]
    );
    assert_eq!(
        patched.chunk_offsets().as_::<Primitive>().as_slice::<u32>(),
        &[0, 1, 2]
    );

    let mut expected: Vec<u32> = (0..2000).collect();
    expected[5] = u32::MAX;
    expected[1020] = u32::MAX - 1;
    assert_arrays_eq!(
        PrimitiveArray::from_iter(expected),
        patched.into_array().execute::<PrimitiveArray>(&mut ctx)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn try_new_rejects_invalid_layouts() {
    let inner = || buffer![0u64; 3000].into_array();
    let values = || buffer![1u64, 2].into_array();

    // Unsorted within a chunk.
    assert!(
        Patched::try_new(
            inner(),
            buffer![100u16, 5].into_array(),
            values(),
            buffer![0u32, 2, 2, 2].into_array(),
            0,
        )
        .is_err()
    );
    // Wrong number of chunk offsets.
    assert!(
        Patched::try_new(
            inner(),
            buffer![5u16, 100].into_array(),
            values(),
            buffer![0u32, 2].into_array(),
            0,
        )
        .is_err()
    );
    // Decreasing chunk offsets.
    assert!(
        Patched::try_new(
            inner(),
            buffer![5u16, 100].into_array(),
            values(),
            buffer![0u32, 2, 1, 2].into_array(),
            0,
        )
        .is_err()
    );
    // Offset outside the first chunk.
    assert!(
        Patched::try_new(
            inner(),
            buffer![5u16, 100].into_array(),
            values(),
            buffer![0u32, 2, 2, 2, 2].into_array(),
            PATCH_CHUNK_SIZE,
        )
        .is_err()
    );
    // Chunk offsets pointing past the patches.
    assert!(
        Patched::try_new(
            inner(),
            buffer![5u16, 100].into_array(),
            values(),
            buffer![0u32, 2, 3, 3].into_array(),
            0,
        )
        .is_err()
    );
    // And the well-formed version of the same layout.
    assert!(
        Patched::try_new(
            inner(),
            buffer![5u16, 100].into_array(),
            values(),
            buffer![0u32, 2, 2, 2].into_array(),
            0,
        )
        .is_ok()
    );
}
