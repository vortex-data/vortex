// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::*;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::patches::Patches;

fn test_patches(ctx: &mut ExecutionCtx) -> VortexResult<PatchesV2> {
    // Patches at global rows 5, 100, 1023, 1024, 2050 in a 3000-row array.
    let indices = PrimitiveArray::new(buffer![5u64, 100, 1023, 1024, 2050], Validity::NonNullable);
    let values = PrimitiveArray::new(buffer![50u64, 51, 52, 53, 54], Validity::NonNullable);
    let global = Patches::new(3000, 0, indices.into_array(), values.into_array(), None)?;
    PatchesV2::from_patches(&global, ctx)
}

#[test]
fn from_global_roundtrip() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = test_patches(&mut ctx)?;
    assert_eq!(patches.num_patches(), 5);
    assert_eq!(patches.offset(), 0);

    let back = patches.to_patches(&mut ctx)?;
    let globals = back.indices().clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(globals.as_slice::<u64>(), &[5, 100, 1023, 1024, 2050]);
    Ok(())
}

#[test]
fn validates_components() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = test_patches(&mut ctx)?;
    PatchesV2::try_new(
        patches.array_len(),
        patches.offset(),
        patches.indices().clone(),
        patches.values().clone(),
        patches.chunk_offsets().clone(),
        &mut ctx,
    )?;

    // Unsorted local indices within one chunk are rejected.
    let unsorted = PatchesV2::try_new(
        3000,
        0,
        PrimitiveArray::new(buffer![100u16, 5], Validity::NonNullable).into_array(),
        PrimitiveArray::new(buffer![1u64, 2], Validity::NonNullable).into_array(),
        PrimitiveArray::new(buffer![0u32, 2, 2, 2], Validity::NonNullable).into_array(),
        &mut ctx,
    );
    assert!(unsorted.is_err());
    Ok(())
}

#[test]
fn search_across_chunks() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = test_patches(&mut ctx)?;
    assert_eq!(
        patches.search_index(1023, &mut ctx)?,
        SearchResult::Found(2)
    );
    assert_eq!(
        patches.search_index(1024, &mut ctx)?,
        SearchResult::Found(3)
    );
    assert_eq!(
        patches.search_index(1500, &mut ctx)?,
        SearchResult::NotFound(4)
    );
    assert_eq!(
        patches.search_index(2999, &mut ctx)?,
        SearchResult::NotFound(5)
    );

    let value = patches.get_patched(2050, &mut ctx)?;
    assert_eq!(value, Some(Scalar::primitive(54u64, NonNullable)));
    assert_eq!(patches.get_patched(2051, &mut ctx)?, None);
    Ok(())
}

#[test]
fn apply_each_visits_all_patches() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = test_patches(&mut ctx)?;
    let mut visited = Vec::new();
    patches.apply_each(&mut ctx, |logical, ordinal| {
        visited.push((logical, ordinal))
    })?;
    assert_eq!(
        visited,
        vec![(5, 0), (100, 1), (1023, 2), (1024, 3), (2050, 4)]
    );

    // A sliced patch set reports logical indices relative to the slice.
    let sliced = patches
        .slice(100..2050, &mut ctx)?
        .expect("patches remain in slice");
    let mut visited = Vec::new();
    sliced.apply_each(&mut ctx, |logical, ordinal| {
        visited.push((logical, ordinal))
    })?;
    assert_eq!(visited, vec![(0, 0), (923, 1), (924, 2)]);
    Ok(())
}

#[test]
fn apply_into_scatters_patch_values() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // Patches at rows 1 and 3 of a 4-row array.
    let indices = PrimitiveArray::new(buffer![1u64, 3], Validity::NonNullable);
    let values = PrimitiveArray::new(buffer![10u64, 20], Validity::NonNullable);
    let global = Patches::new(4, 0, indices.into_array(), values.into_array(), None)?;
    let patches = PatchesV2::from_patches(&global, &mut ctx)?;

    let mut out = [7u64; 4];
    patches.apply_into(&mut out, &mut ctx)?;
    assert_eq!(out, [7, 10, 7, 20]);

    // The scatter is not integer-only: float output works the same way.
    let values = PrimitiveArray::new(buffer![1.5f64, -2.5], Validity::NonNullable);
    let global = Patches::new(
        4,
        0,
        PrimitiveArray::new(buffer![1u64, 3], Validity::NonNullable).into_array(),
        values.into_array(),
        None,
    )?;
    let patches = PatchesV2::from_patches(&global, &mut ctx)?;
    let mut out = [0.0f64; 4];
    patches.apply_into(&mut out, &mut ctx)?;
    assert_eq!(out, [0.0, 1.5, 0.0, -2.5]);

    // A length mismatch is an error rather than a panic.
    let mut wrong = [0.0f64; 3];
    assert!(patches.apply_into(&mut wrong, &mut ctx).is_err());
    Ok(())
}

#[test]
fn view_matches_generic_search() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = test_patches(&mut ctx)?;
    let view = patches.view().expect("canonical children");
    for index in [0, 5, 100, 1023, 1024, 1500, 2050, 2999, 5000] {
        assert_eq!(
            view.search_index(index),
            patches.search_index(index, &mut ctx)?
        );
    }
    Ok(())
}

#[test]
fn slice_unaligned() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = test_patches(&mut ctx)?;

    // Slice 100..2050 keeps rows 100, 1023, 1024 and drops 5 and 2050.
    let sliced = patches
        .slice(100..2050, &mut ctx)?
        .expect("patches remain in slice");
    assert_eq!(sliced.array_len(), 1950);
    assert_eq!(sliced.num_patches(), 3);
    assert_eq!(sliced.offset(), 100);
    assert_eq!(sliced.search_index(0, &mut ctx)?, SearchResult::Found(0));
    assert_eq!(sliced.search_index(923, &mut ctx)?, SearchResult::Found(1));
    assert_eq!(sliced.search_index(924, &mut ctx)?, SearchResult::Found(2));
    assert_eq!(
        sliced.get_patched(924, &mut ctx)?,
        Some(Scalar::primitive(53u64, NonNullable))
    );
    assert_eq!(sliced.get_patched(925, &mut ctx)?, None);

    // Slicing a slice rebases again.
    let inner = sliced
        .slice(900..1000, &mut ctx)?
        .expect("patches remain in inner slice");
    assert_eq!(inner.num_patches(), 2);
    assert_eq!(inner.search_index(23, &mut ctx)?, SearchResult::Found(0));
    assert_eq!(inner.search_index(24, &mut ctx)?, SearchResult::Found(1));

    // A gap with no patches slices to None.
    assert!(patches.slice(1100..2000, &mut ctx)?.is_none());
    Ok(())
}

/// Randomized model-based tests.
///
/// Each case builds a [`Model`] describing which logical rows carry a patch, constructs the
/// equivalent [`PatchesV2`], and asserts every operation agrees with the model. Seeds are fixed
/// so failures reproduce.
mod property {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    /// The reference semantics a [`PatchesV2`] must implement: a sorted set of patched logical
    /// rows, each carrying one value.
    #[derive(Clone, Debug)]
    struct Model {
        array_len: usize,
        /// Strictly increasing logical row positions carrying a patch.
        positions: Vec<usize>,
        values: Vec<u64>,
    }

    impl Model {
        /// Draw an array length spanning several chunks and a random subset of patched rows.
        fn generate(rng: &mut StdRng) -> Self {
            let array_len = rng.random_range(1usize..4096);
            let num_patches = rng.random_range(1usize..=array_len.min(64));
            let mut positions: Vec<usize> = Vec::with_capacity(num_patches);
            while positions.len() < num_patches {
                let candidate = rng.random_range(0..array_len);
                if let Err(idx) = positions.binary_search(&candidate) {
                    positions.insert(idx, candidate);
                }
            }
            let values = (0..positions.len()).map(|_| rng.random::<u64>()).collect();
            Self {
                array_len,
                positions,
                values,
            }
        }

        fn search(&self, index: usize) -> SearchResult {
            if index >= self.array_len {
                return SearchResult::NotFound(self.positions.len());
            }
            match self.positions.binary_search(&index) {
                Ok(ordinal) => SearchResult::Found(ordinal),
                Err(ordinal) => SearchResult::NotFound(ordinal),
            }
        }

        /// The model restricted to `range`, with positions rebased onto the slice.
        fn slice(&self, range: Range<usize>) -> Option<Self> {
            let positions: Vec<usize> = self
                .positions
                .iter()
                .filter(|&&position| range.contains(&position))
                .map(|&position| position - range.start)
                .collect();
            if positions.is_empty() {
                return None;
            }
            let values = self
                .positions
                .iter()
                .zip(&self.values)
                .filter(|&(&position, _)| range.contains(&position))
                .map(|(_, &value)| value)
                .collect();
            Some(Self {
                array_len: range.len(),
                positions,
                values,
            })
        }

        /// Build the equivalent patch set by way of a global-index [`Patches`].
        fn build(&self, ctx: &mut ExecutionCtx) -> VortexResult<PatchesV2> {
            let indices: Vec<u64> = self.positions.iter().map(|&p| p as u64).collect();
            let global = Patches::new(
                self.array_len,
                0,
                PrimitiveArray::new(Buffer::from(indices), Validity::NonNullable).into_array(),
                PrimitiveArray::new(Buffer::from(self.values.clone()), Validity::NonNullable)
                    .into_array(),
                None,
            )?;
            PatchesV2::from_patches(&global, ctx)
        }

        /// Assert every [`PatchesV2`] operation agrees with this model.
        fn assert_matches(&self, patches: &PatchesV2, ctx: &mut ExecutionCtx) -> VortexResult<()> {
            assert_eq!(patches.array_len(), self.array_len);
            assert_eq!(patches.num_patches(), self.positions.len());

            // Constructed components pass validation.
            PatchesV2::try_new(
                patches.array_len(),
                patches.offset(),
                patches.indices().clone(),
                patches.values().clone(),
                patches.chunk_offsets().clone(),
                ctx,
            )?;

            let view = patches.view().expect("canonical children");
            for index in 0..self.array_len {
                let expected = self.search(index);
                assert_eq!(
                    patches.search_index(index, ctx)?,
                    expected,
                    "search_index({index}) on {self:?}"
                );
                assert_eq!(view.search_index(index), expected, "view search({index})");

                let expected_value = expected
                    .to_found()
                    .map(|ordinal| Scalar::primitive(self.values[ordinal], NonNullable));
                assert_eq!(
                    patches.get_patched(index, ctx)?,
                    expected_value,
                    "get_patched({index})"
                );
            }
            // Out-of-bounds lookups report the end of the patch list rather than panicking.
            assert_eq!(
                patches.search_index(self.array_len, ctx)?,
                SearchResult::NotFound(self.positions.len())
            );

            let mut visited = Vec::new();
            patches.apply_each(ctx, |logical, ordinal| visited.push((logical, ordinal)))?;
            let expected: Vec<(usize, usize)> = self
                .positions
                .iter()
                .enumerate()
                .map(|(ordinal, &position)| (position, ordinal))
                .collect();
            assert_eq!(visited, expected, "apply_each on {self:?}");

            // Replace-mode scatter writes exactly the patched rows.
            let mut out = vec![0u64; self.array_len];
            patches.apply_into(&mut out, ctx)?;
            let mut expected_out = vec![0u64; self.array_len];
            for (&position, &value) in self.positions.iter().zip(&self.values) {
                expected_out[position] = value;
            }
            assert_eq!(out, expected_out, "apply_into on {self:?}");
            Ok(())
        }
    }

    #[test]
    fn model_matches_across_seeds() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let model = Model::generate(&mut rng);
            let patches = model.build(&mut ctx)?;
            model.assert_matches(&patches, &mut ctx)?;
        }
        Ok(())
    }

    #[test]
    fn slices_match_the_model() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5115);
            let model = Model::generate(&mut rng);
            let patches = model.build(&mut ctx)?;

            for _ in 0..8 {
                let start = rng.random_range(0..model.array_len);
                let end = rng.random_range(start..=model.array_len);
                let sliced = patches.slice(start..end, &mut ctx)?;
                match model.slice(start..end) {
                    // An empty range, or one holding no patches, slices away entirely.
                    None => assert!(
                        sliced.is_none(),
                        "expected no patches in {start}..{end} of {model:?}"
                    ),
                    Some(expected) => {
                        let sliced =
                            sliced.unwrap_or_else(|| panic!("patches remain in {start}..{end}"));
                        expected.assert_matches(&sliced, &mut ctx)?;

                        // Slicing a slice rebases onto the inner range.
                        let inner_start = rng.random_range(0..expected.array_len);
                        let inner_end = rng.random_range(inner_start..=expected.array_len);
                        let inner = sliced.slice(inner_start..inner_end, &mut ctx)?;
                        match expected.slice(inner_start..inner_end) {
                            None => assert!(inner.is_none()),
                            Some(expected_inner) => {
                                let inner = inner.expect("patches remain in inner slice");
                                expected_inner.assert_matches(&inner, &mut ctx)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn global_index_roundtrip() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed ^ 0x9001);
            let model = Model::generate(&mut rng);
            let patches = model.build(&mut ctx)?;

            let back = patches.to_patches(&mut ctx)?;
            let globals = back.indices().clone().execute::<PrimitiveArray>(&mut ctx)?;
            let expected: Vec<u64> = model.positions.iter().map(|&p| p as u64).collect();
            assert_eq!(globals.as_slice::<u64>(), expected.as_slice());
            assert_eq!(back.array_len(), model.array_len);

            // Round-tripping back through the chunk-local form is a fixed point.
            let again = PatchesV2::from_patches(&back, &mut ctx)?;
            model.assert_matches(&again, &mut ctx)?;
        }
        Ok(())
    }
}

/// The container addresses patches by position, so it carries values of any dtype — including
/// extension types, which [`Sparse`] itself cannot yet canonicalize.
///
/// [`Sparse`]: https://docs.rs/vortex-sparse
#[test]
fn carries_extension_typed_values() -> VortexResult<()> {
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::dtype::extension::ExtDType;
    use crate::extension::uuid::Uuid;
    use crate::extension::uuid::UuidMetadata;

    let mut ctx = array_session().create_execution_ctx();
    let storage = FixedSizeListArray::try_new(
        PrimitiveArray::new(Buffer::from(vec![7u8; 32]), Validity::NonNullable).into_array(),
        16,
        Validity::NonNullable,
        2,
    )?
    .into_array();
    let ext_dtype =
        ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage.dtype().clone())?.erased();
    let values = ExtensionArray::new(ext_dtype, storage).into_array();
    let expected_dtype = values.dtype().clone();

    let global = Patches::new(
        2048,
        0,
        PrimitiveArray::new(buffer![10u64, 1500], Validity::NonNullable).into_array(),
        values,
        None,
    )?;
    let patches = PatchesV2::from_patches(&global, &mut ctx)?;

    assert_eq!(patches.dtype(), &expected_dtype);
    assert_eq!(patches.num_patches(), 2);
    assert_eq!(patches.search_index(10, &mut ctx)?, SearchResult::Found(0));
    assert_eq!(
        patches.search_index(1500, &mut ctx)?,
        SearchResult::Found(1)
    );
    assert!(patches.get_patched(10, &mut ctx)?.is_some());
    assert_eq!(patches.get_patched(11, &mut ctx)?, None);

    // Slicing keeps the value dtype intact.
    let sliced = patches.slice(1000..2000, &mut ctx)?.expect("patch remains");
    assert_eq!(sliced.dtype(), &expected_dtype);
    assert_eq!(sliced.num_patches(), 1);
    assert_eq!(sliced.search_index(500, &mut ctx)?, SearchResult::Found(0));

    // And the round trip back to global indices is dtype-agnostic too.
    assert_eq!(patches.to_patches(&mut ctx)?.dtype(), &expected_dtype);
    Ok(())
}
