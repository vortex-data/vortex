// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::block::BlockBuf;
use vortex_array::block::BlockSource;
use vortex_array::block::BlockValidity;
use vortex_array::block::SourceBlock;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Sparse;
use crate::SparseExt;

pub(super) fn sparse_source(
    array: ArrayView<'_, Sparse>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    let DType::Primitive(ptype, _) = array.dtype() else {
        return Ok(None);
    };
    let patches = array.patches();
    let (Some(indices), Some(values)) = (
        patches.indices().as_opt::<Primitive>(),
        patches.values().as_opt::<Primitive>(),
    ) else {
        return Ok(None);
    };
    let fill = array.data().fill_scalar().as_primitive();
    let mask = values.validity()?.execute_mask(values.len(), ctx)?;
    let fill_valid = array.data().fill_scalar().is_valid();
    let validity = if fill_valid {
        Mask::new_true(array.len())
    } else {
        Mask::new_false(array.len())
    };
    // Uniform patch validity matching the fill needs no edits at all.
    let patch_validity = match (&mask, fill_valid) {
        (Mask::AllTrue(_), true) | (Mask::AllFalse(_), false) => None,
        _ => Some(mask),
    };
    Ok(Some(match_each_native_ptype!(*ptype, |T| {
        let values = values.to_buffer::<T>();
        let fill = fill.typed_value::<T>().unwrap_or_default();
        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            Box::new(SparseSource {
                indices: indices.to_buffer::<I>(),
                values,
                fill,
                validity,
                patch_validity,
                offset: patches.offset(),
                next: 0,
            }) as Box<dyn BlockSource>
        })
    })))
}

struct SparseSource<I, T> {
    indices: Buffer<I>,
    values: Buffer<T>,
    fill: T,
    validity: Mask,
    patch_validity: Option<Mask>,
    offset: usize,
    next: usize,
}

impl<I: IntegerPType, T: NativePType> BlockSource for SparseSource<I, T> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let values = scratch.values_mut::<T>(rows.len());
        values.fill(self.fill);
        while let Some(&index) = self.indices.get(self.next) {
            let row = index.as_() - self.offset;
            if row >= rows.end {
                break;
            }
            let lane = row - rows.start;
            values[lane] = self.values[self.next];
            if let Some(mask) = &self.patch_validity {
                validity.set(lane, mask.value(self.next));
            }
            self.next += 1;
        }
        Ok(SourceBlock::Scratch {
            offset: 0,
            len: rows.len(),
        })
    }
}

#[cfg(all(test, not(codspeed)))]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::session::ArraySession;
    use vortex_array::test_harness::fusion::fuse_parent;
    use vortex_array::test_harness::fusion::fusion_trace;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Sparse;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    });

    #[derive(Clone, Copy, Debug)]
    enum PatchValidity {
        NonNullable,
        AllValid,
        AllNull,
        Mixed,
    }

    #[rstest]
    #[case::non_nullable(Some(42i64), PatchValidity::NonNullable)]
    #[case::valid_fill_valid_patches(Some(42i64), PatchValidity::AllValid)]
    #[case::valid_fill_null_patches(Some(42i64), PatchValidity::AllNull)]
    #[case::valid_fill_mixed_patches(Some(42i64), PatchValidity::Mixed)]
    #[case::null_fill_valid_patches(None, PatchValidity::AllValid)]
    #[case::null_fill_null_patches(None, PatchValidity::AllNull)]
    #[case::null_fill_mixed_patches(None, PatchValidity::Mixed)]
    fn sparse_source_fuses_across_blocks_and_slices(
        #[case] fill: Option<i64>,
        #[case] patch_validity: PatchValidity,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 3500u32;
        // Include block-boundary rows alongside strided positions across all four blocks.
        let mut positions: Vec<u32> = (0..len).step_by(37).collect();
        positions.extend([0, 17, 1023, 1024, 1025, 2047, 2048, 3204, 3499]);
        positions.sort_unstable();
        positions.dedup();

        let patch_count = i64::try_from(positions.len())?;
        let (values, fill_scalar) = match patch_validity {
            PatchValidity::NonNullable => (
                PrimitiveArray::from_iter((0..patch_count).map(|i| i * 13 - 500)).into_array(),
                Scalar::primitive(fill.unwrap_or(0), Nullability::NonNullable),
            ),
            PatchValidity::AllValid => (
                PrimitiveArray::from_option_iter((0..patch_count).map(|i| Some(i * 13 - 500)))
                    .into_array(),
                Scalar::from(fill),
            ),
            PatchValidity::AllNull => (
                PrimitiveArray::from_option_iter((0..patch_count).map(|_| None::<i64>))
                    .into_array(),
                Scalar::from(fill),
            ),
            PatchValidity::Mixed => (
                PrimitiveArray::from_option_iter(
                    (0..patch_count).map(|i| (i % 3 != 0).then_some(i * 13 - 500)),
                )
                .into_array(),
                Scalar::from(fill),
            ),
        };

        let sparse = Sparse::try_new(
            PrimitiveArray::from_iter(positions).into_array(),
            values,
            len as usize,
            fill_scalar,
            &mut ctx,
        )?
        .into_array();

        for candidate in [sparse.clone(), sparse.slice(17..3205)?] {
            let trace = fusion_trace(fuse_parent(candidate), &SESSION)?;
            assert!(trace.contains("fuse start array=vortex.sparse"), "{trace}");
        }
        Ok(())
    }
}
