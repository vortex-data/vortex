// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compare;
mod filter;
mod take;
mod validity;
pub(super) use validity::PatchValidity;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::patches::Patches;

pub(super) fn select_rows(
    array: ArrayView<'_, Patched>,
    base: ArrayRef,
    patches: Option<Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    Ok(Some(match patches {
        Some(patches) => {
            let indexed = patches.with_chunk_offsets(ctx)?;
            Patched::from_indexed_patches(base, &indexed, array.strategy())?.into_array()
        }
        None => base,
    }))
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn make_patched(
        len: usize,
        nullable: bool,
        patch_indices: &[u32],
        patch_values: &[u16],
        slice: Option<Range<usize>>,
    ) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        let inner = if nullable {
            PrimitiveArray::from_option_iter(
                (0..len).map(|i| (i % 5 != 0).then_some(u16::try_from(i % 251).unwrap())),
            )
            .into_array()
        } else {
            PrimitiveArray::from_iter((0..len).map(|i| u16::try_from(i % 251).unwrap()))
                .into_array()
        };
        let patches = Patches::new(
            len,
            0,
            PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array(),
            PrimitiveArray::from_iter(patch_values.iter().copied()).into_array(),
            None,
            &mut ctx,
        )
        .unwrap();
        let arr = Patched::from_array_and_patches(inner, &patches, &mut ctx)
            .unwrap()
            .into_array();
        if let Some(range) = slice {
            arr.slice(range).unwrap().optimize_ctx(&SESSION).unwrap()
        } else {
            arr
        }
    }

    #[rstest]
    #[case::small(make_patched(10, false, &[1, 4, 8], &[500, 600, 700], None))]
    #[case::single(make_patched(1, false, &[0], &[42], None))]
    #[case::multi_chunk(make_patched(3072, false, &[10, 1024, 2048, 3071], &[900, 901, 902, 903], None))]
    #[case::nullable(make_patched(64, true, &[1, 13, 37, 62], &[500, 600, 700, 800], None))]
    #[case::sliced(make_patched(3072, false, &[5, 1030, 2050], &[900, 901, 902], Some(5..2500)))]
    #[case::nullable_sliced(make_patched(1500, true, &[1, 1031, 1499], &[500, 600, 700], Some(1030..1500)))]
    fn test_patched_conformance(#[case] array: ArrayRef) {
        let mut ctx = SESSION.create_execution_ctx();
        test_array_consistency(&array, &mut ctx);
        test_filter_conformance(&array, &mut ctx);
        test_take_conformance(&array, &mut ctx);
    }
}
