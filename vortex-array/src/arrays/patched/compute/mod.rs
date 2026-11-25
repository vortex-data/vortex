// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compare;
mod filter;
pub(crate) mod rules;
mod take;

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
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    pub(super) fn make_patched(
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
    #[case::multi_chunk(make_patched(1500, false, &[3, 512, 1024, 1499], &[500, 600, 700, 800], None))]
    #[case::nullable(make_patched(64, true, &[1, 13, 37, 62], &[500, 600, 700, 800], None))]
    #[case::sliced(make_patched(2048, false, &[5, 1030, 1500], &[500, 600, 700], Some(5..1600)))]
    #[case::nullable_sliced(make_patched(1500, true, &[1, 1031, 1499], &[500, 600, 700], Some(1030..1500)))]
    fn test_patched_consistency(#[case] array: ArrayRef) {
        test_array_consistency(&array, &mut SESSION.create_execution_ctx());
    }
}
