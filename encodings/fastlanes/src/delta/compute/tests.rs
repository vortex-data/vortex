// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The filter and take kernels against the decoded array.

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::Delta;
use crate::DeltaArray;
use crate::DeltaArraySlotsExt;
use crate::test::SESSION;

#[derive(Clone, Copy, Debug)]
enum Deltas {
    Primitive,
    BitPacked,
    /// Bit-packed at a width that leaves some deltas as patches.
    Patched,
}

const LEN: usize = 5000;

/// A nullable Delta array of `LEN` values over `deltas`, and its source values.
fn fixture(deltas: Deltas, signed: bool) -> VortexResult<(DeltaArray, PrimitiveArray)> {
    let mut ctx = SESSION.create_execution_ctx();
    let jumps = matches!(deltas, Deltas::Patched);
    // Values never decrease, so deltas are non-negative and bit-pack for either signedness.
    // The patched case steps up every 997 rows, by more than 2 bits hold.
    let value = |i: usize| {
        let i = i32::try_from(i).vortex_expect("small index");
        let step = if jumps { (i / 997) * 5000 } else { 0 };
        i / 3 + step - if signed { 1000 } else { 0 }
    };
    let input = if signed {
        PrimitiveArray::from_option_iter((0..LEN).map(|i| (i % 11 != 5).then(|| value(i))))
    } else {
        PrimitiveArray::from_option_iter(
            (0..LEN).map(|i| (i % 11 != 5).then(|| value(i).unsigned_abs())),
        )
    };
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let delta = match deltas {
        Deltas::Primitive => delta,
        Deltas::BitPacked | Deltas::Patched => {
            let packed = BitPacked::encode(delta.deltas(), 2, &mut ctx)?;
            assert_eq!(packed.patches().is_some(), jumps);
            Delta::try_new(
                delta.bases().clone(),
                packed.into_array(),
                delta.validity()?,
                0,
                LEN,
            )?
        }
    };
    Ok((delta, input))
}

/// `array` sliced to `range` and reduced back to a Delta array, or `array` itself.
fn sliced(array: DeltaArray, range: Option<std::ops::Range<usize>>) -> VortexResult<ArrayRef> {
    Ok(match range {
        Some(range) => array
            .into_array()
            .slice(range)?
            .optimize_ctx(&SESSION.create_execution_ctx())?,
        None => array.into_array(),
    })
}

#[rstest]
#[case::sparse(vec![0, 1, 1023, 1024, 3500, 3999])]
#[case::one_chunk((1024..2048).collect())]
#[case::every_third((0..3000).step_by(3).collect())]
fn filter_matches_decoded(
    #[case] selected: Vec<usize>,
    #[values(Deltas::Primitive, Deltas::BitPacked, Deltas::Patched)] deltas: Deltas,
    #[values(false, true)] signed: bool,
    #[values(None, Some(777..4777))] range: Option<std::ops::Range<usize>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (delta, input) = fixture(deltas, signed)?;
    let array = sliced(delta, range.clone())?;
    let expected_input = match range {
        Some(range) => input.into_array().slice(range)?,
        None => input.into_array(),
    };
    let mask = Mask::from_indices(array.len(), selected);
    let filtered = <Delta as FilterKernel>::filter(array.as_::<Delta>(), &mask, &mut ctx)?
        .vortex_expect("Delta filters a values mask");
    assert_arrays_eq!(filtered, expected_input.filter(mask)?, &mut ctx);
    Ok(())
}

#[rstest]
fn take_matches_decoded(
    #[values(Deltas::Primitive, Deltas::BitPacked, Deltas::Patched)] deltas: Deltas,
    #[values(false, true)] signed: bool,
    #[values(None, Some(777..4777))] range: Option<std::ops::Range<usize>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (delta, input) = fixture(deltas, signed)?;
    let array = sliced(delta, range.clone())?;
    let expected_input = match range {
        Some(range) => input.into_array().slice(range)?,
        None => input.into_array(),
    };
    // Out of order, revisiting chunks, with a repeated index and null indices. Few enough
    // indices that the kernel groups them by chunk instead of decoding everything.
    let indices = PrimitiveArray::from_option_iter([
        Some(3999u32),
        Some(0),
        None,
        Some(1024),
        Some(1023),
        Some(3999),
        Some(2047),
        None,
        Some(11),
    ])
    .into_array();
    let taken = <Delta as TakeExecute>::take(array.as_::<Delta>(), &indices, &mut ctx)?
        .vortex_expect("Delta takes");
    assert_arrays_eq!(taken, expected_input.take(indices)?, &mut ctx);
    Ok(())
}

/// Repeated indices can select more values from one chunk than the chunk holds. 9000 rows keep
/// 1100 indices under the kernel's full-decode ratio, so it groups them by chunk.
#[test]
fn take_more_positions_than_a_chunk_from_one_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter((0..9000u32).map(|i| i / 3));
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let indices = PrimitiveArray::from_iter((0..1100u32).map(|i| 1024 + i % 1024)).into_array();
    let taken = <Delta as TakeExecute>::take(delta.as_view(), &indices, &mut ctx)?
        .vortex_expect("Delta takes");
    assert_arrays_eq!(taken, input.into_array().take(indices)?, &mut ctx);
    Ok(())
}

#[test]
fn take_out_of_bounds_errors() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (delta, _) = fixture(Deltas::Primitive, false)?;
    let indices = PrimitiveArray::from_iter([1u32, 5000]).into_array();
    assert!(<Delta as TakeExecute>::take(delta.as_view(), &indices, &mut ctx).is_err());
    Ok(())
}
