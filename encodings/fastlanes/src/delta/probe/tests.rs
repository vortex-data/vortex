// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::Delta;
use crate::DeltaArraySlotsExt;

/// A one-off or retained probe over `array`, the retained one living in `retained`.
fn probe_for<'a>(
    array: &'a ArrayRef,
    retained: &'a mut Option<RepeatedArrayProbe>,
    repeated: bool,
) -> ArrayProbe<'a> {
    if repeated {
        retained.insert(array.repeated_probe()).as_probe()
    } else {
        array.probe()
    }
}

#[derive(Clone, Copy, Debug)]
enum Deltas {
    Primitive,
    BitPacked,
    /// Bit-packed at a width that leaves some deltas as patches.
    Patched,
}

#[rstest]
fn random_access_across_chunks_and_nulls(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] sliced: bool,
    #[values(Deltas::Primitive, Deltas::BitPacked, Deltas::Patched)] deltas: Deltas,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    // Jumps make deltas too wide for 2 bits, so the patched case packs them as patches.
    let jumps = matches!(deltas, Deltas::Patched);
    let input = PrimitiveArray::from_option_iter(
        (0..5000u32)
            .map(|i| (i % 11 != 0).then_some(if jumps && i % 997 == 5 { i * 7 } else { i / 3 })),
    );
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
                input.len(),
            )?
        }
    };
    let range = if sliced { 1777..4333 } else { 0..5000 };
    let source = if sliced {
        delta
            .into_array()
            .slice(range.clone())?
            .optimize_ctx(&ctx)?
    } else {
        delta.into_array()
    };
    assert!(source.is::<Delta>());
    let input = input.slice(range)?;
    // Revisits earlier chunks, so a retained probe must replace its cached chunk.
    let indices = [0u32, 1, 1023, 1024, 2048, 2047, 11, 33, 2500, 1500, 0];
    let mut actual = builder_with_capacity_in(source.dtype(), indices.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in indices {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(indices).into_array())?,
        &mut ctx
    );
    assert!(probe.execute_scalar(source.len(), &mut ctx).is_err());
    Ok(())
}
