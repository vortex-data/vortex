// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::Zstd;
use super::ZstdArraySlotsExt;

/// What a repeated probe keeps between reads of a [`Zstd`] array.
///
/// A one-off read executes the validity of the unsliced array to find the stored value of its row,
/// and then decompresses the frame that holds it. A repeated probe finds the frame boundaries
/// once, and keeps the last frame that it decompressed. Reads that stay in one frame thus
/// decompress it only once.
#[derive(Default)]
pub struct ZstdProbeState {
    layout: Option<FrameLayout>,
    frame: Option<DecodedFrame>,
}

/// Where the stored values of a [`Zstd`] array are.
struct FrameLayout {
    /// The validity of the unsliced array. Nulls are not stored, so the number of valid rows
    /// before a row is the index of its stored value.
    validity: Mask,
    /// The first stored value of each frame, followed by the number of stored values.
    value_starts: Vec<usize>,
}

impl FrameLayout {
    fn new(array: ArrayView<'_, Zstd>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let data = array.data();
        let validity = child_to_validity(array.validity_child(), array.dtype().nullability())
            .execute_mask(data.unsliced_n_rows(), ctx)?;
        let value_starts = data.frame_value_starts(array.dtype(), validity.true_count())?;

        Ok(Self {
            validity,
            value_starts,
        })
    }

    /// The frame that holds stored value `value`, and the first stored value of that frame.
    fn frame_of(&self, value: usize) -> VortexResult<(usize, usize)> {
        let frame_index = self
            .value_starts
            .partition_point(|&start| start <= value)
            .checked_sub(1)
            .filter(|&frame_index| frame_index + 1 < self.value_starts.len())
            .ok_or_else(|| {
                vortex_err!("Corrupt zstd metadata: no frame holds stored value {value}")
            })?;

        Ok((frame_index, self.value_starts[frame_index]))
    }
}

/// The stored values of one frame.
struct DecodedFrame {
    index: usize,
    values: ArrayRef,
}

impl OperationsVTable<Zstd> for Zstd {
    type ProbeState = ZstdProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Zstd>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let data = array.data();
        let Some(retained) = state.retained() else {
            let unsliced_validity =
                child_to_validity(array.validity_child(), array.dtype().nullability());
            return data
                .with_slice(index, index + 1)
                .decompress(array.dtype(), &unsliced_validity, ctx)?
                .execute_scalar(0, ctx);
        };

        let layout = match &mut retained.layout {
            Some(layout) => layout,
            slot @ None => slot.insert(FrameLayout::new(array, ctx)?),
        };
        let row = data.slice_start() + index;
        let value = layout.validity.valid_counts_for_indices(&[row])[0];
        let (frame_index, frame_start) = layout.frame_of(value)?;

        let frame = match &mut retained.frame {
            Some(frame) if frame.index == frame_index => frame,
            slot => {
                let n_values = layout.value_starts[frame_index + 1] - frame_start;
                slot.insert(DecodedFrame {
                    index: frame_index,
                    values: data.decode_frame(frame_index, n_values, array.dtype(), ctx)?,
                })
            }
        };
        frame.values.execute_scalar(value - frame_start, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Zstd>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Zstd;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const LEN: usize = 200;
    const VALUES_PER_FRAME: usize = 16;

    fn primitive() -> PrimitiveArray {
        PrimitiveArray::from_iter((0..LEN).map(|i| i as i32 * 7))
    }

    fn nullable_primitive() -> PrimitiveArray {
        PrimitiveArray::from_option_iter((0..LEN).map(|i| (i % 3 != 0).then_some(i as i64)))
    }

    fn nullable_utf8() -> VarBinViewArray {
        VarBinViewArray::from_iter_nullable_str((0..LEN).map(|i| {
            (i % 5 != 0).then(|| format!("value {i} with a suffix long enough to not be inlined"))
        }))
    }

    fn compressed_primitive(input: PrimitiveArray) -> VortexResult<(ArrayRef, ArrayRef)> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = Zstd::from_primitive(&input, 3, VALUES_PER_FRAME, &mut ctx)?;
        Ok((compressed.into_array(), input.into_array()))
    }

    fn compressed_utf8(
        input: VarBinViewArray,
        with_dict: bool,
    ) -> VortexResult<(ArrayRef, ArrayRef)> {
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = if with_dict {
            Zstd::from_var_bin_view(&input, 3, VALUES_PER_FRAME, &mut ctx)?
        } else {
            Zstd::from_var_bin_view_without_dict(&input, 3, VALUES_PER_FRAME, &mut ctx)?
        };
        Ok((compressed.into_array(), input.into_array()))
    }

    /// Frames of 16 values, read across frames, back to earlier frames, and from a slice that
    /// starts and ends in the middle of a frame.
    #[rstest]
    #[case::primitive(compressed_primitive(primitive()))]
    #[case::nullable_primitive(compressed_primitive(nullable_primitive()))]
    #[case::nullable_utf8(compressed_utf8(nullable_utf8(), true))]
    #[case::nullable_utf8_without_dict(compressed_utf8(nullable_utf8(), false))]
    fn repeated_probe_matches_input(
        #[case] arrays: VortexResult<(ArrayRef, ArrayRef)>,
    ) -> VortexResult<()> {
        let (compressed, input) = arrays?;
        let mut ctx = SESSION.create_execution_ctx();

        for (array, expected) in [
            (compressed.clone(), input.clone()),
            (
                compressed.slice(5..LEN - 9)?.optimize_ctx(&SESSION)?,
                input.slice(5..LEN - 9)?,
            ),
        ] {
            // The optimized slice is Zstd, so the probe reads the frames with a slice offset.
            assert!(array.is::<Zstd>());

            let mut probe = array.repeated_probe();
            let order = (0..array.len())
                .rev()
                .chain(0..array.len())
                .chain((0..array.len()).step_by(37));
            for index in order {
                assert_eq!(
                    probe.execute_scalar(index, &mut ctx)?,
                    expected.execute_scalar(index, &mut ctx)?,
                    "row {index}"
                );
            }

            test_array_consistency(&array, &mut ctx);
        }

        Ok(())
    }
}
