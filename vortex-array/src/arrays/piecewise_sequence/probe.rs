// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::ProbeState;
use crate::arrays::PiecewiseSequence;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::piecewise_sequence::array::PiecewiseSequenceSlots;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::PValue;
use crate::scalar::Scalar;

const MAX_CHECKPOINTS: usize = 4096;
const LENGTH_BLOCK_SIZE: usize = 256;

/// Bounded piece lookup state kept by one repeated probe.
#[derive(Default)]
pub struct PiecewiseProbeState {
    index: Option<PieceIndex>,
}

struct PieceIndex {
    checkpoints: Vec<Checkpoint>,
    last_piece: Option<LocatedPiece>,
}

#[derive(Clone, Copy)]
struct Checkpoint {
    next_piece: usize,
    end: usize,
}

#[derive(Clone, Copy)]
struct LocatedPiece {
    piece: usize,
    start: usize,
    end: usize,
}

pub(super) fn scalar_at(
    state: &mut ProbeState<'_, PiecewiseSequence>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let output_len = state.array().len();
    if !state.is_valid(index, ctx)? {
        return Ok(Scalar::null(state.array().dtype().clone()));
    }

    let (mut retained, mut children) = state.split();
    let lengths = children
        .slot(PiecewiseSequenceSlots::LENGTHS)?
        .ok_or_else(|| vortex_err!("PiecewiseSequence lengths slot is missing"))?
        .array()
        .clone();

    let located = if let Some(constant) = lengths.as_constant() {
        let length = unsigned_scalar_usize(&constant)?;
        if length == 0 || index / length >= lengths.len() {
            return Err(out_of_bounds(index));
        }
        let piece = index / length;
        let start = piece * length;
        LocatedPiece {
            piece,
            start,
            end: start.saturating_add(length).min(output_len),
        }
    } else if let Some(retained) = retained.as_mut() {
        if retained.index.is_none() {
            retained.index = Some(prepare_index(&lengths, output_len, ctx)?);
        }
        retained
            .index
            .as_mut()
            .expect("index was just prepared")
            .locate(&lengths, index, output_len, ctx)?
    } else {
        locate_once(&lengths, index, output_len, ctx)?
    };

    let start = children
        .slot(PiecewiseSequenceSlots::STARTS)?
        .ok_or_else(|| vortex_err!("PiecewiseSequence starts slot is missing"))?
        .execute_scalar(located.piece, ctx)?;
    let multiplier = children
        .slot(PiecewiseSequenceSlots::MULTIPLIERS)?
        .ok_or_else(|| vortex_err!("PiecewiseSequence multipliers slot is missing"))?
        .execute_scalar(located.piece, ctx)?;
    let start = unsigned_scalar_usize(&start)?;
    let multiplier = unsigned_scalar_usize(&multiplier)?;
    let offset = (index - located.start)
        .checked_mul(multiplier)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
    let value = start
        .checked_add(offset)
        .ok_or_else(|| vortex_err!("PiecewiseSequenceArray range overflows usize"))?;
    Ok(Scalar::from(value as u64))
}

fn locate_once(
    lengths: &ArrayRef,
    index: usize,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<LocatedPiece> {
    if let Some(primitive) = lengths.as_opt::<Primitive>() {
        return match_each_unsigned_integer_ptype!(primitive.ptype(), |T| {
            let values = primitive.as_slice::<T>();
            let mut remaining = index;
            for (piece, &length) in values.iter().enumerate() {
                let length: usize = length.as_();
                if remaining < length {
                    let start = index - remaining;
                    let end = start + length.min(output_len - start);
                    return Ok(LocatedPiece { piece, start, end });
                }
                remaining -= length;
            }
            Err(out_of_bounds(index))
        });
    }
    locate_in_range(lengths, 0..lengths.len(), 0, index, output_len, ctx)
}

impl PieceIndex {
    fn locate(
        &mut self,
        lengths: &ArrayRef,
        index: usize,
        output_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<LocatedPiece> {
        if let Some(last) = self.last_piece
            && last.start <= index
            && index < last.end
        {
            return Ok(last);
        }

        let checkpoint = self.checkpoints.partition_point(|entry| entry.end <= index);
        let stop = self
            .checkpoints
            .get(checkpoint)
            .ok_or_else(|| out_of_bounds(index))?;
        let (first_piece, previous_end) = if checkpoint == 0 {
            (0, 0)
        } else {
            let previous = &self.checkpoints[checkpoint - 1];
            (previous.next_piece, previous.end)
        };
        let located = locate_in_range(
            lengths,
            first_piece..stop.next_piece,
            previous_end,
            index,
            output_len,
            ctx,
        )?;
        self.last_piece = Some(located);
        Ok(located)
    }
}

fn prepare_index(
    lengths: &ArrayRef,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PieceIndex> {
    let stride = lengths.len().div_ceil(MAX_CHECKPOINTS).max(1);
    let mut checkpoints = Vec::with_capacity(lengths.len().div_ceil(stride));
    let mut end = 0usize;
    visit_lengths(lengths, 0..lengths.len(), ctx, &mut |piece, length| {
        end += length.min(output_len - end);
        let next_piece = piece + 1;
        if next_piece % stride == 0 || next_piece == lengths.len() || end == output_len {
            checkpoints.push(Checkpoint { next_piece, end });
        }
        Ok(end != output_len)
    })?;
    Ok(PieceIndex {
        checkpoints,
        last_piece: None,
    })
}

fn locate_in_range(
    lengths: &ArrayRef,
    pieces: Range<usize>,
    previous_end: usize,
    index: usize,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<LocatedPiece> {
    let mut start = previous_end;
    let mut located = None;
    visit_lengths(lengths, pieces, ctx, &mut |piece, length| {
        let end = start + length.min(output_len - start);
        if index < end {
            located = Some(LocatedPiece { piece, start, end });
            return Ok(false);
        }
        start = end;
        Ok(true)
    })?;
    located.ok_or_else(|| out_of_bounds(index))
}

fn visit_lengths(
    lengths: &ArrayRef,
    pieces: Range<usize>,
    ctx: &mut ExecutionCtx,
    visit: &mut impl FnMut(usize, usize) -> VortexResult<bool>,
) -> VortexResult<()> {
    if let Some(primitive) = lengths.as_opt::<Primitive>() {
        visit_primitive(primitive, pieces, 0, visit)?;
        return Ok(());
    }

    let mut begin = pieces.start;
    while begin < pieces.end {
        let end = begin.saturating_add(LENGTH_BLOCK_SIZE).min(pieces.end);
        let block = lengths.slice(begin..end)?.execute::<PrimitiveArray>(ctx)?;
        if !visit_primitive(block.as_view(), 0..block.len(), begin, visit)? {
            break;
        }
        begin = end;
    }
    Ok(())
}

fn visit_primitive(
    primitive: ArrayView<'_, Primitive>,
    pieces: Range<usize>,
    base: usize,
    visit: &mut impl FnMut(usize, usize) -> VortexResult<bool>,
) -> VortexResult<bool> {
    match_each_unsigned_integer_ptype!(primitive.ptype(), |T| {
        let values = primitive.as_slice::<T>();
        for piece in pieces {
            if !visit(base + piece, values[piece].as_())? {
                return Ok(false);
            }
        }
        Ok(true)
    })
}

fn unsigned_scalar_usize(scalar: &Scalar) -> VortexResult<usize> {
    match scalar.as_primitive_opt().and_then(|value| value.pvalue()) {
        Some(PValue::U8(value)) => Ok(value as usize),
        Some(PValue::U16(value)) => Ok(value as usize),
        Some(PValue::U32(value)) => Ok(value as usize),
        Some(PValue::U64(value)) => Ok(value.as_()),
        _ => Err(vortex_err!("PiecewiseSequence child value is not an unsigned integer")),
    }
}

fn out_of_bounds(index: usize) -> vortex_error::VortexError {
    vortex_err!("PiecewiseSequenceArray index {index} out of bounds")
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::MAX_CHECKPOINTS;
    use super::prepare_index;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn checkpoint_count_stays_bounded_with_many_pieces() -> VortexResult<()> {
        let count = MAX_CHECKPOINTS * 3 + 1;
        let lengths = PrimitiveArray::from_iter(std::iter::repeat_n(1u8, count)).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let mut index = prepare_index(&lengths, count, &mut ctx)?;
        assert!(index.checkpoints.len() <= MAX_CHECKPOINTS);

        for row in [count - 1, 0, count / 2, count - 2, 1, count - 1] {
            let piece = index.locate(&lengths, row, count, &mut ctx)?;
            assert_eq!(piece.piece, row);
            assert_eq!(piece.start, row);
            assert_eq!(piece.end, row + 1);
        }
        Ok(())
    }
}
