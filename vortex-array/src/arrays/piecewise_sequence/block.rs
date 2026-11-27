// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayView;
use crate::arrays::PiecewiseSequence;
use crate::arrays::Primitive;
use crate::arrays::piecewise_sequence::array::PiecewiseSequenceArraySlotsExt;
use crate::block::BlockBuf;
use crate::block::BlockSource;
use crate::block::BlockValidity;
use crate::block::SourceBlock;
use crate::dtype::NativePType;
use crate::dtype::PType;

/// RunEnd's dictionary conversion emits one constant piece per run.
pub(crate) fn is_repeated(array: ArrayView<'_, PiecewiseSequence>) -> bool {
    [array.starts(), array.lengths()]
        .iter()
        .all(|a| a.dtype().as_ptype() == PType::U64 && a.is::<Primitive>())
        && array
            .multipliers()
            .as_constant()
            .is_some_and(|s| s.as_primitive().as_::<u64>() == Some(0))
}

/// Map each run's code just once, and expand its value directly into the consumer's block.
pub(crate) fn repeated_source<T: NativePType>(
    array: ArrayView<'_, PiecewiseSequence>,
    lookup: impl Fn(u64) -> VortexResult<T> + Send + 'static,
) -> Box<dyn BlockSource> {
    Box::new(Runs {
        starts: array.starts().as_::<Primitive>().to_buffer::<u64>(),
        lengths: array.lengths().as_::<Primitive>().to_buffer::<u64>(),
        piece: 0,
        within_piece: 0,
        validity: Mask::new_true(array.len()),
        lookup,
        output: PhantomData,
    })
}

struct Runs<T, F> {
    starts: Buffer<u64>,
    lengths: Buffer<u64>,
    piece: usize,
    within_piece: u64,
    validity: Mask,
    lookup: F,
    output: PhantomData<T>,
}

impl<T: NativePType, F: Fn(u64) -> VortexResult<T> + Send> BlockSource for Runs<T, F> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'a>(
        &'a mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'a>> {
        let mut output = scratch.values_mut::<T>(rows.len());
        while !output.is_empty() {
            let &length = self.lengths.get(self.piece).ok_or_else(|| {
                vortex_err!("PiecewiseSequenceArray pieces end before its declared length")
            })?;
            let count = usize::try_from((length - self.within_piece).min(output.len() as u64))?;
            if count != 0 {
                let (part, rest) = output.split_at_mut(count);
                part.fill((self.lookup)(self.starts[self.piece])?);
                output = rest;
            }
            self.within_piece += count as u64;
            if self.within_piece == length {
                self.piece += 1;
                self.within_piece = 0;
            }
        }
        if rows.end == self.validity.len() {
            vortex_ensure!(
                self.within_piece == 0 && self.lengths[self.piece..].iter().all(|&len| len == 0),
                "PiecewiseSequenceArray pieces exceed its declared length"
            );
        }
        Ok(SourceBlock::Scratch {
            offset: 0,
            len: rows.len(),
        })
    }
}
