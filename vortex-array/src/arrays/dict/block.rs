// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ArrayView;
use crate::arrays::Dict;
use crate::arrays::PiecewiseSequence;
use crate::arrays::Primitive;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::piecewise_sequence::block::is_repeated;
use crate::arrays::piecewise_sequence::block::repeated_source;
use crate::block::BlockBuf;
use crate::block::BlockOut;
use crate::block::BlockSource;
use crate::block::BlockValidity;
use crate::block::BlockValue;
use crate::block::SourceBlock;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;

pub(super) fn can_source(codes: &ArrayRef, values: &ArrayRef) -> bool {
    !values.dtype().is_nullable()
        && !codes.dtype().is_nullable()
        && codes.dtype().is_unsigned_int()
        && values.is::<Primitive>()
        && (codes.is::<Primitive>() || codes.as_opt::<PiecewiseSequence>().is_some_and(is_repeated))
}

pub(super) fn source(array: ArrayView<'_, Dict>) -> Option<Box<dyn BlockSource>> {
    if !can_source(array.codes(), array.values()) {
        return None;
    }
    let values = array.values().as_::<Primitive>();
    if let Some(runs) = array.codes().as_opt::<PiecewiseSequence>() {
        return Some(match_each_native_ptype!(values.ptype(), |T| {
            let table = values.to_buffer::<T>();
            repeated_source(runs, move |code| {
                table.get(usize::try_from(code)?).copied().ok_or_else(|| {
                    vortex_err!(
                        "dictionary code {code} out of bounds for {} values",
                        table.len()
                    )
                })
            })
        }));
    }
    let codes = array.codes().as_::<Primitive>();
    Some(match_each_unsigned_integer_ptype!(codes.ptype(), |I| {
        match_each_native_ptype!(values.ptype(), |T| {
            Box::new(Gather {
                codes: codes.to_buffer::<I>(),
                table: values.to_buffer::<T>(),
                validity: Mask::new_true(array.len()),
            }) as Box<dyn BlockSource>
        })
    }))
}

struct Gather<I, T> {
    codes: Buffer<I>,
    table: Buffer<T>,
    validity: Mask,
}

impl<I: UnsignedPType, T: NativePType> BlockSource for Gather<I, T> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let len = rows.len();
        gather_checked(
            &self.codes[rows],
            self.table.as_slice(),
            scratch.values_mut::<T>(len),
        )?;
        Ok(SourceBlock::Scratch { offset: 0, len })
    }
}

pub(crate) trait GatherSink<T: BlockValue> {
    fn write_checked<I: UnsignedPType>(self, codes: &[I], f: impl Fn(I) -> T);
}

impl<T: BlockValue> GatherSink<T> for &mut [T] {
    #[inline]
    fn write_checked<I: UnsignedPType>(self, codes: &[I], f: impl Fn(I) -> T) {
        for (out, &code) in self.iter_mut().zip(codes) {
            *out = f(code);
        }
    }
}

impl<T: BlockValue> GatherSink<T> for &mut BlockOut<'_> {
    #[inline]
    fn write_checked<I: UnsignedPType>(self, codes: &[I], f: impl Fn(I) -> T) {
        self.push_map(codes, f);
    }
}

/// Bounds-check every code in `codes` against `table` in one pass, then gather the values into
/// `sink`. Keeping the check outside the gather loop lets independent table loads execute together.
pub(crate) fn gather_checked<I: UnsignedPType, T: NativePType>(
    codes: &[I],
    table: &[T],
    sink: impl GatherSink<T>,
) -> VortexResult<()> {
    vortex_ensure!(
        codes
            .iter()
            .copied()
            .max()
            .and_then(|code| code.to_usize())
            .is_some_and(|max| max < table.len()),
        "dictionary code out of bounds for {} values",
        table.len()
    );
    // SAFETY: `GatherSink` is crate-private and implemented only for `&mut [T]` and
    // `&mut BlockOut`, both of which invoke the closure exclusively on elements of `codes`,
    // every one of which is `< table.len()` as checked above.
    sink.write_checked(codes, |code| unsafe { *table.get_unchecked(code.as_()) });
    Ok(())
}
