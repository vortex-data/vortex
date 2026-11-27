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
use crate::block::BlockSource;
use crate::block::BlockValidity;
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
        let output = scratch.values_mut::<T>(len);
        let codes = &self.codes[rows];
        let table = self.table.as_slice();
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
        for (out, &code) in output.iter_mut().zip(codes) {
            // SAFETY: Every code is at most the maximum checked above.
            *out = unsafe { *table.get_unchecked(code.as_()) };
        }
        Ok(SourceBlock::Scratch { offset: 0, len })
    }
}
