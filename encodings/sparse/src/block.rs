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
