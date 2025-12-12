// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;

use fastlanes::Delta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use num_traits::WrappingAdd;
use vortex_array::kernel::Kernel;
use vortex_array::kernel::KernelRef;
use vortex_array::kernel::PushDownResult;
use vortex_dtype::NativePType;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::delta::array::delta_decompress::decompress_primitive;

/// Holds the bound kernels and metadata needed to execute delta decompression.
pub struct DeltaKernel {
    pub(super) bases_kernel: KernelRef,
    pub(super) deltas_kernel: KernelRef,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) validity: Mask,
}

impl Kernel for DeltaKernel {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        // Extract all fields to avoid borrow issues.
        let DeltaKernel {
            bases_kernel,
            deltas_kernel,
            start,
            end,
            validity,
        } = *self;

        let bases = bases_kernel.execute()?.into_primitive();
        let deltas = deltas_kernel.execute()?.into_primitive();

        Ok(match bases {
            PrimitiveVector::U8(pv) => {
                decompress::<u8, { u8::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::U16(pv) => {
                decompress::<u16, { u16::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::U32(pv) => {
                decompress::<u32, { u32::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::U64(pv) => {
                decompress::<u64, { u64::LANES }>(&pv, &deltas, start, end, validity)
            }
            PrimitiveVector::I8(_)
            | PrimitiveVector::I16(_)
            | PrimitiveVector::I32(_)
            | PrimitiveVector::I64(_)
            | PrimitiveVector::F16(_)
            | PrimitiveVector::F32(_)
            | PrimitiveVector::F64(_) => {
                vortex_panic!("Tried to match a non-unsigned vector in an unsigned match statement")
            }
        })
    }

    fn push_down_filter(self: Box<Self>, _selection: &Mask) -> VortexResult<PushDownResult> {
        Ok(PushDownResult::NotPushed(self))
    }
}

/// Decompresses delta-encoded data for a specific primitive type.
fn decompress<T, const LANES: usize>(
    bases: &PVector<T>,
    deltas: &PrimitiveVector,
    start: usize,
    end: usize,
    validity: Mask,
) -> Vector
where
    T: NativePType + Delta + Transpose + WrappingAdd,
{
    let buffer = decompress_primitive::<T, LANES>(bases.as_ref(), deltas.downcast::<T>().as_ref());
    let buffer = buffer.slice(start..end);

    // SAFETY: We slice the buffer and the validity by the same range.
    unsafe { PVector::<T>::new_unchecked(buffer, validity) }.into()
}

impl Debug for DeltaKernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaKernel")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish_non_exhaustive()
    }
}
