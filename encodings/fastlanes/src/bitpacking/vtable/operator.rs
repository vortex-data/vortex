// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): Refactor this entire module!

use std::any::Any;
use std::cmp::min;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use fastlanes::{BitPacking, FastLanes};
use vortex_array::operator::{
    LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{
    BindContext, Element, Kernel, KernelContext, N, PipelinedOperator, RowSelection,
};
use vortex_array::vtable::OperatorVTable;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, PhysicalPType, match_each_integer_ptype};
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedVTable};

impl OperatorVTable<BitPackedVTable> for BitPackedVTable {
    fn to_operator(array: &BitPackedArray) -> VortexResult<Option<OperatorRef>> {
        if array.dtype.is_nullable() {
            log::trace!("BitPackedVTable does not support nullable arrays");
            return Ok(None);
        }
        if array.patches.is_some() {
            log::trace!("BitPackedVTable does not support nullable arrays");
            return Ok(None);
        }
        if array.offset != 0 {
            log::trace!("BitPackedVTable does not support non-zero offsets");
            return Ok(None);
        }

        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for BitPackedArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.offset.hash(state);
        self.len.hash(state);
        self.dtype.hash(state);
        self.bit_width.hash(state);
        self.packed.operator_hash(state);
        // We don't care about patches because they're not yet supported by the operator.
        // OperatorHash(&self.patches).hash(state);
        self.validity.operator_hash(state);
    }
}

impl OperatorEq for BitPackedArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.offset == other.offset
            && self.len == other.len
            && self.dtype == other.dtype
            && self.bit_width == other.bit_width
            && self.packed.operator_eq(&other.packed)
            && self.validity.operator_eq(&other.validity)
    }
}

impl Operator for BitPackedArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn bounds(&self) -> LengthBounds {
        self.len.into()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }
}

impl PipelinedOperator for BitPackedArray {
    fn row_selection(&self) -> RowSelection {
        RowSelection::Domain(self.len)
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        assert!(self.bit_width > 0);
        match_each_integer_ptype!(self.ptype(), |T| {
            let packed_stride =
                self.bit_width as usize * <<T as PhysicalPType>::Physical as FastLanes>::LANES;
            let buffer = Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
                self.packed.clone().into_byte_buffer(),
            );

            if self.offset == 0 {
                Ok(Box::new(BitPackedKernel::<T>::new(
                    self.bit_width as usize,
                    packed_stride,
                    buffer,
                )) as Box<dyn Kernel>)
            } else {
                // TODO(ngates): the unaligned kernel needs fixing for the non-masked API
                // Ok(Box::new(BitPackedUnalignedKernel::<T>::new(
                //     self.bit_width as usize,
                //     packed_stride,
                //     buffer,
                //     0,
                //     self.offset,
                // )) as Box<dyn Kernel>)
                unreachable!("Offset must be zero")
            }
        })
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
#[derive(Clone)]
pub struct BitPackedKernel<T: PhysicalPType<Physical: BitPacking>> {
    width: usize,
    packed_stride: usize,
    buffer: Buffer<<T as PhysicalPType>::Physical>,
}

impl<T: PhysicalPType<Physical: BitPacking>> BitPackedKernel<T> {
    pub fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
    ) -> Self {
        Self {
            width,
            packed_stride,
            buffer,
        }
    }
}

impl<T> Kernel for BitPackedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn step(
        &self,
        _ctx: &KernelContext,
        chunk_idx: usize,
        _selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        assert_eq!(
            N % 1024,
            0,
            "BitPackedKernel assumes N is a multiple of 1024"
        );

        // We re-interpret the output view as the unsigned bitpacked type.
        out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = out.as_array_mut::<<T as PhysicalPType>::Physical>();

        let packed_offset = ((chunk_idx * N) / 1024) * self.packed_stride;
        let packed = &self.buffer.as_slice()[packed_offset..];

        // We compute the number of FastLanes vectors for this chunk.
        let nvecs = min(N / 1024, packed.len() / self.packed_stride);

        for i in 0..nvecs {
            // TODO(ngates): decide if the selection mask is sufficiently sparse to warrant
            //  unpacking only the selected elements.
            unsafe {
                BitPacking::unchecked_unpack(
                    self.width,
                    &packed[(i * self.packed_stride)..][..self.packed_stride],
                    &mut elements[(i * 1024)..],
                );
            }
        }

        out.reinterpret_as::<T>();

        Ok(())
    }
}
