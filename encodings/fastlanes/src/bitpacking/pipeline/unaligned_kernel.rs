use std::mem::MaybeUninit;
use fastlanes::BitPacking;
use vortex_array::pipeline::{Element, Kernel, KernelContext, N};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::view::ViewMut;
use vortex_buffer::Buffer;
use vortex_dtype::PhysicalPType;
use vortex_error::VortexResult;

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
pub(crate) struct BitPackedUnalignedKernel<T: PhysicalPType<Physical: BitPacking>> {
    width: usize,
    packed_stride: usize,

    buffer: Buffer<<T as PhysicalPType>::Physical>,
    packed_offset: usize,
    value_offset: u16,
}

impl<T> BitPackedUnalignedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
pub     fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
        packed_offset: usize,
        value_offset: u16,
    ) -> Self {
        assert!(value_offset < 1024);
        BitPackedUnalignedKernel::<T> {
            width,
            packed_stride,
            buffer,
            packed_offset,
            value_offset,
        }
    }

    fn unpack_sliced_chunk(
        &self,
        packed_chunk: &[<T as PhysicalPType>::Physical],
        temp_buffer: &mut [MaybeUninit<<T as PhysicalPType>::Physical>; 1024],
        output: &mut [<T as PhysicalPType>::Physical],
        source_offset: usize,
    ) {
        unsafe {
            let temp_slice = std::slice::from_raw_parts_mut(
                temp_buffer.as_mut_ptr() as *mut <T as PhysicalPType>::Physical,
                1024,
            );
            BitPacking::unchecked_unpack(self.width, packed_chunk, temp_slice);

            let copy_count = output.len();
            output.copy_from_slice(&temp_slice[source_offset..source_offset + copy_count]);
        }
    }
}

impl<T> Kernel for BitPackedUnalignedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    #[allow(clippy::unwrap_in_result, clippy::expect_used)]
    fn step(
        &mut self,
        _ctx: &KernelContext,
        selected: BitView,
        physical_out: &mut ViewMut,
    ) -> VortexResult<()> {
        let mut temp_buffer: [MaybeUninit<<T as PhysicalPType>::Physical>; 1024] =
            [const { MaybeUninit::uninit() }; 1024];
        // We re-interpret the output view as the unsigned bitpacked type.
        physical_out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = physical_out.as_slice_mut::<<T as PhysicalPType>::Physical>();
        let packed = &self.buffer.as_slice()[self.packed_offset..];

        let chunk_value_offset = self.value_offset as usize;

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        if selected.true_count() > 8 {
            let mut output_idx = 0;

            // Pre-calculate what we need to do
            let first_chunk_needs_slicing = chunk_value_offset > 0;
            let elements_from_first_chunk =
                (1024 - chunk_value_offset).min(elements.len());

            let elements_after_first = elements.len() - elements_from_first_chunk;
            let full_chunks_count = elements_after_first / 1024;
            let final_chunk_size = elements_after_first % 1024;
            let final_chunk_needs_slicing = final_chunk_size > 0;

            let total_chunks_needed = 1
                + full_chunks_count
                + (final_chunk_needs_slicing as usize);
            let available_chunks = packed.len() / self.packed_stride;
            let actual_chunks_to_process = total_chunks_needed.min(available_chunks);

            // Part 1: Handle first sliced chunk (if there's a value_offset)
            if actual_chunks_to_process > 0 {
                self.unpack_sliced_chunk(
                    &packed[0..self.packed_stride],
                    &mut temp_buffer,
                    &mut elements[output_idx..output_idx + elements_from_first_chunk],
                    chunk_value_offset,
                );
                output_idx += elements_from_first_chunk;
            }

            // Part 2: Handle all non-sliced full chunks (for loop)
            let last_full_chunk_idx = full_chunks_count + 1;

            for packed_idx in
                1..last_full_chunk_idx.min(actual_chunks_to_process)
            {
                unsafe {
                    BitPacking::unchecked_unpack(
                        self.width,
                        &packed[(packed_idx * self.packed_stride)..][..self.packed_stride],
                        &mut elements[output_idx..output_idx + 1024],
                    );
                }
                output_idx += 1024;
            }

            // Part 3: Handle final sliced chunk (if needed)
            if last_full_chunk_idx < actual_chunks_to_process {
                self.unpack_sliced_chunk(
                    &packed[(last_full_chunk_idx * self.packed_stride)..][..self.packed_stride],
                    &mut temp_buffer,
                    &mut elements[output_idx..output_idx + final_chunk_size],
                    0,
                );
            }

            let nvecs = (first_chunk_needs_slicing as usize) + full_chunks_count;

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            physical_out.select_mask::<<T as PhysicalPType>::Physical>(&selected);
        } else {
            let mut offset = 0;
            selected.iter_ones(|idx| {
                let adjusted_idx = idx + chunk_value_offset;
                let chunk_idx = adjusted_idx / 1024;
                let bit_idx = adjusted_idx % 1024;

                let start_idx = chunk_idx * self.packed_stride;
                if start_idx + self.packed_stride <= packed.len() {
                    unsafe {
                        *elements.get_unchecked_mut(offset) = BitPacking::unchecked_unpack_single(
                            self.width,
                            &packed[start_idx..start_idx + self.packed_stride],
                            bit_idx,
                        );
                    }
                } else {
                    // Not enough packed data - set to default value
                    elements[offset] = Default::default();
                }
                offset += 1;
            });

            let elements_needed = elements.len() + chunk_value_offset;
            let chunks_needed = elements_needed.div_ceil(1024);
            let nvecs = chunks_needed
                .min(packed.len() / self.packed_stride)
                .min(N / 1024);
            self.packed_offset += nvecs * self.packed_stride;
        }

        Ok(())
    }
}
