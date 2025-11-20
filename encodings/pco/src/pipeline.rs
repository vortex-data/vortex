// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pco::data_types::{Number, NumberType};
use pco::match_number_enum;
use pco::wrapped::{ChunkDecompressor, FileDecompressor};
use vortex_array::pipeline::{
    BindContext, BitView, Kernel, KernelCtx, N, PipelineInputs, PipelinedNode,
};
use vortex_buffer::ByteBuffer;
use vortex_compute::expand::Expand;
use vortex_dtype::{NativePType, PTypeDowncastExt, half};
use vortex_error::{VortexResult, VortexUnwrap, vortex_err};
use vortex_mask::MaskMut;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::{Vector, VectorMutOps, VectorOps};

use crate::array::{number_type_from_dtype, vortex_err_from_pco};
use crate::{PcoArray, PcoMetadata};

impl PipelinedNode for PcoArray {
    fn inputs(&self) -> PipelineInputs {
        PipelineInputs::Source
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let number_type = number_type_from_dtype(self.dtype());
        match_number_enum!(
            number_type,
            NumberType<T> => {
                Ok(Box::new(PcoKernel::<T>::new(self)?))
            }
        )
    }
}

pub struct PcoKernel<T: Number + NativePType> {
    file_decompressor: FileDecompressor,
    chunk_decompressor: Option<ChunkDecompressor<T>>,

    chunk_metas: Vec<ByteBuffer>,
    pages: Vec<ByteBuffer>,
    metadata: PcoMetadata,
    validity: MaskMut,

    current_chunk_idx: usize,
    current_page_idx_in_chunk: usize,
    global_page_idx: usize,
    page_position: usize, // Position within current page
    page_buffer: Vec<T>,  // Buffer for current page
    values_processed: usize,
}

impl<T: Number + NativePType> PcoKernel<T> {
    pub fn new(array: &PcoArray) -> VortexResult<Self> {
        let (fd, _) = FileDecompressor::new(array.metadata.header.as_slice())
            .map_err(vortex_err_from_pco)
            .vortex_unwrap();

        Ok(Self {
            file_decompressor: fd,
            chunk_decompressor: None,
            chunk_metas: array.chunk_metas.clone(),
            pages: array.pages.clone(),
            metadata: array.metadata.clone(),
            validity: array
                .unsliced_validity
                .to_mask(array.unsliced_n_rows())
                .into_mut(),
            current_chunk_idx: 0,
            current_page_idx_in_chunk: 0,
            global_page_idx: 0,
            page_position: 0,
            page_buffer: Vec::new(),
            values_processed: 0,
        })
    }

    fn decompress_current_page(&mut self) -> VortexResult<()> {
        // Ensure the chunk decompressor is set.
        if self.chunk_decompressor.is_none() {
            let chunk_meta_bytes: &[u8] = self.chunk_metas[self.current_chunk_idx].as_ref();
            let (chunk_decompressor, _) = self
                .file_decompressor
                .chunk_decompressor(chunk_meta_bytes)
                .map_err(vortex_err_from_pco)?;
            self.chunk_decompressor = Some(chunk_decompressor);
        }

        let chunk_info = &self.metadata.chunks[self.current_chunk_idx];
        let page_n_values = chunk_info.pages[self.current_page_idx_in_chunk].n_values as usize;
        let page_bytes: &[u8] = self.pages[self.global_page_idx].as_ref();

        if self.page_buffer.capacity() < page_n_values {
            self.page_buffer
                .reserve(page_n_values - self.page_buffer.capacity());
        }
        unsafe {
            self.page_buffer.set_len(page_n_values);
        }

        let chunk_decompressor = self
            .chunk_decompressor
            .as_mut()
            .ok_or_else(|| vortex_err!("No chunk decompressor available"))?;

        let mut page_decompressor = chunk_decompressor
            .page_decompressor(page_bytes, page_n_values)
            .map_err(vortex_err_from_pco)?;

        page_decompressor
            .decompress(&mut self.page_buffer)
            .map_err(vortex_err_from_pco)?;

        Ok(())
    }

    fn advance_to_next_page(&mut self) {
        // SAFETY: Setting the length to 0 is always safe.
        unsafe {
            self.page_buffer.set_len(0);
        }
        self.page_position = 0;
        self.current_page_idx_in_chunk += 1;
        self.global_page_idx += 1;

        if self.current_chunk_idx < self.metadata.chunks.len() {
            let chunk_info = &self.metadata.chunks[self.current_chunk_idx];
            if self.current_page_idx_in_chunk >= chunk_info.pages.len() {
                self.current_chunk_idx += 1;
                self.current_page_idx_in_chunk = 0;
                self.chunk_decompressor = None;
            }
        }
    }
}

impl<T: Number + NativePType> Kernel for PcoKernel<T> {
    fn step(&mut self, _ctx: &KernelCtx, selection: &BitView, out: Vector) -> VortexResult<Vector> {
        let remaining_validity = self.validity.split_off(N.min(self.validity.len()));
        let step_validity = std::mem::take(&mut self.validity).freeze();
        let step_true_count = step_validity.true_count();
        self.validity = remaining_validity;

        if selection.true_count() == 0 {
            debug_assert!(out.is_empty());
            return Ok(out);
        }

        let (elements, _validity) = out.into_primitive().downcast::<T>().into_parts();

        let mut elements = elements.into_mut();

        while elements.len() < step_true_count {
            // Ensure the page to read is decompressed.
            if self.page_buffer.is_empty() {
                self.decompress_current_page()?;
            }

            let remaining_in_page = self.page_buffer.len() - self.page_position;
            let copy_count = (step_true_count - elements.len()).min(remaining_in_page);
            let page_slice = &self.page_buffer[self.page_position..][..copy_count];

            // SAFETY: Sufficient capacity is pre-allocated.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    page_slice.as_ptr() as _,
                    elements.spare_capacity_mut().as_mut_ptr(),
                    copy_count,
                );
                elements.set_len(elements.len() + copy_count);
            }

            self.page_position += copy_count;
            self.values_processed += copy_count;

            if self.page_position >= self.page_buffer.len() {
                self.advance_to_next_page();
            }
        }

        let mut vec = PVectorMut::new(elements.expand(&step_validity), step_validity.into_mut());
        if vec.len() < N && vec.len() > selection.true_count() {
            vec.append_values(T::default(), N - vec.len());
        }

        Ok(vec.freeze().into())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;

    use crate::PcoArray;

    const COMPRESSION_LEVEL: usize = 3;
    const CHUNK_SIZE: usize = 512;
    const PAGE_SIZE: usize = 128;

    #[rstest]
    #[case(50)]
    #[case(100)]
    #[case(1024)]
    #[case(1025)]
    #[case(2048)]
    #[case(3000)]
    #[case(5120)]
    fn test_pco_pipeline_roundtrip(#[case] array_size: usize) {
        let values: Vec<i32> = (0..array_size).map(|i| i32::try_from(i).unwrap()).collect();
        let primitive = PrimitiveArray::from_iter(values);

        let pco_array = PcoArray::from_primitive_with_values_per_chunk(
            &primitive,
            COMPRESSION_LEVEL,
            CHUNK_SIZE,
            PAGE_SIZE,
        )
        .unwrap();

        let mask = Mask::new_true(array_size);
        let result = pco_array.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(result.len(), array_size);

        let pvector = result.as_primitive().into_i32();
        let result_vec: Vec<i32> = pvector.elements().to_vec();
        let expected_vec: Vec<i32> = primitive.as_slice::<i32>().to_vec();
        assert_eq!(result_vec, expected_vec);
    }

    #[rstest]
    #[case(50)]
    #[case(100)]
    #[case(1024)]
    #[case(1025)]
    #[case(2048)]
    #[case(3000)]
    #[case(5120)]
    fn test_pco_pipeline_with_mixed_mask(#[case] array_size: usize) {
        let values: Vec<i32> = (0..array_size).map(|i| i32::try_from(i).unwrap()).collect();
        let primitive = PrimitiveArray::from_iter(values);

        let pco_array = PcoArray::from_primitive_with_values_per_chunk(
            &primitive,
            COMPRESSION_LEVEL,
            CHUNK_SIZE,
            PAGE_SIZE,
        )
        .unwrap();

        let mask_bits: Vec<bool> = (0..array_size).map(|i| i % 2 == 0).collect();
        let mask = Mask::from_iter(mask_bits.iter().copied());

        let result = pco_array.to_array().execute_with_selection(&mask).unwrap();

        let expected_len = mask_bits.iter().filter(|&&b| b).count();
        assert_eq!(result.len(), expected_len);
        let pvector_i32 = result.as_primitive().into_i32();

        let expected_values: Vec<i32> = (0..array_size)
            .filter(|i| i % 2 == 0)
            .map(|i| i32::try_from(i).unwrap())
            .collect();
        let result_vec: Vec<i32> = pvector_i32.elements().to_vec();
        assert_eq!(result_vec, expected_values);
    }

    #[rstest]
    #[case(50)]
    #[case(100)]
    #[case(1024)]
    #[case(1025)]
    #[case(2048)]
    #[case(3000)]
    #[case(5120)]
    fn test_pco_pipeline_with_validity(#[case] array_size: usize) {
        // Create array with alternating null values: [0, null, 2, null, 4, null, ...]
        let values: Vec<Option<i32>> = (0..array_size)
            .map(|i| (i % 2 == 0).then(|| i32::try_from(i).unwrap()))
            .collect();
        let primitive = PrimitiveArray::from_option_iter(values.iter().cloned());

        let pco_array = PcoArray::from_primitive_with_values_per_chunk(
            &primitive,
            COMPRESSION_LEVEL,
            CHUNK_SIZE,
            PAGE_SIZE,
        )
        .unwrap();

        let mask = Mask::new_true(array_size);
        let result = pco_array.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(result.len(), array_size);

        let pvector = result.as_primitive().into_i32();
        let result_slice = pvector.elements();
        let expected_slice = primitive.as_slice::<i32>();

        assert_eq!(result_slice.as_slice(), expected_slice);
    }

    #[rstest]
    #[case(100, 10, 50)]
    #[case(100, 0, 50)]
    #[case(100, 50, 100)]
    #[case(256, 20, 100)]
    #[case(512, 100, 300)]
    #[case(1024, 0, 256)]
    #[case(1024, 512, 768)]
    #[case(1024, 768, 1024)]
    #[case(4000, 0, 256)]
    #[case(4000, 512, 768)]
    #[case(4000, 768, 1024)]
    fn test_pco_pipeline_with_slice_offsets(
        #[case] array_size: usize,
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) {
        let values: Vec<i32> = (0..array_size).map(|i| i32::try_from(i).unwrap()).collect();
        let primitive = PrimitiveArray::from_iter(values);

        let pco_array = PcoArray::from_primitive_with_values_per_chunk(
            &primitive,
            COMPRESSION_LEVEL,
            CHUNK_SIZE,
            PAGE_SIZE,
        )
        .unwrap();

        let sliced_pco_array = pco_array.slice(slice_start..slice_end);
        assert_eq!(sliced_pco_array.len(), slice_end - slice_start);

        let decompressed = sliced_pco_array.to_primitive();
        assert_eq!(decompressed.len(), slice_end - slice_start);

        let expected_values: Vec<i32> = (slice_start..slice_end)
            .map(|i| i32::try_from(i).unwrap())
            .collect();
        let result_slice = decompressed.as_slice::<i32>();
        assert_eq!(result_slice, expected_values.as_slice());
    }
}
