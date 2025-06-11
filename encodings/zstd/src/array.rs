use std::fmt::Debug;

use vortex_array::arrays::{PrimitiveArray, PrimitiveVTable};
use vortex_array::compute::filter;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_err};
use vortex_scalar::Scalar;

use crate::serde::{ZstdFrameMetadata, ZstdMetadata};

// Zstd doesn't support training dictionaries on very few samples.
const MIN_SAMPLES_FOR_DICTIONARY: usize = 8;

// Overall approach here:
// Zstd can be used on the whole array (rows_per_frame = 0), resulting in a single Zstd
// frame, or it can be used with a dictionary (rows_per_frame < # rows), resulting in
// multiple Zstd frames sharing a common dictionary. This latter case is helpful if you
// want somewhat faster access to slices or individual rows, allowing us to only
// decompress the necessary frames.

// Visually, during compression and decompression, we have an interval of frames we're
// compressing/decompressing and a tighter interval of the slice we actually care about:
//
// |=====================validity========================|
// |=======================rows==========================|
//    |----------------frames_rows-------------------|
//    <--row_offset->|----slice-------------------|
//                   ^                            ^
//                   |<------slice_n_rows-------->|
//                slice_start                 slice_stop
//
// |=====values (all valid elements)====|
//     |-------frames_values------|
//         |----slice_values-----|

vtable!(Zstd);

impl VTable for ZstdVTable {
    type Array = ZstdArray;
    type Encoding = ZstdEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.zstd")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ZstdEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ZstdEncoding;

#[derive(Clone, Debug)]
pub struct ZstdArray {
    pub(crate) dictionary: Option<ByteBuffer>,
    pub(crate) frames: Vec<ByteBuffer>,
    pub(crate) validity: Validity,
    pub(crate) metadata: ZstdMetadata,
    dtype: DType,
    stats_set: ArrayStats,
    slice_start: usize,
    slice_stop: usize,
}

fn choose_max_dict_size(uncompressed_size: usize) -> usize {
    // following recommendations from
    // https://github.com/facebook/zstd/blob/v1.5.5/lib/zdict.h#L190
    // that is, 1/100 the data size, up to 100kB.
    // It appears that zstd can't train dictionaries with <256 bytes.
    (uncompressed_size / 100).clamp(256, 100 * 1024)
}

fn collect_valid(parray: &PrimitiveArray) -> VortexResult<PrimitiveArray> {
    let mask = parray.validity_mask()?;
    filter(&parray.to_array(), &mask)?.to_primitive()
}

impl ZstdArray {
    pub fn new(
        dictionary: Option<ByteBuffer>,
        frames: Vec<ByteBuffer>,
        dtype: DType,
        metadata: ZstdMetadata,
        n_rows: usize,
        validity: Validity,
    ) -> Self {
        Self {
            dictionary,
            frames,
            validity,
            metadata,
            dtype,
            stats_set: Default::default(),
            slice_start: 0,
            slice_stop: n_rows,
        }
    }

    pub fn uncompressed_size(&self) -> usize {
        (self.slice_stop - self.slice_start) * self.dtype.as_ptype().byte_width()
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        rows_per_frame: usize,
    ) -> VortexResult<Self> {
        let dtype = parray.dtype().clone();
        let byte_width = parray.ptype().byte_width();
        let mask = parray.validity_mask()?;
        let n_rows = parray.len();
        let rows_per_frame = if rows_per_frame > 0 {
            rows_per_frame
        } else {
            n_rows
        };
        let frame_row_indices = (0..n_rows).step_by(rows_per_frame).collect::<Vec<_>>();
        let n_frames = frame_row_indices.len();

        // We compress only the valid elements.
        let values = collect_valid(parray)?;
        let mut valid_counts = mask.valid_counts_for_indices(&frame_row_indices)?;
        valid_counts.push(values.len()); // for convenience
        let values = values.byte_buffer();
        let value_bytes = values.inner();

        // Would-be sample sizes if we end up applying zstd dictionary
        let sample_sizes: Vec<usize> = valid_counts
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) * byte_width)
            .filter(|&size| size > 0)
            .collect();
        debug_assert_eq!(sample_sizes.iter().sum::<usize>(), value_bytes.len());

        let (dictionary, mut compressor) = if sample_sizes.len() < MIN_SAMPLES_FOR_DICTIONARY {
            // no dictionary
            (None, zstd::bulk::Compressor::new(level)?)
        } else {
            // with dictionary
            let max_dict_size = choose_max_dict_size(values.len());
            let dict = zstd::dict::from_continuous(value_bytes, &sample_sizes, max_dict_size)
                .map_err(|err| VortexError::from(err).with_context("while training dictionary"))?;

            let compressor = zstd::bulk::Compressor::with_dictionary(level, &dict)?;
            (Some(ByteBuffer::from(dict)), compressor)
        };

        let mut frame_metas = vec![];
        let mut frames = vec![];
        for i in 0..n_frames {
            let uncompressed =
                &value_bytes.slice(valid_counts[i] * byte_width..valid_counts[i + 1] * byte_width);
            let compressed = compressor
                .compress(uncompressed)
                .map_err(|err| VortexError::from(err).with_context("while compressing"))?;
            let frame_n_rows = (frame_row_indices.get(i + 1).cloned().unwrap_or(n_rows)
                - frame_row_indices[i]) as u64;
            frame_metas.push(ZstdFrameMetadata {
                n_rows: frame_n_rows,
                compressed_size: compressed.len() as u64,
                uncompressed_size: uncompressed.len() as u64,
            });
            frames.push(ByteBuffer::from(compressed));
        }

        let metadata = ZstdMetadata {
            dictionary_size: dictionary
                .as_ref()
                .map_or(0, |dict| dict.len())
                .try_into()?,
            frames: frame_metas,
        };

        Ok(ZstdArray::new(
            dictionary,
            frames,
            dtype,
            metadata,
            n_rows,
            parray.validity().clone(),
        ))
    }

    pub fn from_array(array: ArrayRef, level: i32, rows_per_frame: usize) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            Self::from_primitive(parray, level, rows_per_frame)
        } else {
            Err(vortex_err!("Zstd can only encode primitive arrays"))
        }
    }

    pub fn decompress(&self) -> VortexResult<ArrayRef> {
        // To start, we figure out which frames we need to decompress, and with
        // what row offset into the first such frame.
        let slice_n_rows = self.slice_stop - self.slice_start;
        let byte_width = self.dtype.as_ptype().byte_width();
        let mut frame_start_row = 0;
        let mut frame_idx_lb = 0;
        let mut frame_idx_ub = 0;
        let mut row_offset = 0;
        for (i, frame_meta) in self.metadata.frames.iter().enumerate() {
            let buf_stop = frame_start_row + usize::try_from(frame_meta.n_rows)?;
            if frame_start_row < self.slice_start {
                frame_idx_lb = i;
                row_offset = self.slice_start - frame_start_row
            }
            if frame_start_row < self.slice_stop {
                frame_idx_ub = i + 1
            }
            frame_start_row = buf_stop;
        }

        // then we actually decompress those frames
        let frame_metas = &self.metadata.frames[frame_idx_lb..frame_idx_ub];
        let total_uncompressed_size: usize = frame_metas
            .iter()
            .map(|meta| meta.uncompressed_size)
            .sum::<u64>()
            .try_into()?;

        let mut decompressor = if let Some(dictionary) = &self.dictionary {
            zstd::bulk::Decompressor::with_dictionary(dictionary)
        } else {
            zstd::bulk::Decompressor::new()
        }?;

        // we could make this empty initialized for better performance
        let mut frames_values_bytes = ByteBufferMut::with_capacity_aligned(
            total_uncompressed_size,
            Alignment::new(byte_width),
        );
        unsafe {
            // safety: we immediately fill all bytes in the following loop,
            // assuming our metadata's uncompressed size is correct
            frames_values_bytes.set_len(total_uncompressed_size);
        }
        let mut start_byte = 0;
        for (frame, meta) in self.frames[frame_idx_lb..frame_idx_ub]
            .iter()
            .zip(frame_metas)
        {
            let stop_byte = start_byte + usize::try_from(meta.uncompressed_size)?;
            decompressor.decompress_to_buffer(
                frame.as_slice(),
                &mut frames_values_bytes[start_byte..stop_byte],
            )?;
            start_byte = stop_byte;
        }

        // Last, we apply our offset. We need to copy since the decompressed
        // frame start/end might not align with our slice. And we need to
        // align the data to our (dynamic) dtype.
        let frames_validity = self
            .validity
            .slice(self.slice_start - row_offset, self.slice_stop)?;
        let frames_mask = frames_validity.to_mask(row_offset + slice_n_rows)?;
        let frames_values_start_stop =
            frames_mask.valid_counts_for_indices(&[row_offset, row_offset + slice_n_rows])?;
        let slice_values_buffer = frames_values_bytes.freeze().slice(
            frames_values_start_stop[0] * byte_width..frames_values_start_stop[1] * byte_width,
        );

        let primitive = PrimitiveArray::from_values_byte_buffer(
            slice_values_buffer,
            self.dtype.as_ptype(),
            frames_validity.slice(row_offset, row_offset + slice_n_rows)?,
            slice_n_rows,
        )?;

        Ok(primitive.into_array())
    }

    fn _slice(&self, start: usize, stop: usize) -> ZstdArray {
        ZstdArray {
            slice_start: self.slice_start + start,
            slice_stop: self.slice_start + stop,
            ..self.clone()
        }
    }
}

impl ValidityHelper for ZstdArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<ZstdVTable> for ZstdVTable {
    fn len(array: &ZstdArray) -> usize {
        array.slice_stop - array.slice_start
    }

    fn dtype(array: &ZstdArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZstdArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ZstdVTable> for ZstdVTable {
    fn canonicalize(array: &ZstdArray) -> VortexResult<Canonical> {
        array.decompress()?.to_canonical()
    }
}

impl OperationsVTable<ZstdVTable> for ZstdVTable {
    fn slice(array: &ZstdArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(array._slice(start, stop).into_array())
    }

    fn scalar_at(array: &ZstdArray, index: usize) -> VortexResult<Scalar> {
        array._slice(index, index + 1).decompress()?.scalar_at(0)
    }
}
