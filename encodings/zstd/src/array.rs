// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::arrays::{PrimitiveArray, PrimitiveVTable};
use vortex_array::compute::filter;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValiditySliceHelper, ValidityVTableFromValiditySliceHelper,
};
use vortex_array::{ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable};
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::serde::{ZstdFrameMetadata, ZstdMetadata};

// Zstd doesn't support training dictionaries on very few samples.
const MIN_SAMPLES_FOR_DICTIONARY: usize = 8;

// Overall approach here:
// Zstd can be used on the whole array (values_per_frame = 0), resulting in a single Zstd
// frame, or it can be used with a dictionary (values_per_frame < # values), resulting in
// multiple Zstd frames sharing a common dictionary. This latter case is helpful if you
// want somewhat faster access to slices or individual rows, allowing us to only
// decompress the necessary frames.

// Visually, during decompression, we have an interval of frames we're
// decompressing and a tighter interval of the slice we actually care about.
// |=============values (all valid elements)==============|
// |<-skipped_uncompressed->|----decompressed-------------|
//                              |------slice-------|
//                              ^                  ^
// |<-slice_uncompressed_start->|                  |
// |<------------slice_uncompressed_stop---------->|
// We then insert these values to the correct position using a primitive array
// constructor.

vtable!(Zstd);

impl VTable for ZstdVTable {
    type Array = ZstdArray;
    type Encoding = ZstdEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValiditySliceHelper;
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
    pub(crate) metadata: ZstdMetadata,
    dtype: DType,
    pub(crate) unsliced_validity: Validity,
    unsliced_n_rows: usize,
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
            metadata,
            dtype,
            unsliced_validity: validity,
            unsliced_n_rows: n_rows,
            stats_set: Default::default(),
            slice_start: 0,
            slice_stop: n_rows,
        }
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        values_per_frame: usize,
    ) -> VortexResult<Self> {
        let dtype = parray.dtype().clone();
        let byte_width = parray.ptype().byte_width();

        // We compress only the valid elements.
        let values = collect_valid(parray)?;
        let n_values = values.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let mut frame_value_starts = (0..n_values).step_by(values_per_frame).collect::<Vec<_>>();
        let n_frames = frame_value_starts.len();
        frame_value_starts.push(values.len()); // for convenience, include the stop of the last frame
        let value_bytes = values.byte_buffer();

        // Would-be sample sizes if we end up applying zstd dictionary
        let sample_sizes: Vec<usize> = frame_value_starts
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) * byte_width)
            .collect();
        debug_assert_eq!(sample_sizes.iter().sum::<usize>(), value_bytes.len());

        let (dictionary, mut compressor) = if sample_sizes.len() < MIN_SAMPLES_FOR_DICTIONARY {
            // no dictionary
            (None, zstd::bulk::Compressor::new(level)?)
        } else {
            // with dictionary
            let max_dict_size = choose_max_dict_size(value_bytes.len());
            let dict = zstd::dict::from_continuous(value_bytes, &sample_sizes, max_dict_size)
                .map_err(|err| VortexError::from(err).with_context("while training dictionary"))?;

            let compressor = zstd::bulk::Compressor::with_dictionary(level, &dict)?;
            (Some(ByteBuffer::from(dict)), compressor)
        };

        let mut frame_metas = vec![];
        let mut frames = vec![];
        for i in 0..n_frames {
            let uncompressed = &value_bytes
                .slice(frame_value_starts[i] * byte_width..frame_value_starts[i + 1] * byte_width);
            let compressed = compressor
                .compress(uncompressed)
                .map_err(|err| VortexError::from(err).with_context("while compressing"))?;
            frame_metas.push(ZstdFrameMetadata {
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
            parray.len(),
            parray.validity().clone(),
        ))
    }

    pub fn from_array(array: ArrayRef, level: i32, values_per_frame: usize) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<PrimitiveVTable>() {
            Self::from_primitive(parray, level, values_per_frame)
        } else {
            Err(vortex_err!("Zstd can only encode primitive arrays"))
        }
    }

    pub fn decompress(&self) -> VortexResult<ArrayRef> {
        // To start, we figure out which frames we need to decompress, and with
        // what row offset into the first such frame.
        let ptype = self.dtype.as_ptype();
        let byte_width = ptype.byte_width();
        let slice_n_rows = self.slice_stop - self.slice_start;
        let slice_value_indices = self
            .unsliced_validity
            .to_mask(self.unsliced_n_rows)?
            .valid_counts_for_indices(&[self.slice_start, self.slice_stop])?;
        let slice_uncompressed_start = slice_value_indices[0] * byte_width;
        let slice_uncompressed_stop = slice_value_indices[1] * byte_width;

        let mut frames_to_decompress = vec![];
        let mut uncompressed_start = 0;
        let mut uncompressed_size_to_decompress = 0;
        let mut skipped_uncompressed = 0;
        for (frame, frame_meta) in self.frames.iter().zip(&self.metadata.frames) {
            if uncompressed_start >= slice_uncompressed_stop {
                break;
            }
            let frame_uncompressed = usize::try_from(frame_meta.uncompressed_size)?;

            let uncompressed_stop = uncompressed_start + frame_uncompressed;
            if uncompressed_stop > slice_uncompressed_start {
                // we need this frame
                frames_to_decompress.push(frame);
                uncompressed_size_to_decompress += frame_uncompressed;
            } else {
                skipped_uncompressed += frame_uncompressed;
            }
            uncompressed_start = uncompressed_stop;
        }

        // then we actually decompress those frames
        let mut decompressor = if let Some(dictionary) = &self.dictionary {
            zstd::bulk::Decompressor::with_dictionary(dictionary)
        } else {
            zstd::bulk::Decompressor::new()
        }?;
        let mut decompressed = ByteBufferMut::with_capacity_aligned(
            uncompressed_size_to_decompress,
            Alignment::new(byte_width),
        );
        unsafe {
            // safety: we immediately fill all bytes in the following loop,
            // assuming our metadata's uncompressed size is correct
            decompressed.set_len(uncompressed_size_to_decompress);
        }
        let mut uncompressed_start = 0;
        for frame in frames_to_decompress {
            let uncompressed_written = decompressor
                .decompress_to_buffer(frame.as_slice(), &mut decompressed[uncompressed_start..])?;
            uncompressed_start += uncompressed_written;
        }
        if uncompressed_start != uncompressed_size_to_decompress {
            vortex_bail!(
                "Zstd metadata or frames were corrupt; expected {} byte but decompressed {}",
                uncompressed_size_to_decompress,
                uncompressed_start
            );
        }

        // Last, we slice the exact values requested out of the decompressed data.
        let slice_validity = self
            .unsliced_validity
            .slice(self.slice_start, self.slice_stop)?;
        let slice_values_buffer = decompressed.freeze().slice(
            slice_uncompressed_start - skipped_uncompressed
                ..slice_uncompressed_stop - skipped_uncompressed,
        );

        let primitive = PrimitiveArray::from_values_byte_buffer(
            slice_values_buffer,
            ptype,
            slice_validity,
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

impl ValiditySliceHelper for ZstdArray {
    fn unsliced_validity_and_slice(&self) -> (&Validity, usize, usize) {
        (&self.unsliced_validity, self.slice_start, self.slice_stop)
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
