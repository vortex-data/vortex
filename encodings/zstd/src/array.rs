// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::{BinaryView, PrimitiveArray, VarBinViewArray};
use vortex_array::compute::filter;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValiditySliceHelper, ValidityVTableFromValiditySliceHelper,
};
use vortex_array::{ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable};
use vortex_buffer::{Alignment, Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};
use vortex_mask::AllOr;
use vortex_scalar::Scalar;

use crate::serde::{ZstdFrameMetadata, ZstdMetadata};

// Zstd doesn't support training dictionaries on very few samples.
const MIN_SAMPLES_FOR_DICTIONARY: usize = 8;
type ViewLen = u32;

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

struct Frames {
    dictionary: Option<ByteBuffer>,
    frames: Vec<ByteBuffer>,
    frame_metas: Vec<ZstdFrameMetadata>,
}

fn choose_max_dict_size(uncompressed_size: usize) -> usize {
    // following recommendations from
    // https://github.com/facebook/zstd/blob/v1.5.5/lib/zdict.h#L190
    // that is, 1/100 the data size, up to 100kB.
    // It appears that zstd can't train dictionaries with <256 bytes.
    (uncompressed_size / 100).clamp(256, 100 * 1024)
}

fn collect_valid_primitive(parray: &PrimitiveArray) -> VortexResult<PrimitiveArray> {
    let mask = parray.validity_mask()?;
    filter(&parray.to_array(), &mask)?.to_primitive()
}

fn collect_valid_vbv(vbv: &VarBinViewArray) -> VortexResult<(ByteBuffer, Vec<usize>)> {
    let mask = vbv.validity_mask()?;
    let buffer_and_value_byte_indices = match mask.boolean_buffer() {
        AllOr::None => (Buffer::empty(), Vec::new()),
        _ => {
            let mut buffer =
                BufferMut::with_capacity(vbv.nbytes() + mask.true_count() * size_of::<ViewLen>());
            let mut value_byte_indices = Vec::new();
            vbv.with_iterator(|iterator| {
                // by flattening, we should omit nulls
                for value in iterator.flatten() {
                    value_byte_indices.push(buffer.len());
                    // here's where we write the string lengths
                    buffer
                        .extend_trusted(ViewLen::try_from(value.len())?.to_le_bytes().into_iter());
                    buffer.extend_from_slice(value);
                }
                Ok::<_, VortexError>(())
            })??;
            (buffer.freeze(), value_byte_indices)
        }
    };
    Ok(buffer_and_value_byte_indices)
}

fn reconstruct_views(buffer: ByteBuffer) -> VortexResult<Buffer<BinaryView>> {
    let mut res = BufferMut::<BinaryView>::empty();
    let mut offset = 0;
    while offset < buffer.len() {
        let str_len = ViewLen::from_le_bytes(
            buffer
                .get(offset..offset + size_of::<ViewLen>())
                .ok_or_else(|| vortex_err!("Zstd buffer for VarBinView was corrupt"))?
                .try_into()?,
        ) as usize;
        offset += size_of::<ViewLen>();
        let value = &buffer[offset..offset + str_len];
        res.push(BinaryView::make_view(value, 0, u32::try_from(offset)?));
        offset += str_len;
    }
    Ok(res.freeze())
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

    fn compress_values(
        value_bytes: &ByteBuffer,
        frame_byte_starts: &[usize],
        level: i32,
        values_per_frame: usize,
        n_values: usize,
    ) -> VortexResult<Frames> {
        let n_frames = frame_byte_starts.len();

        // Would-be sample sizes if we end up applying zstd dictionary
        let mut sample_sizes = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let frame_byte_end = frame_byte_starts
                .get(i + 1)
                .copied()
                .unwrap_or(value_bytes.len());
            sample_sizes.push(frame_byte_end - frame_byte_starts[i]);
        }
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
            let frame_byte_end = frame_byte_starts
                .get(i + 1)
                .copied()
                .unwrap_or(value_bytes.len());
            let uncompressed = &value_bytes.slice(frame_byte_starts[i]..frame_byte_end);
            let compressed = compressor
                .compress(uncompressed)
                .map_err(|err| VortexError::from(err).with_context("while compressing"))?;
            frame_metas.push(ZstdFrameMetadata {
                uncompressed_size: uncompressed.len() as u64,
                n_values: values_per_frame.min(n_values - i * values_per_frame) as u64,
            });
            frames.push(ByteBuffer::from(compressed));
        }

        Ok(Frames {
            dictionary,
            frames,
            frame_metas,
        })
    }

    pub fn from_primitive(
        parray: &PrimitiveArray,
        level: i32,
        values_per_frame: usize,
    ) -> VortexResult<Self> {
        let dtype = parray.dtype().clone();
        let byte_width = parray.ptype().byte_width();

        // We compress only the valid elements.
        let values = collect_valid_primitive(parray)?;
        let n_values = values.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let value_bytes = values.byte_buffer();
        let frame_byte_starts = (0..n_values * byte_width)
            .step_by(values_per_frame * byte_width)
            .collect::<Vec<_>>();
        let Frames {
            dictionary,
            frames,
            frame_metas,
        } = Self::compress_values(
            value_bytes,
            &frame_byte_starts,
            level,
            values_per_frame,
            n_values,
        )?;

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

    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        level: i32,
        values_per_frame: usize,
    ) -> VortexResult<Self> {
        // Approach for strings: we prefix each string with its length as a u32.
        // This is the same as what Parquet does. In some cases it may be better
        // to separate the binary data and lengths as two separate streams, but
        // this approach is simpler and can be best in cases when there is
        // mutual information between strings and their lengths.
        let dtype = vbv.dtype().clone();

        // We compress only the valid elements.
        let (value_bytes, value_byte_indices) = collect_valid_vbv(vbv)?;
        let n_values = value_byte_indices.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let frame_byte_starts = (0..n_values)
            .step_by(values_per_frame)
            .map(|i| value_byte_indices[i])
            .collect::<Vec<_>>();
        let Frames {
            dictionary,
            frames,
            frame_metas,
        } = Self::compress_values(
            &value_bytes,
            &frame_byte_starts,
            level,
            values_per_frame,
            n_values,
        )?;

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
            vbv.len(),
            vbv.validity().clone(),
        ))
    }

    pub fn from_canonical(
        canonical: &Canonical,
        level: i32,
        values_per_frame: usize,
    ) -> VortexResult<Option<Self>> {
        match canonical {
            Canonical::Primitive(parray) => Ok(Some(ZstdArray::from_primitive(
                parray,
                level,
                values_per_frame,
            )?)),
            Canonical::VarBinView(vbv) => Ok(Some(ZstdArray::from_var_bin_view(
                vbv,
                level,
                values_per_frame,
            )?)),
            _ => Ok(None),
        }
    }

    pub fn from_array(array: ArrayRef, level: i32, values_per_frame: usize) -> VortexResult<Self> {
        Self::from_canonical(&array.to_canonical()?, level, values_per_frame)?
            .ok_or_else(|| vortex_err!("Zstd can only encode Primitive and VarBinView arrays"))
    }

    fn byte_width(&self) -> usize {
        if self.dtype.is_primitive() {
            self.dtype.as_ptype().byte_width()
        } else {
            1
        }
    }

    pub fn decompress(&self) -> VortexResult<ArrayRef> {
        // To start, we figure out which frames we need to decompress, and with
        // what row offset into the first such frame.
        let byte_width = self.byte_width();
        let slice_n_rows = self.slice_stop - self.slice_start;
        let slice_value_indices = self
            .unsliced_validity
            .to_mask(self.unsliced_n_rows)?
            .valid_counts_for_indices(&[self.slice_start, self.slice_stop])?;

        let slice_value_idx_start = slice_value_indices[0];
        let slice_value_idx_stop = slice_value_indices[1];

        let mut frames_to_decompress = vec![];
        let mut value_idx_start = 0;
        let mut uncompressed_size_to_decompress = 0;
        let mut n_skipped_values = 0;
        for (frame, frame_meta) in self.frames.iter().zip(&self.metadata.frames) {
            if value_idx_start >= slice_value_idx_stop {
                break;
            }

            let frame_uncompressed_size = usize::try_from(frame_meta.uncompressed_size)?;
            let frame_n_values = if frame_meta.n_values == 0 {
                // possibly older primitive-only metadata that just didn't store this
                frame_uncompressed_size / byte_width
            } else {
                usize::try_from(frame_meta.n_values)?
            };

            let value_idx_stop = value_idx_start + frame_n_values;
            if value_idx_stop > slice_value_idx_start {
                // we need this frame
                frames_to_decompress.push(frame);
                uncompressed_size_to_decompress += frame_uncompressed_size;
            } else {
                n_skipped_values += frame_n_values;
            }
            value_idx_start = value_idx_stop;
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
                .decompress_to_buffer(frame.as_slice(), &mut decompressed[uncompressed_start..])
                .map_err(|err| VortexError::from(err).with_context("while decompressing"))?;
            uncompressed_start += uncompressed_written;
        }
        if uncompressed_start != uncompressed_size_to_decompress {
            vortex_bail!(
                "Zstd metadata or frames were corrupt; expected {} bytes but decompressed {}",
                uncompressed_size_to_decompress,
                uncompressed_start
            );
        }

        let decompressed = decompressed.freeze();
        // Last, we slice the exact values requested out of the decompressed data.
        let slice_validity = self
            .unsliced_validity
            .slice(self.slice_start, self.slice_stop)?;

        match &self.dtype {
            DType::Primitive(..) => {
                let slice_values_buffer = decompressed.slice(
                    (slice_value_idx_start - n_skipped_values) * byte_width
                        ..(slice_value_idx_stop - n_skipped_values) * byte_width,
                );
                let primitive = PrimitiveArray::from_values_byte_buffer(
                    slice_values_buffer,
                    self.dtype.as_ptype(),
                    slice_validity,
                    slice_n_rows,
                )?;

                Ok(primitive.into_array())
            }
            DType::Binary(_) | DType::Utf8(_) => {
                // The decompressed buffer is a bunch of interleaved u32 lengths
                // and strings of those lengths, we we need to reconstruct the
                // views into those strings by passing through the buffer.
                let views = reconstruct_views(decompressed.clone())?.slice(
                    slice_value_idx_start - n_skipped_values
                        ..slice_value_idx_stop - n_skipped_values,
                );

                let vbv = VarBinViewArray::try_new(
                    views,
                    vec![decompressed],
                    self.dtype.clone(),
                    slice_validity,
                )?;
                Ok(vbv.into_array())
            }
            _ => Err(vortex_err!(
                "Unsupported dtype for Zstd array: {:?}",
                self.dtype
            )),
        }
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
