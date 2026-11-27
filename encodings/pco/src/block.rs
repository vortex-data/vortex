// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode compact Pco pages and scatter their valid values directly into blocks.

use std::ops::Range;

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use pco::wrapped::ChunkDecompressor;
use pco::wrapped::FileDecompressor;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::block::BlockBuf;
use vortex_array::block::BlockSource;
use vortex_array::block::BlockValidity;
use vortex_array::block::SourceBlock;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Pco;
use crate::PcoArray;
use crate::PcoArrayExt;
use crate::array::number_type_from_ptype;
use crate::array::vortex_err_from_pco;

pub(crate) fn pco_source(
    array: ArrayView<'_, Pco>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    let unsliced_validity = array.unsliced_validity();
    // Dense Pco already decodes directly into a buffer that its parent can reuse in place.
    if unsliced_validity.definitely_no_nulls() {
        return Ok(None);
    }
    let mask = unsliced_validity.execute_mask(array.unsliced_n_rows(), ctx)?;
    let next_value = mask.valid_counts_for_indices(&[array.slice_start()])[0];
    let validity = mask.slice(array.slice_start()..array.slice_stop());
    if matches!(validity, Mask::AllTrue(_)) {
        return Ok(None);
    }
    let (file, _) =
        FileDecompressor::new(array.metadata.header.as_slice()).map_err(vortex_err_from_pco)?;
    Ok(Some(
        match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
            Box::new(Pages::<T> {
                array: array.into_owned(),
                file,
                chunk: None,
                chunk_idx: 0,
                page_in_chunk: 0,
                next_page: 0,
                decoded: Vec::new(),
                range: 0..0,
                validity,
                next_value,
            }) as Box<dyn BlockSource>
        }),
    ))
}

struct Pages<T: Number> {
    array: PcoArray,
    file: FileDecompressor,
    chunk: Option<ChunkDecompressor<T>>,
    chunk_idx: usize,
    page_in_chunk: usize,
    next_page: usize,
    decoded: Vec<T>,
    /// Next position in the stream of non-null values, in unsliced page coordinates.
    next_value: usize,
    range: Range<usize>,
    validity: Mask,
}

impl<T: Number + NativePType> Pages<T> {
    /// Position at the page containing `row`, skipping whole pages before a slice.
    fn seek(&mut self, row: usize) -> VortexResult<()> {
        while row >= self.range.end {
            let meta = self
                .array
                .metadata
                .chunks
                .get(self.chunk_idx)
                .ok_or_else(|| vortex_err!("Pco pages end before requested row {row}"))?;
            let Some(page) = meta.pages.get(self.page_in_chunk) else {
                self.chunk_idx += 1;
                self.page_in_chunk = 0;
                self.chunk = None;
                continue;
            };
            let len = page.n_values as usize;
            self.range = self.range.end..self.range.end + len;
            let page_idx = self.next_page;
            self.page_in_chunk += 1;
            self.next_page += 1;
            if self.range.end <= row {
                continue;
            }
            let chunk = match &mut self.chunk {
                Some(chunk) => chunk,
                slot @ None => slot.insert(
                    self.file
                        .chunk_decompressor::<T, _>(
                            self.array.chunk_metas[self.chunk_idx].as_slice(),
                        )
                        .map_err(vortex_err_from_pco)?
                        .0,
                ),
            };
            let mut page = chunk
                .page_decompressor(self.array.pages[page_idx].as_slice(), len)
                .map_err(vortex_err_from_pco)?;
            self.decoded.resize(len, T::default());
            let progress = page.read(&mut self.decoded).map_err(vortex_err_from_pco)?;
            vortex_ensure_eq!(progress.n_processed, len);
        }
        Ok(())
    }
}

impl<T: Number + NativePType> BlockSource for Pages<T> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let output = scratch.values_mut::<T>(rows.len());
        output.fill(T::default());
        if let Mask::Values(mask) = &self.validity {
            let bits = mask.bit_buffer().slice(rows.clone());
            let mut indices = bits.set_indices().peekable();
            while indices.peek().is_some() {
                self.seek(self.next_value)?;
                let remaining = &self.decoded[self.next_value - self.range.start..];
                for (&value, index) in remaining.iter().zip(&mut indices) {
                    output[index] = value;
                    self.next_value += 1;
                }
            }
        }
        Ok(SourceBlock::Scratch {
            offset: 0,
            len: rows.len(),
        })
    }
}

#[cfg(test)]
#[cfg(not(codspeed))]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::test_harness::fusion::fuse_parent;
    use vortex_array::test_harness::fusion::fusion_trace;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Pco;
    use crate::PcoData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    });

    /// Multi-chunk, multi-page nullable fixture:
    /// - `0..2500`: mixed nulls across small pages (90 values) and chunks (350 values)
    /// - `2500..4700`: 2200 contiguous nulls (covers an all-null slice > `MAX_UNFUSED_LEN`)
    /// - `4700..6900`: 2200 contiguous valid values (covers a no-null slice > `MAX_UNFUSED_LEN`)
    /// - `6900..7500`: mixed nulls in trailing pages/chunks
    fn multi_chunk_fixture() -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let plain = PrimitiveArray::from_option_iter((0..7500i32).map(|row| {
            let valid = if (2500..4700).contains(&row) {
                false
            } else if (4700..6900).contains(&row) {
                true
            } else {
                row % 5 != 0 && (row % 37) >= 9
            };
            valid.then_some(row * 19 - 12_345)
        }));
        let validity = plain.validity()?;
        let data =
            PcoData::from_primitive_with_values_per_chunk(plain.as_view(), 3, 350, 90, &mut ctx)?;
        assert!(data.metadata.chunks.len() >= 8);
        assert!(data.metadata.chunks[0].pages.len() >= 3);
        Ok(Pco::try_new(plain.dtype().clone(), data, validity)?.into_array())
    }

    #[rstest]
    #[case::unsliced(0..7500, true)]
    #[case::mid_first_page(25..3100, true)]
    #[case::skip_first_page_same_chunk(180..3300, true)]
    #[case::later_chunk_across_null_and_dense_runs(950..7400, true)]
    #[case::all_null_slice(2510..4690, true)]
    #[case::no_null_slice(4710..6890, false)]
    fn pco_pages_stream_across_pages_chunks_and_slices(
        #[case] slice: Range<usize>,
        #[case] expect_pco_source: bool,
    ) -> VortexResult<()> {
        let pco = multi_chunk_fixture()?.slice(slice)?;
        let trace = fusion_trace(fuse_parent(pco), &SESSION)?;
        assert_eq!(
            trace.contains("fuse start array=vortex.pco"),
            expect_pco_source,
            "{trace}"
        );
        Ok(())
    }

    #[test]
    fn pco_all_null_array_streams_without_pages() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let plain = PrimitiveArray::from_option_iter((0..2500).map(|_| None::<i64>));
        let pco = Pco::from_primitive(plain.as_view(), 3, 64, &mut ctx)?.into_array();
        let trace = fusion_trace(fuse_parent(pco), &SESSION)?;
        assert!(trace.contains("fuse start array=vortex.pco"), "{trace}");
        Ok(())
    }

    #[test]
    fn pco_pages_seeded_null_patterns_and_slices() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut state = 0x9e37_79b9_u32;
        let mut next_u32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for round in 0..12u32 {
            let len = 3200 + (next_u32() % 1800) as usize;
            let threshold = 15 + (round * 7) % 70;
            let plain = PrimitiveArray::from_option_iter((0..len).map(|row| {
                let roll = next_u32() % 100;
                (roll >= threshold)
                    .then_some(i64::try_from(row).unwrap_or(0) * 31 - i64::from(round) * 997)
            }));
            let validity = plain.validity()?;
            let values_per_page = 40 + (round as usize * 19) % 90;
            let values_per_chunk = values_per_page * (2 + (round as usize % 3)) + 17;
            let data = PcoData::from_primitive_with_values_per_chunk(
                plain.as_view(),
                3,
                values_per_chunk,
                values_per_page,
                &mut ctx,
            )?;
            let pco = Pco::try_new(plain.dtype().clone(), data, validity)?.into_array();

            let max_start = len - 2100;
            let start = (next_u32() as usize) % max_start;
            let span = 2060 + (next_u32() as usize) % (len - start - 2059);
            let sliced = pco.slice(start..start + span)?;
            let _trace = fusion_trace(fuse_parent(sliced), &SESSION)?;
        }
        Ok(())
    }
}
