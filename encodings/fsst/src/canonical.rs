// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;
use std::sync::Arc;

use fsst::Decompressor;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::FSST;
use crate::FSSTArrayExt;
use crate::FSSTArraySlotsExt;

pub(super) fn canonicalize_fsst(
    array: ArrayView<'_, FSST>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (buffers, views) = fsst_decode_views(array, 0, ctx)?;
    // SAFETY: FSST already validates the bytes for binary/UTF-8. We build views directly on
    //  top of them, so the view pointers will all be valid.
    Ok(unsafe {
        VarBinViewArray::new_unchecked(
            views,
            Arc::from(buffers),
            array.dtype().clone(),
            array.codes().validity()?,
        )
        .into_array()
    })
}

/// Extra headroom that keeps [`Decompressor::decompress_into`] on its fast path.
///
/// It never writes past the slice it is given — every store is bounded by the end of the output —
/// but it emits whole 8-byte symbols, so it can only use the wide-store loop while at least 8
/// bytes remain. Handing it 7 spare bytes lets that loop run through the final value instead of
/// finishing byte at a time. This is a performance knob, not a safety requirement.
///
/// [`Decompressor::decompress_into`]: fsst::Decompressor::decompress_into
pub(crate) const FSST_DECODE_SLACK: usize = 7;

/// Everything needed to decode an FSST array's values in one bulk `decompress_into` call.
pub(crate) struct FsstDecodePlan {
    codes: ByteBuffer,
    /// Per-row uncompressed lengths, zero for null rows.
    pub(crate) lengths: PrimitiveArray,
    /// Total decoded size, i.e. the sum of `lengths`.
    pub(crate) total_size: usize,
}

impl FsstDecodePlan {
    pub(crate) fn new(
        fsst_array: ArrayView<'_, FSST>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let codes = fsst_array.codes().sliced_bytes();
        let lengths = fsst_array
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;

        #[expect(clippy::cast_possible_truncation)]
        let total_size: usize = match_each_integer_ptype!(lengths.ptype(), |P| {
            lengths.as_slice::<P>().iter().map(|x| *x as usize).sum()
        });

        Ok(Self {
            codes,
            lengths,
            total_size,
        })
    }

    /// Bulk-decompresses the whole code stream into `out`, which must hold at least
    /// `total_size + FSST_DECODE_SLACK` bytes.
    ///
    /// Kept inlinable so the decoder is not called from behind an extra frame in whichever
    /// codegen unit the caller lands in; see OnPair's equivalent for why that matters.
    #[inline]
    pub(crate) fn decode_into(
        &self,
        decompressor: &Decompressor<'_>,
        out: &mut [MaybeUninit<u8>],
    ) -> VortexResult<usize> {
        let len = decompressor.decompress_into(self.codes.as_slice(), out);
        vortex_ensure!(
            len == self.total_size,
            "FSST decoded {len} bytes, expected {}",
            self.total_size
        );
        Ok(len)
    }
}

pub(crate) fn fsst_decode_bytes(
    fsst_array: ArrayView<'_, FSST>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ByteBufferMut, PrimitiveArray)> {
    let plan = FsstDecodePlan::new(fsst_array, ctx)?;
    let mut uncompressed_bytes = ByteBufferMut::with_capacity(plan.total_size + FSST_DECODE_SLACK);
    let len = plan.decode_into(
        &fsst_array.decompressor(),
        uncompressed_bytes.spare_capacity_mut(),
    )?;
    // SAFETY: `decode_into` initialized the first `len` bytes.
    unsafe { uncompressed_bytes.set_len(len) };
    Ok((uncompressed_bytes, plan.lengths))
}

pub(crate) fn fsst_decode_views(
    fsst_array: ArrayView<'_, FSST>,
    start_buf_index: u32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<ByteBuffer>, Buffer<BinaryView>)> {
    let (uncompressed_bytes, uncompressed_lens_array) = fsst_decode_bytes(fsst_array, ctx)?;
    match_each_integer_ptype!(uncompressed_lens_array.ptype(), |P| {
        Ok(build_views(
            start_buf_index,
            MAX_BUFFER_LEN,
            uncompressed_bytes,
            uncompressed_lens_array.as_slice::<P>(),
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::varbin::VarBinArrayExt;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::fsst_decode_bytes;
    use crate::FSST;
    use crate::FSSTArrayExt;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    fn make_data() -> (VarBinArray, Vec<Option<Vec<u8>>>) {
        const STRING_COUNT: usize = 1000;
        let mut rng = StdRng::seed_from_u64(0);
        let mut strings = Vec::with_capacity(STRING_COUNT);

        for _ in 0..STRING_COUNT {
            if rng.random_bool(0.9) {
                strings.push(None)
            } else {
                // Generate a random string with length around `avg_len`. The number of possible
                // characters within the random string is defined by `unique_chars`.
                let len = 10 * rng.random_range(50..=150) / 100;
                strings.push(Some(
                    (0..len)
                        .map(|_| rng.random_range(b'a'..=b'z') as char)
                        .collect::<String>()
                        .into_bytes(),
                ));
            }
        }

        (
            VarBinArray::from_iter(
                strings
                    .clone()
                    .into_iter()
                    .map(|opt_s| opt_s.map(Vec::into_boxed_slice)),
                DType::Binary(Nullability::Nullable),
            ),
            strings,
        )
    }

    fn make_data_chunked() -> (ChunkedArray, Vec<Option<Vec<u8>>>) {
        let mut ctx = SESSION.create_execution_ctx();
        #[expect(clippy::type_complexity)]
        let (arr_vec, data_vec): (Vec<ArrayRef>, Vec<Vec<Option<Vec<u8>>>>) = (0..10)
            .map(|_| {
                let (array, data) = make_data();
                let array = array.into_array();
                let compressor = fsst_train_compressor(&array, &mut ctx).unwrap();
                (
                    fsst_compress(&array, &compressor, &mut ctx)
                        .unwrap()
                        .into_array(),
                    data,
                )
            })
            .unzip();

        (
            ChunkedArray::from_iter(arr_vec),
            data_vec.into_iter().flatten().collect(),
        )
    }

    #[test]
    fn test_to_canonical() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (chunked_arr, data) = make_data_chunked();

        let mut builder =
            VarBinViewBuilder::with_capacity(chunked_arr.dtype().clone(), chunked_arr.len());
        chunked_arr
            .clone()
            .into_array()
            .append_to_builder(&mut builder, &mut ctx)?;

        {
            let arr = builder.finish_into_canonical(&mut ctx).into_varbinview();
            let mask = arr.validity()?.execute_mask(arr.len(), &mut ctx)?;
            let res1 = (0..arr.len())
                .map(|i| mask.value(i).then(|| arr.bytes_at(i).to_vec()))
                .collect::<Vec<_>>();
            assert_eq!(data, res1);
        };

        {
            let arr2 = chunked_arr
                .as_array()
                .clone()
                .execute::<VarBinViewArray>(&mut ctx)?;
            let mask = arr2.validity()?.execute_mask(arr2.len(), &mut ctx)?;
            let res2 = (0..arr2.len())
                .map(|i| mask.value(i).then(|| arr2.bytes_at(i).to_vec()))
                .collect::<Vec<_>>();
            assert_eq!(data, res2)
        };

        {
            let mut builder =
                VarBinBuilder::<i32>::with_capacity(chunked_arr.dtype().clone(), data.len());
            chunked_arr
                .into_array()
                .append_to_builder(&mut builder, &mut ctx)?;
            let arr = builder.finish_into_varbin();
            let mask = arr.validity()?.execute_mask(arr.len(), &mut ctx)?;
            let actual = (0..arr.len())
                .map(|i| mask.value(i).then(|| arr.bytes_at(i).to_vec()))
                .collect::<Vec<_>>();
            assert_eq!(data, actual);
        }
        Ok(())
    }

    #[test]
    fn test_append_after_in_progress_buffer() -> VortexResult<()> {
        let dtype = DType::Binary(Nullability::NonNullable);
        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), 2);
        builder.append_value(b"long enough!!!");

        let varbin = VarBinArray::from_iter(
            [Some(b"long enough too".to_vec().into_boxed_slice())],
            dtype,
        )
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let fsst_array = fsst_compress(
            &varbin,
            &fsst_train_compressor(&varbin, &mut ctx)?,
            &mut ctx,
        )?
        .into_array();
        fsst_array.append_to_builder(&mut builder, &mut ctx)?;

        let _result = builder.finish_into_varbinview();
        Ok(())
    }

    #[test]
    fn test_rejects_incorrect_uncompressed_lengths() -> VortexResult<()> {
        let input = VarBinViewArray::from_iter_str(["hello"]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = fsst_compress(&input, &fsst_train_compressor(&input, &mut ctx)?, &mut ctx)?;
        let invalid = FSST::try_new(
            encoded.dtype().clone(),
            encoded.symbols().clone(),
            encoded.symbol_lengths().clone(),
            encoded.codes(),
            PrimitiveArray::from_iter([4i32]).into_array(),
            &mut ctx,
        )?;

        assert!(fsst_decode_bytes(invalid.as_view(), &mut ctx).is_err());
        Ok(())
    }
}
