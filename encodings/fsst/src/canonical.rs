use arrow_array::builder::make_view;
use fsst::Decompressor;
use vortex_array::arrays::{BinaryView, VarBinArray, VarBinViewArray};
use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayCanonicalImpl, ArrayExt, Canonical, IntoArray, ToCanonical};
use vortex_buffer::{BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::FSSTArray;

impl ArrayCanonicalImpl for FSSTArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        self.with_decompressor(|decompressor| {
            fsst_into_varbin_view(decompressor, self, 0).map(Canonical::VarBinView)
        })
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            return builder.extend_from_array(&self.to_canonical()?.into_array());
        };
        let view = self.with_decompressor(|decompressor| {
            fsst_into_varbin_view(decompressor, self, builder.completed_block_count())
        })?;

        builder.push_buffer_and_adjusted_views(
            view.buffers().iter().cloned(),
            view.views().iter().cloned(),
            self.validity_mask()?,
        );
        Ok(())
    }
}

// Decompresses a fsst encoded array into a varbinview, a block_offset can be passed if the decoding
// if happening as part of the larger view and is used to set the block_offset in each view.
fn fsst_into_varbin_view(
    decompressor: Decompressor,
    fsst_array: &FSSTArray,
    block_offset: usize,
) -> VortexResult<VarBinViewArray> {
    // FSSTArray has two child arrays:
    //
    //  1. A VarBinArray, which holds the string heap of the compressed codes.
    //  2. An uncompressed_lengths primitive array, storing the length of each original
    //     string element.
    //
    // To speed up canonicalization, we can decompress the entire string-heap in a single
    // call. We then turn our uncompressed_lengths into an offsets buffer
    // necessary for a VarBinViewArray and construct the canonical array.
    let bytes = fsst_array.codes().as_::<VarBinArray>().sliced_bytes();

    let uncompressed_lens_array = fsst_array.uncompressed_lengths().to_primitive()?;

    // Decompres the full dataset.
    #[allow(clippy::cast_possible_truncation)]
    let total_size: usize = match_each_integer_ptype!(uncompressed_lens_array.ptype(), |$P| {
       uncompressed_lens_array.as_slice::<$P>().iter().map(|x| *x as usize).sum()
    });

    // Bulk-decompress the entire array.
    let mut uncompressed_bytes = ByteBufferMut::with_capacity(total_size + 7);
    // SAFETY: uncompressed bytes is large enough to contain all data + the 7 additional bytes
    //  of padding required for vectorized decompression. See the docstring for `decompress_into`
    //  for more details.
    unsafe {
        let len =
            decompressor.decompress_into(bytes.as_slice(), uncompressed_bytes.spare_capacity_mut());
        uncompressed_bytes.set_len(len);
    };

    let block_offset = u32::try_from(block_offset)?;

    // Directly create the binary views.
    let mut views = BufferMut::<BinaryView>::with_capacity(uncompressed_lens_array.len());

    match_each_integer_ptype!(uncompressed_lens_array.ptype(), |$P| {
        let mut offset = 0;
        for len in uncompressed_lens_array.as_slice::<$P>() {
            let len = *len as usize;
            let view = make_view(
                &uncompressed_bytes[offset..][..len],
                block_offset,
                offset as u32,
            );
            // SAFETY: we reserved the right capacity beforehand
            unsafe { views.push_unchecked(view.into()) };
            offset += len;
        }
    });

    let views = views.freeze();
    let uncompressed_bytes_array = ByteBuffer::from(uncompressed_bytes);

    VarBinViewArray::try_new(
        views,
        vec![uncompressed_bytes_array],
        fsst_array.dtype().clone(),
        Validity::copy_from_array(fsst_array)?,
    )
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::{ChunkedArray, VarBinArray};
    use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
    use vortex_array::{Array, ArrayRef, ToCanonical};
    use vortex_dtype::{DType, Nullability};

    use crate::{fsst_compress, fsst_train_compressor};

    fn make_data() -> (ArrayRef, Vec<Option<Vec<u8>>>) {
        const STRING_COUNT: usize = 1000;
        let mut rng = StdRng::seed_from_u64(0);
        let mut strings = Vec::with_capacity(STRING_COUNT);

        for _ in 0..STRING_COUNT {
            if rng.gen_bool(0.9) {
                strings.push(None)
            } else {
                // Generate a random string with length around `avg_len`. The number of possible
                // characters within the random string is defined by `unique_chars`.
                let len = 10 * rng.gen_range(50..=150) / 100;
                strings.push(Some(
                    (0..len)
                        .map(|_| rng.gen_range(b'a'..=b'z') as char)
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
            )
            .into_array(),
            strings,
        )
    }

    fn make_data_chunked() -> (ChunkedArray, Vec<Option<Vec<u8>>>) {
        #[allow(clippy::type_complexity)]
        let (arr_vec, data_vec): (Vec<ArrayRef>, Vec<Vec<Option<Vec<u8>>>>) = (0..10)
            .map(|_| {
                let (array, data) = make_data();
                let compressor = fsst_train_compressor(&array).unwrap();
                (
                    fsst_compress(&array, &compressor).unwrap().into_array(),
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
    fn test_to_canonical() {
        let (chunked_arr, data) = make_data_chunked();

        let mut builder =
            VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), chunked_arr.len());
        chunked_arr.append_to_builder(&mut builder).unwrap();

        {
            let arr = builder.finish().to_varbinview().unwrap();
            let res1 = arr
                .with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(data, res1);
        };

        {
            let arr2 = chunked_arr.to_varbinview().unwrap();
            let res2 = arr2
                .with_iterator(|iter| iter.map(|b| b.map(|v| v.to_vec())).collect::<Vec<_>>())
                .unwrap();
            assert_eq!(data, res2)
        };
    }
}
