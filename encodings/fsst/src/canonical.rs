use arrow_array::builder::make_view;
use vortex_array::array::{BinaryView, VarBinArray, VarBinViewArray};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Canonical, IntoArrayVariant};
use vortex_buffer::{BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::{FSSTArray, FSSTEncoding};

impl CanonicalVTable<FSSTArray> for FSSTEncoding {
    fn into_canonical(&self, array: FSSTArray) -> VortexResult<Canonical> {
        array.with_decompressor(|decompressor| {
            // FSSTArray has two child arrays:
            //
            //  1. A VarBinArray, which holds the string heap of the compressed codes.
            //  2. An uncompressed_lengths primitive array, storing the length of each original
            //     string element.
            //
            // To speed up canonicalization, we can decompress the entire string-heap in a single
            // call. We then turn our uncompressed_lengths into an offsets buffer
            // necessary for a VarBinViewArray and construct the canonical array.

            let bytes = VarBinArray::try_from(array.codes())?.sliced_bytes();

            let uncompressed_lens_array = array.uncompressed_lengths().into_primitive()?;

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
                let len = decompressor
                    .decompress_into(bytes.as_slice(), uncompressed_bytes.spare_capacity_mut());
                uncompressed_bytes.set_len(len);
            };

            // Directly create the binary views.
            let mut views = BufferMut::<BinaryView>::with_capacity(uncompressed_lens_array.len());

            match_each_integer_ptype!(uncompressed_lens_array.ptype(), |$P| {
                let mut offset = 0;
                for len in uncompressed_lens_array.as_slice::<$P>() {
                    let len = *len as usize;
                    let view = make_view(
                        &uncompressed_bytes[offset..][..len],
                        0u32,
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
                array.dtype().clone(),
                array.validity(),
            )
            .map(Canonical::VarBinView)
        })
    }
}
