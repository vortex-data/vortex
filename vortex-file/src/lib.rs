#![doc(html_logo_url = "/vortex/docs/_static/vortex_spiral_logo.svg")]
//! Read and write Vortex layouts, a serialization of Vortex arrays.
//!
//! A layout is a serialized array which is stored in some linear and contiguous block of
//! memory. Layouts are recursively defined in terms of one of three kinds:
//!
//! 1. The [flat layout][layouts::FlatLayout]. A contiguously serialized array using the [Vortex
//!    flatbuffer Batch message][vortex_flatbuffers::message].
//!
//! 2. The [columnar layout][layouts::ColumnarLayout]. Each column of a
//!    [StructArray][vortex_array::array::StructArray] is sequentially laid out at known
//!    offsets. This permits reading a subset of columns in time linear in the number of kept
//!    columns.
//!
//! 3. The [chunked layout][layouts::ChunkedLayout]. Each chunk of a
//!    [ChunkedArray][vortex_array::array::ChunkedArray] is sequentially laid out at known
//!    offsets. This permits reading a subset of rows in time linear in the number of kept rows.
//!
//! A layout, alone, is _not_ a standalone Vortex file because layouts are not self-describing. They
//! neither contain a description of the kind of layout (e.g. flat, column of flat, chunked of
//! column of flat) nor a [data type][vortex_dtype::DType]. A standalone Vortex file comprises seven
//! sections, the first of which is the serialized array bytes. The interpretation of those bytes,
//! i.e. which particular layout was used, is given in the fourth section: the footer.
//!
//! <table>
//! <thead>
//! <tr>
//! <th>Section</th>
//! <th>Size</th>
//! <th>Description</th>
//! </tr>
//! </thead>
//! <tr>
//! <td>
//! Data
//! </td>
//! <td>
//! In the Footer.
//! </td>
//! <td>
//! The serialized arrays.
//! </td>
//! </tr><tr>
//! <td>
//! Metadata
//! </td>
//! <td>
//! In the Footer.
//! </td>
//! <td>
//! A table per column with a row per chunk. Contains statistics.
//! </td>
//! </tr><tr>
//! <td>
//! Schema
//! </td>
//! <td>
//! In the Postscript.
//! </td>
//! <td>
//! A serialized data type.
//! </td>
//! </tr><tr>
//! <td>
//! Footer
//! </td>
//! <td>
//! In the Postscript.
//! </td>
//! <td>
//! A recursive description of the layout including the number of rows.
//! </td>
//! </tr><tr>
//! <td>
//! Postscript
//! </td>
//! <td>
//! 32 bytes
//! </td>
//! <td>
//! Two 64-bit offsets pointing at schema and the footer.
//! </td>
//! </tr><tr>
//! <td>
//! Version
//! </td>
//! <td>
//! 4 bytes
//! </td>
//! <td>
//! The file format version.
//! </td>
//! </tr><tr>
//! <td>
//! Magic bytes
//! </td>
//! <td>
//! 4 bytes
//! </td>
//! <td>
//! The ASCII bytes "VRTX" (86, 82, 84, 88; 0x56525458).
//! </td>
//! </tr>
//! </table>
//!
//! A Parquet-style file format is realized by using a chunked layout containing column layouts
//! containing chunked layouts containing flat layouts. The outer chunked layout represents row
//! groups. The inner chunked layout represents pages.
//!
//! All the chunks of a chunked layout and all the columns of a column layout need not use the same
//! layout.
//!
//! Anything implementing [VortexReadAt][vortex_io::VortexReadAt], for example local files, byte
//! buffers, and [cloud storage][vortex_io::ObjectStoreReadAt], can be used as the "linear and
//! contiguous memory".
//!
//! # Reading
//!
//! Layout reading is implemented by [VortexFileArrayStream]. The VortexFileArrayStream should be
//! constructed by a [VortexReadBuilder], which first uses an [InitialRead] to read the footer (schema,
//! layout, postscript, version, and magic bytes). In most cases, these entire footer can be read by
//! a single read of the suffix of the file.
//!
//! A VortexFileArrayStream internally contains a [LayoutMessageCache] which is shared by its layout
//! reader and the layout reader's descendents. The cache permits the reading system to "read" the
//! bytes of a layout multiple times without triggering reads to the underlying storage. For
//! example, the VortexFileArrayStream reads an array, evaluates the row filter, and then reads the
//! array again with the filter mask.
//!
//! [`read_layout_from_initial`] produces a [LayoutReader] which assembles one or more Vortex arrays
//! by reading the serialized data and metadata.
//!
//! # Apache Arrow
//!
//! If you ultimately seek Arrow arrays, [VortexRecordBatchReader] converts a [VortexFileArrayStream]
//! into a RecordBatchReader.

pub mod chunked_reader;
mod dtype_reader;

pub use dtype_reader::*;

mod read;
mod write;

mod pruning;
#[cfg(test)]
mod tests;

/// The current version of the Vortex file format
pub const VERSION: u16 = 1;
/// The size of the footer in bytes in Vortex version 1
pub const V1_FOOTER_FBS_SIZE: usize = 32;

/// Constants that will never change (i.e., doing so would break backwards compatibility)
mod forever_constant {
    use super::*;

    /// The extension for Vortex files
    pub const VORTEX_FILE_EXTENSION: &str = "vortex";

    /// The maximum length of a Vortex footer in bytes
    pub const MAX_FOOTER_SIZE: u16 = u16::MAX - 8;
    /// The magic bytes for a Vortex file
    pub const MAGIC_BYTES: [u8; 4] = *b"VTXF";
    /// The size of the EOF marker in bytes
    pub const EOF_SIZE: usize = 8;

    /// The layout ID for a flat layout
    pub const FLAT_LAYOUT_ID: LayoutId = LayoutId(1);
    /// The layout ID for a chunked layout
    pub const CHUNKED_LAYOUT_ID: LayoutId = LayoutId(2);
    /// The layout ID for a column layout
    pub const COLUMNAR_LAYOUT_ID: LayoutId = LayoutId(3);
    /// The layout ID for an inline schema layout
    pub const INLINE_SCHEMA_LAYOUT_ID: LayoutId = LayoutId(4);

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn never_change_these_constants() {
            assert_eq!(V1_FOOTER_FBS_SIZE, 32);
            assert_eq!(MAX_FOOTER_SIZE, 65527);
            assert_eq!(MAGIC_BYTES, *b"VTXF");
            assert_eq!(EOF_SIZE, 8);
            assert_eq!(FLAT_LAYOUT_ID, LayoutId(1));
            assert_eq!(CHUNKED_LAYOUT_ID, LayoutId(2));
            assert_eq!(COLUMNAR_LAYOUT_ID, LayoutId(3));
            assert_eq!(INLINE_SCHEMA_LAYOUT_ID, LayoutId(4));
        }
    }
}

pub use forever_constant::*;
pub use read::*;
pub use write::*;

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod test {
    use std::sync::Arc;

    use bytes::Bytes;
    use futures_executor::block_on;
    use futures_util::{pin_mut, StreamExt, TryStreamExt};
    use itertools::Itertools;
    use vortex_array::array::{ChunkedArray, PrimitiveArray, PrimitiveEncoding};
    use vortex_array::encoding::EncodingVTable;
    use vortex_array::stream::ArrayStreamExt;
    use vortex_array::{ArrayDType, Context, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_io::VortexBufReader;
    use vortex_ipc::stream_reader::StreamArrayReader;
    use vortex_ipc::stream_writer::StreamArrayWriter;

    fn write_ipc<A: IntoArrayData>(array: A) -> Vec<u8> {
        block_on(async {
            StreamArrayWriter::new(vec![])
                .write_array(array.into_array())
                .await
                .unwrap()
                .into_inner()
        })
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_empty_index() -> VortexResult<()> {
        let data = PrimitiveArray::from((0i32..3_000_000).collect_vec());
        let buffer = write_ipc(data);

        let indices = PrimitiveArray::from(vec![1, 2, 10]).into_array();

        let ctx = Arc::new(Context::default());
        let stream_reader =
            StreamArrayReader::try_new(VortexBufReader::new(Bytes::from(buffer)), ctx)
                .await
                .unwrap()
                .load_dtype()
                .await
                .unwrap();
        let reader = stream_reader.into_array_stream();

        let result_iter = reader.take_rows(indices)?;
        pin_mut!(result_iter);

        let _result = block_on(async { result_iter.next().await.unwrap().unwrap() });
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_write_read_chunked() -> VortexResult<()> {
        let indices = PrimitiveArray::from(vec![
            10u32, 11, 12, 13, 100_000, 2_999_999, 2_999_999, 3_000_000,
        ])
        .into_array();

        // NB: the order is reversed here to ensure we aren't grabbing indexes instead of values
        let data = PrimitiveArray::from((0i32..3_000_000).rev().collect_vec()).into_array();
        let data2 =
            PrimitiveArray::from((3_000_000i32..6_000_000).rev().collect_vec()).into_array();
        let chunked = ChunkedArray::try_new(vec![data.clone(), data2], data.dtype().clone())?;
        let buffer = write_ipc(chunked);
        let buffer = Buffer::from(buffer);

        let ctx = Arc::new(Context::default());
        let stream_reader = StreamArrayReader::try_new(VortexBufReader::new(buffer), ctx)
            .await
            .unwrap()
            .load_dtype()
            .await
            .unwrap();

        let take_iter = stream_reader.into_array_stream().take_rows(indices)?;
        pin_mut!(take_iter);

        let next = block_on(async { take_iter.try_next().await })?.expect("Expected a chunk");
        assert_eq!(next.encoding().id(), PrimitiveEncoding.id());

        assert_eq!(
            next.into_primitive().unwrap().maybe_null_slice::<i32>(),
            vec![2999989, 2999988, 2999987, 2999986, 2899999, 0, 0]
        );
        assert_eq!(
            block_on(async { take_iter.try_next().await })?
                .expect("Expected a chunk")
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            vec![5999999]
        );

        Ok(())
    }
}
