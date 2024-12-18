#![allow(clippy::cast_possible_truncation)]
#![doc(html_logo_url = "/vortex/docs/_static/vortex_spiral_logo.svg")]

//! Read and write Vortex layouts, a serialization of Vortex arrays.
//!
//! A layout is a serialized array which is stored in some linear and contiguous block of
//! memory. Layouts are recursively defined in terms of one of three kinds:
//!
//! 1. The [`FlatLayout`](layouts::FlatLayout). A contiguously serialized array using the Vortex
//!    flatbuffer Batch [`message`](vortex_flatbuffers::message).
//!
//! 2. The [`ColumnarLayout`](layouts::ColumnarLayout). Each column of a
//!    [`StructArray`][vortex_array::array::StructArray] is sequentially laid out at known offsets.
//!    This permits reading a subset of columns in time linear in the number of kept columns.
//!
//! 3. The [`ChunkedLayout`](layouts::ChunkedLayout). Each chunk of a
//!    [`ChunkedArray`](vortex_array::array::ChunkedArray) is sequentially laid out at known
//!    offsets. This permits reading a subset of rows in time linear in the number of kept rows.
//!
//! A layout, alone, is _not_ a standalone Vortex file because layouts are not self-describing. They
//! neither contain a description of the kind of layout (e.g. flat, column of flat, chunked of
//! column of flat) nor a data type ([`DType`](vortex_dtype::DType)). A standalone Vortex file
//! comprises seven sections, the first of which is the serialized array bytes. The interpretation
//! of those bytes, i.e. which particular layout was used, is given in the fourth section: the
//! footer.
//!
//! | Section     | Size               | Description                                                         |
//! | ----------- | ------------------ | ------------------------------------------------------------------- |
//! | Data        | In the Footer.     | The serialized arrays.                                              |
//! | Metadata    | In the Footer.     | A table per column with a row per chunk. Contains statistics.       |
//! | Schema      | In the Postscript. | A serialized data type.                                             |
//! | Footer      | In the Postscript. | A recursive description of the layout including the number of rows. |
//! | Postscript  | 32 bytes           | Two 64-bit offsets pointing at schema and the footer.               |
//! | Version     | 4 bytes            | The file format version.                                            |
//! | Magic bytes | 4 bytes            | The ASCII bytes "VRTX" (86, 82, 84, 88; 0x56525458).                |
//!
//! A Parquet-style file format is realized by using a chunked layout containing column layouts
//! containing chunked layouts containing flat layouts. The outer chunked layout represents row
//! groups. The inner chunked layout represents pages.
//!
//! All the chunks of a chunked layout and all the columns of a column layout need not use the same
//! layout.
//!
//! Anything implementing [`VortexReadAt`](vortex_io::VortexReadAt), for example local files, byte
//! buffers, and [cloud storage](vortex_io::ObjectStoreReadAt), can be used as the "linear and
//! contiguous memory".
//!
//! # Reading
//!
//! Layout reading is implemented by [`VortexFileArrayStream`]. The [`VortexFileArrayStream`] should
//! be constructed by a [`VortexReadBuilder`], which first uses an [InitialRead] to read the footer
//! (schema, layout, postscript, version, and magic bytes). In most cases, these entire footer can
//! be read by a single read of the suffix of the file.
//!
//! A [`VortexFileArrayStream`] internally contains a [`LayoutMessageCache`] which is shared by its
//! layout reader and the layout reader's descendants. The cache permits the reading system to
//! "read" the bytes of a layout multiple times without triggering reads to the underlying storage.
//! For example, the [`VortexFileArrayStream`] reads an array, evaluates the row filter, and then
//! reads the array again with the filter mask.
//!
//! A [`LayoutReader`] then assembles one or more Vortex arrays by reading the serialized data and
//! metadata.
//!
//! # Apache Arrow
//!
//! If you ultimately seek Arrow arrays, [`VortexRecordBatchReader`] converts a
//! [`VortexFileArrayStream`] into a [`RecordBatchReader`](arrow_array::RecordBatchReader).

mod read;
mod write;

mod byte_range;
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
        }
    }
}

pub use forever_constant::*;
pub use read::*;
pub use write::*;
