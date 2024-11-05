//! Read and write Vortex layouts, a serialization of Vortex arrays.
//!
//! A layout is a serialized array which is stored in some linear and contiguous block of
//! memory. Layouts are recursively defined in terms of one of three kinds:
//!
//! 1. The [flat layout][layouts::FlatLayoutSpec]. A contiguously serialized array using the [Vortex
//!    flatbuffer Batch message][vortex_flatbuffers::message].
//!
//! 2. The [column layout][layouts::ColumnLayoutSpec]. Each column of a
//!    [StructArray][vortex_array::array::StructArray] is sequentially laid out at known
//!    offsets. This permits reading a subset of columns in time linear in the number of kept
//!    columns.
//!
//! 3. The [chunked layout][layouts::ChunkedLayoutSpec]. Each chunk of a
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
//! Anything implementing [VortexReadAt][crate::io::VortexReadAt], for example local files, byte
//! buffers, and [cloud storage][crate::io::ObjectStoreReadAt], can be used as the "linear and
//! contiguous memory".
//!
//! # Reading
//!
//! Layout reading is implemented by [LayoutBatchStream]. The LayoutBatchStream uses a
//! [LayoutDescriptorReader] to read the schema, footer, postscript, version, and magic bytes into a
//! [LayoutDescriptor]. In most cases, these five sections can be read by a single read of the
//! suffix of the file.
//!
//! A LayoutBatchStream internally contains a [LayoutMessageCache] which is shared by its layout
//! reader and the layout reader's descendents. The cache permits the reading system to "read" the
//! bytes of a layout multiple times without triggering reads to the underlying storage. For
//! example, the LayoutBatchStream reads an array, evaluates the row filter, and then reads the
//! array again with the filter mask.
//!
//! [`LayoutDescriptor::layout`] produces a [LayoutReader] which assembles one or more Vortex arrays
//! by reading the serialized data and metadata.
//!
//! # Apache Arrow
//!
//! If you ultimately seek Arrow arrays, [VortexRecordBatchReader] converts a [LayoutBatchStream]
//! into a RecordBatchReader.

mod read;
mod write;

mod pruning;
#[cfg(test)]
mod tests;

pub const VERSION: u16 = 1;
pub const MAGIC_BYTES: [u8; 4] = *b"VRTX";
// Size of serialized Postscript Flatbuffer
pub const FOOTER_POSTSCRIPT_SIZE: usize = 32;
pub const EOF_SIZE: usize = 8;
pub const FLAT_LAYOUT_ID: LayoutId = LayoutId(1);
pub const CHUNKED_LAYOUT_ID: LayoutId = LayoutId(2);
pub const COLUMN_LAYOUT_ID: LayoutId = LayoutId(3);
pub const INLINE_SCHEMA_LAYOUT_ID: LayoutId = LayoutId(4);

pub use read::*;
pub use write::*;
