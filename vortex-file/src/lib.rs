#![allow(clippy::cast_possible_truncation)]
#![doc(html_logo_url = "/vortex/docs/_static/vortex_spiral_logo.svg")]

//! Read and write Vortex layouts, a serialization of Vortex arrays.
//!
//! A layout is a serialized array which is stored in some linear and contiguous block of
//! memory. Layouts are recursively defined in terms of one of three kinds:
//!
//! 1. The [`FlatLayout`](vortex_layout::layouts::flat::FlatLayout). A contiguously serialized array of buffers, with a specific in-memory [`Alignment`](vortex_buffer::Alignment).
//!
//! 2. The [`StructLayout`](vortex_layout::layouts::struct_::StructLayout). Each column of a
//!    [`StructArray`][vortex_array::array::StructArray] is sequentially laid out at known offsets.
//!    This permits reading a subset of columns in time linear in the number of kept columns.
//!
//! 3. The [`ChunkedLayout`](vortex_layout::layouts::chunked::ChunkedLayout). Each chunk of a
//!    [`ChunkedArray`](vortex_array::array::ChunkedArray) is sequentially laid out at known
//!    offsets. This permits reading a subset of rows in time linear in the number of kept rows.
//!
//! A layout, alone, is _not_ a standalone Vortex file because layouts are not self-describing. They
//! neither contain a description of the kind of layout (e.g. flat, column of flat, chunked of
//! column of flat) nor a data type ([`DType`](vortex_dtype::DType)).
//!
//! # Reading
//!
//! Reading is implemented by [`VortexFile`]. It's "opened" by [`VortexOpenOptions`], which can be provided with information about's the file's
//! structure to save on IO before the actual data read. Once the file is open and has done the initial IO work to understand its own structure,
//! it can be turned into a stream by calling [`VortexFile::scan`] with a [`Scan`], which defines filtering and projection on the file.
//!
//! The file manages IO-oriented work and CPU-oriented work on two different underlying runtimes, which are configurable and pluggable with multiple provided implementations (Tokio, Rayon etc.).
//! It also caches buffers between stages of the scan, saving on duplicate IO. The cache can also be reused between scans of the same file (See [`SegmentCache`](`crate::segments::SegmentCache`)).
//!
//! # File Format
//!
//! Succinctly, the file format specification is as follows:
//!
//! 1. Data is written first, in a form that is describable by a Layout (typically Array IPC Messages).
//!     a. To allow for more efficient IO & pruning, our writer implementation first writes the "data" arrays,
//!        and then writes the "metadata" arrays (i.e., per-column statistics)
//! 2. We write what is collectively referred to as the "Footer", which contains:
//!     a. An optional Schema, which if present is a valid flatbuffer representing a message::Schema
//!     b. The Layout, which is a valid footer::Layout flatbuffer, and describes the physical byte ranges & relationships amongst
//!        the those byte ranges that we wrote in part 1.
//!     c. The Postscript, which is a valid footer::Postscript flatbuffer, containing the absolute start offsets of the Schema & Layout
//!        flatbuffers within the file.
//!     d. The End-of-File marker, which is 8 bytes, and contains the u16 version, u16 postscript length, and 4 magic bytes.
//!
//! ## Reified File Format
//! ```text
//! ┌────────────────────────────┐
//! │                            │
//! │            Data            │
//! │    (Array IPC Messages)    │
//! │                            │
//! ├────────────────────────────┤
//! │                            │
//! │   Per-Column Statistics    │
//! │                            │
//! ├────────────────────────────┤
//! │                            │
//! │     Schema Flatbuffer      │
//! │                            │
//! ├────────────────────────────┤
//! │                            │
//! │     Layout Flatbuffer      │
//! │                            │
//! ├────────────────────────────┤
//! │                            │
//! │    Postscript Flatbuffer   │
//! │  (Schema & Layout Offsets) │
//! │                            │
//! ├────────────────────────────┤
//! │     8-byte End of File     │
//! │(Version, Postscript Length,│
//! │       Magic Bytes)         │
//! └────────────────────────────┘
//! ```
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
//! # Apache Arrow
//!
//! If you ultimately seek Arrow arrays, [`VortexRecordBatchReader`][`crate::read::VortexRecordBatchReader`] converts an open
//! Vortex file into a [`RecordBatchReader`](arrow_array::RecordBatchReader).

mod exec;
mod file;
mod footer;
pub mod io;
mod open;
pub mod read;
pub mod segments;
#[cfg(test)]
mod tests;
mod writer;

pub use file::*;
pub use footer::{FileLayout, Segment};
pub use forever_constant::*;
pub use open::*;
pub use writer::*;

/// The current version of the Vortex file format
pub const VERSION: u16 = 1;
/// The size of the footer in bytes in Vortex version 1
pub const V1_FOOTER_FBS_SIZE: usize = 32;

/// Constants that will never change (i.e., doing so would break backwards compatibility)
mod forever_constant {
    use vortex_layout::LayoutId;

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
        use crate::*;

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
