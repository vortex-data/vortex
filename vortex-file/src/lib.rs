#![allow(clippy::cast_possible_truncation)]
#![doc(html_logo_url = "/vortex/docs/_static/vortex_spiral_logo.svg")]
//! Read and write Vortex layouts, a serialization of Vortex arrays.
//!
//! A layout is a serialized array which is stored in some linear and contiguous block of
//! memory. Layouts are recursive, and there are currently three types:
//!
//! 1. The [`FlatLayout`](vortex_layout::layouts::flat::FlatLayout). A contiguously serialized array of buffers, with a specific in-memory [`Alignment`](vortex_buffer::Alignment).
//!
//! 2. The [`StructLayout`](vortex_layout::layouts::struct_::StructLayout). Each column of a
//!    [`StructArray`][vortex_array::arrays::StructArray] is sequentially laid out at known offsets.
//!    This permits reading a subset of columns in linear time, as well as constant-time random
//!    access to any column.
//!
//! 3. The [`ChunkedLayout`](vortex_layout::layouts::chunked::ChunkedLayout). Each chunk of a
//!    [`ChunkedArray`](vortex_array::arrays::ChunkedArray) is sequentially laid out at known
//!    offsets. Finding the chunks containing row range is an `Nlog(N)` operation of searching the
//!    offsets.
//!
//! 4. The [`ZonedLayout`](vortex_layout::layouts::zoned::ZonedLayout).
//!
//! A layout, alone, is _not_ a standalone Vortex file because layouts are not self-describing. They
//! neither contain a description of the kind of layout (e.g. flat, column of flat, chunked of
//! column of flat) nor a data type ([`DType`](vortex_dtype::DType)).
//!
//! # Reading
//!
//! Vortex files are read using [`VortexOpenOptions`], which can be provided with information about the file's
//! structure to save on IO before the actual data read. Once the file is open and has done the initial IO work to understand its own structure,
//! it can be turned into a stream by calling [`VortexFile::scan`].
//!
//! The file manages IO-oriented work and CPU-oriented work on two different underlying runtimes, which are configurable and pluggable with multiple provided implementations (Tokio, Rayon etc.).
//! It also caches buffers between stages of the scan, saving on duplicate IO. The cache can also be reused between scans of the same file (See [`SegmentCache`](`crate::segments::SegmentCache`)).
//!
//! # File Format
//!
//! Succinctly, the file format specification is as follows:
//!
//! 1. Data is written first, in a form that is describable by a Layout (typically Array IPC Messages).
//!    1. To allow for more efficient IO & pruning, our writer implementation first writes the "data" arrays,
//!       and then writes the "metadata" arrays (i.e., per-column statistics)
//! 2. We write what is collectively referred to as the "Footer", which contains:
//!    1. An optional Schema, which if present is a valid flatbuffer representing a message::Schema
//!    2. The Layout, which is a valid footer::Layout flatbuffer, and describes the physical byte ranges & relationships amongst
//!       the those byte ranges that we wrote in part 1.
//!    3. The Postscript, which is a valid footer::Postscript flatbuffer, containing the absolute start offsets of the Schema & Layout
//!       flatbuffers within the file.
//!    4. The End-of-File marker, which is 8 bytes, and contains the u16 version, u16 postscript length, and 4 magic bytes.
//!
//! ## Illustrated File Format
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
//! Layouts are adaptive, and the writer is free to build arbitrarily complex layouts to suit their
//! goals of locality or parallelism. For example, one may write a column in a Struct Layout with
//! or without chunking, or completely elide statistics to save space or if they are not needed, for
//! example if the metadata is being stored in an external index.
//!
//! Anything implementing [`VortexReadAt`](vortex_io::VortexReadAt), for example local files, byte
//! buffers, and [cloud storage](vortex_io::ObjectStoreReadAt), can be used as the backing store.

mod driver;
mod file;
mod footer;
mod generic;
mod memory;
mod open;
pub mod segments;
mod strategy;
#[cfg(test)]
mod tests;
mod writer;

use std::sync::{Arc, LazyLock};

pub use file::*;
pub use footer::{Footer, SegmentSpec};
pub use forever_constant::*;
pub use generic::*;
pub use memory::*;
pub use open::*;
pub use strategy::*;
use vortex_alp::{ALPEncoding, ALPRDEncoding};
use vortex_array::{ArrayRegistry, EncodingRef};
use vortex_bytebool::ByteBoolEncoding;
use vortex_datetime_parts::DateTimePartsEncoding;
use vortex_decimal_byte_parts::DecimalBytePartsEncoding;
use vortex_dict::DictEncoding;
use vortex_fastlanes::{BitPackedEncoding, DeltaEncoding, FoREncoding};
use vortex_fsst::FSSTEncoding;
pub use vortex_layout::scan;
use vortex_runend::RunEndEncoding;
use vortex_sparse::SparseEncoding;
use vortex_zigzag::ZigZagEncoding;
pub use writer::*;

/// The current version of the Vortex file format
pub const VERSION: u16 = 1;
/// The size of the footer in bytes in Vortex version 1
pub const V1_FOOTER_FBS_SIZE: usize = 32;

/// Constants that will never change (i.e., doing so would break backwards compatibility)
mod forever_constant {
    /// The extension for Vortex files
    pub const VORTEX_FILE_EXTENSION: &str = "vortex";

    /// The maximum length of a Vortex footer in bytes
    pub const MAX_FOOTER_SIZE: u16 = u16::MAX - 8;
    /// The magic bytes for a Vortex file
    pub const MAGIC_BYTES: [u8; 4] = *b"VTXF";
    /// The size of the EOF marker in bytes
    pub const EOF_SIZE: usize = 8;

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
        }
    }
}

/// A default registry containing the built-in Vortex encodings and layouts.
pub static DEFAULT_REGISTRY: LazyLock<Arc<ArrayRegistry>> = LazyLock::new(|| {
    // Register the compressed encodings that Vortex ships with.
    let mut registry = ArrayRegistry::canonical_only();
    registry.register_many([
        EncodingRef::new_ref(ALPEncoding.as_ref()),
        EncodingRef::new_ref(ALPRDEncoding.as_ref()),
        EncodingRef::new_ref(BitPackedEncoding.as_ref()),
        EncodingRef::new_ref(ByteBoolEncoding.as_ref()),
        EncodingRef::new_ref(DateTimePartsEncoding.as_ref()),
        EncodingRef::new_ref(DecimalBytePartsEncoding.as_ref()),
        EncodingRef::new_ref(DeltaEncoding.as_ref()),
        EncodingRef::new_ref(DictEncoding.as_ref()),
        EncodingRef::new_ref(FoREncoding.as_ref()),
        EncodingRef::new_ref(FSSTEncoding.as_ref()),
        EncodingRef::new_ref(RunEndEncoding.as_ref()),
        EncodingRef::new_ref(SparseEncoding.as_ref()),
        EncodingRef::new_ref(ZigZagEncoding.as_ref()),
    ]);
    Arc::new(registry)
});
