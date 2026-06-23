// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![doc(html_logo_url = "/vortex/docs/_static/vortex_spiral_logo.svg")]
//! Read and write Vortex layouts, a serialization of Vortex arrays.
//!
//! A Vortex file stores a root [`Layout`](vortex_layout::Layout), the byte segments referenced by
//! that layout, optional file-level statistics, and enough footer metadata to deserialize the tree.
//! Layouts are recursive, so a file may organize data by row groups, columns, dictionaries,
//! statistics, or other writer-chosen structures without changing the logical dtype seen by
//! readers.
//!
//! This crate owns the file reader/writer APIs. The byte-level format lives in the Sphinx docs:
//! <https://docs.vortex.dev/specs/file-format.html>.
//!
//! # Reading
//!
//! Use [`OpenOptionsSessionExt::open_options`] to create [`VortexOpenOptions`] from a session.
//! Opening reads or accepts a footer, builds a segment source, and returns [`VortexFile`]. Scans are
//! configured from [`VortexFile::scan`] with projection/filter expressions, row ranges,
//! [`Selection`](vortex_scan::selection::Selection), split strategy, and concurrency settings from
//! `vortex-layout`.
//!
//! Supplying known metadata can reduce open-time IO:
//!
//! - [`VortexOpenOptions::with_file_size`] avoids a size request.
//! - [`VortexOpenOptions::with_dtype`] is required for files written without an embedded dtype.
//! - [`VortexOpenOptions::with_footer`] can open a file without reading footer bytes.
//! - [`VortexOpenOptions::with_segment_cache`] reuses segment buffers across scans.
//!
//! # Writing
//!
//! Use [`WriteOptionsSessionExt::write_options`] or [`VortexWriteOptions::new`] to write an
//! [`ArrayStream`](vortex_array::stream::ArrayStream). The default [`WriteStrategyBuilder`]
//! repartitions rows, builds statistics layouts, dictionary-encodes suitable columns, compresses
//! chunks with the BtrBlocks-style compressor, and writes flat leaf layouts. Advanced users can
//! replace the whole strategy or override individual fields.
//!
//! # Footer Deserialization
//!
//! [`FooterDeserializer`] supports incremental footer reads. It returns [`DeserializeStep`] values
//! when it needs more bytes or a file size, and returns [`Footer`] once all required footer segments
//! are available. [`VortexOpenOptions`] drives this state machine for ordinary file opens.

mod counting;
mod file;
mod footer;
pub mod multi;
mod open;
mod pruning;
mod read;
/// Segment sources, caches, and sinks used by file readers and writers.
pub mod segments;
mod strategy;
#[cfg(test)]
mod tests;
/// Compatibility readers for newer file-statistics layout behavior.
pub mod v2;
mod writer;

pub use file::*;
pub use footer::*;
pub use forever_constant::*;
pub use open::*;
pub use strategy::*;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::session::ArraySessionExt;
use vortex_pco::Pco;
use vortex_session::VortexSession;
pub use writer::*;

/// The current version of the Vortex file format
pub const VERSION: u16 = 1;
/// The size of the footer in bytes in Vortex version 1
pub const V1_FOOTER_FBS_SIZE: usize = 32;

/// Constants that will never change (i.e., doing so would break backwards compatibility)
mod forever_constant {
    /// The extension for Vortex files
    pub const VORTEX_FILE_EXTENSION: &str = "vortex";

    /// The maximum length of a Vortex postscript in bytes
    pub const MAX_POSTSCRIPT_SIZE: u16 = u16::MAX - 8;
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
            assert_eq!(MAX_POSTSCRIPT_SIZE, 65527);
            assert_eq!(MAGIC_BYTES, *b"VTXF");
            assert_eq!(EOF_SIZE, 8);
        }
    }
}

/// Register the default encodings use in Vortex files with the provided session.
///
/// NOTE: this function will be changed in the future to encapsulate logic for using different
/// Vortex "Editions" that may support different sets of encodings.
pub fn register_default_encodings(session: &VortexSession) {
    vortex_bytebool::initialize(session);
    vortex_fsst::initialize(session);
    #[cfg(feature = "unstable_encodings")]
    vortex_onpair::initialize(session);
    vortex_zigzag::initialize(session);

    {
        let arrays = session.arrays();
        arrays.register(Pco);
        #[cfg(feature = "zstd")]
        arrays.register(vortex_zstd::Zstd);
        #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
        arrays.register(vortex_zstd::ZstdBuffers);
        if use_experimental_patches() {
            arrays.register(Patched);
        }
    }

    vortex_alp::initialize(session);
    vortex_datetime_parts::initialize(session);
    vortex_decimal_byte_parts::initialize(session);
    vortex_fastlanes::initialize(session);
    vortex_runend::initialize(session);
    vortex_sequence::initialize(session);
    vortex_sparse::initialize(session);

    #[cfg(feature = "unstable_encodings")]
    vortex_tensor::initialize(session);
}

#[cfg(test)]
mod default_encoding_tests {
    use vortex_array::VTable as _;
    use vortex_array::array_session;
    use vortex_array::arrays::Filter;
    use vortex_array::optimizer::kernels::ArrayKernelsExt as _;
    use vortex_array::session::ArraySessionExt as _;
    use vortex_fsst::FSST;

    use crate::register_default_encodings;

    #[test]
    fn register_default_encodings_registers_external_execute_parent_kernels() {
        let session = array_session();

        assert!(session.arrays().registry().find(&FSST.id()).is_none());
        assert!(!session.kernels().has_execute_parent(Filter.id(), FSST.id()));

        register_default_encodings(&session);

        assert!(session.arrays().registry().find(&FSST.id()).is_some());
        assert!(session.kernels().has_execute_parent(Filter.id(), FSST.id()));
    }
}
