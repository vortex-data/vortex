// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex data sources.
//!
//! The hand-written C ABI exposed `vx_data_source` (an `arc_dyn_wrapper!` over
//! `Arc<dyn DataSource>`) with the free functions `vx_data_source_new` (open from a comma- or
//! glob-delimited set of paths via `vx_data_source_options`), `vx_data_source_new_buffer` (open
//! from an in-memory Vortex file), `vx_data_source_dtype`, and `vx_data_source_get_row_count`.
//!
//! Under Diplomat the opaque type is `VxDataSource`, the path/buffer entry points become named
//! constructors, schema access is a getter, and the row-count estimate is returned by value as
//! `VxEstimate` (defined in the `scan` bridge). The destructor is generated automatically (no
//! `vx_data_source_free`).
//!
//! ## Callback / pluggable-source caveat (Diplomat limitation)
//!
//! In the core Rust API, `DataSource` is a trait, so a host application could in principle supply
//! its own implementation (for example, a source backed by a host-language read callback that
//! Rust would invoke to pull bytes on demand). Diplomat is **unidirectional** — it generates
//! foreign bindings that call *into* Rust, but does not natively generate the reverse
//! (foreign → Rust) callback machinery, so a fully caller-pluggable `DataSource` cannot be
//! expressed through the bridge. We therefore model `VxDataSource` as an opaque with a fixed set
//! of constructors over the sources Vortex itself provides (file/glob paths, URLs, and in-memory
//! buffers). A custom callback-driven source would require either a hand-written `extern "C"` shim
//! that adapts a C function-pointer callback into a `DataSource` impl (kept outside the Diplomat
//! bridge), or Diplomat's experimental callback support once it stabilises.

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    use bytes::Bytes;
    use vortex::buffer::ByteBuffer;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::multi::MultiFileDataSource;
    use vortex::layout::scan::multi::MultiLayoutDataSource;
    use vortex::scan::DataSource;
    use vortex::scan::DataSourceRef;

    use crate::RUNTIME;
    use crate::dtype::ffi::VxDType;
    use crate::error::ffi::VortexFfiError;
    use crate::scan::ffi::VxEstimate;
    use crate::session::ffi::VxSession;
    use vortex::io::runtime::BlockingRuntime;

    /// A reference to one or more (possibly remote) Vortex data locations.
    ///
    /// Constructing a data source opens the first matched location eagerly to read the schema;
    /// all other IO is deferred until a scan is requested. A single data source may be scanned
    /// multiple times. Internally an `Arc<dyn DataSource>`, mirroring the C
    /// `vx_data_source` opaque (an `arc_dyn_wrapper!`).
    #[diplomat::opaque]
    pub struct VxDataSource(pub(crate) DataSourceRef);

    impl VxDataSource {
        /// Open a data source from one or more file paths.
        ///
        /// `paths` may be a glob such as `*.vortex`, or a comma-delimited list such as
        /// `file1.vortex,../file2.vortex`. The first matched file is opened eagerly to read the
        /// schema. Replaces the C `vx_data_source_new` plus its `vx_data_source_options` struct;
        /// the `*const c_char` paths field becomes a `&str`.
        #[diplomat::attr(auto, named_constructor = "from_path")]
        pub fn from_path(
            session: &VxSession,
            paths: &str,
        ) -> Result<Box<VxDataSource>, Box<VortexFfiError>> {
            let mut data_source = MultiFileDataSource::new(session.inner().clone());
            for glob in paths.split(',') {
                data_source = data_source.with_glob(glob, None);
            }
            let data_source = RUNTIME
                .block_on(async move { data_source.build().await })
                .map_err(Box::<VortexFfiError>::from)?;
            Ok(Box::new(VxDataSource(
                Arc::new(data_source) as DataSourceRef
            )))
        }

        /// Open a data source from a URL.
        ///
        /// A convenience over [`Self::from_path`] for a single (possibly remote) location, kept
        /// as a distinct named constructor so the host-language API reads clearly. The C ABI
        /// folded URLs into the same comma/glob `paths` string.
        #[diplomat::attr(auto, named_constructor = "from_url")]
        pub fn from_url(
            session: &VxSession,
            url: &str,
        ) -> Result<Box<VxDataSource>, Box<VortexFfiError>> {
            Self::from_path(session, url)
        }

        /// Open a data source from a single in-memory Vortex file.
        ///
        /// The bytes are copied into an owned buffer, so the caller need not keep the input alive
        /// (unlike the C `vx_data_source_new_buffer`, which borrowed the bytes via
        /// `'static`). The `(buffer, buffer_len)` C pair becomes a byte slice.
        #[diplomat::attr(auto, named_constructor = "from_buffer")]
        pub fn from_buffer(
            session: &VxSession,
            buffer: &[u8],
        ) -> Result<Box<VxDataSource>, Box<VortexFfiError>> {
            let len = buffer.len() as u64;
            let bytes = ByteBuffer::from(Bytes::copy_from_slice(buffer));
            let session = session.inner();
            let file = session
                .open_options()
                .open_buffer(bytes)
                .map_err(Box::<VortexFfiError>::from)?;
            let ds = MultiLayoutDataSource::new_with_first(
                file.layout_reader().map_err(Box::<VortexFfiError>::from)?,
                Vec::new(),
                vec![Some(len)],
                session.clone(),
            );
            Ok(Box::new(VxDataSource(Arc::new(ds) as DataSourceRef)))
        }

        /// The schema of this data source.
        ///
        /// Replaces `vx_data_source_dtype`. The returned dtype is an owned handle (Diplomat
        /// cannot express the C ABI's "borrowed, lives as long as the source" pointer), so it is
        /// cheaply cloned from the source's schema.
        #[diplomat::attr(auto, getter)]
        pub fn dtype(&self) -> Box<VxDType> {
            Box::new(VxDType(Arc::new(self.0.dtype().clone())))
        }

        /// This data source's row-count estimate.
        ///
        /// Replaces `vx_data_source_get_row_count`, which wrote into an out-parameter; here the
        /// `VxEstimate` is returned by value.
        #[diplomat::attr(auto, getter)]
        pub fn row_count(&self) -> VxEstimate {
            VxEstimate::from_precision(self.0.row_count())
        }
    }
}

impl ffi::VxDataSource {
    /// Borrow the underlying `Arc<dyn DataSource>`.
    ///
    /// Used by the `scan` bridge to request a scan from this source.
    pub(crate) fn inner(&self) -> &vortex::scan::DataSourceRef {
        &self.0
    }
}
