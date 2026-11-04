// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exercises the V2 I/O override through DuckDB's SQL scan, in its own test process.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use tempfile::NamedTempFile;
use url::Url;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::buffer::buffer;
use vortex::error::VortexResult;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexReadAt;
use vortex::io::read_at2::FutureReadAt2;
use vortex::io::read_at2::ReadAt2IoService;
use vortex::io::request::IoService;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::io::std_file::FileReadAt;
use vortex::session::VortexSession;
use vortex_duckdb::duckdb::Database;
use vortex_duckdb::reader::VortexScanIoFactory;
use vortex_duckdb::reader::set_scan_io_factory;

struct Factory {
    reader: Arc<dyn VortexReadAt>,
    calls: AtomicUsize,
}

impl VortexScanIoFactory for Factory {
    fn create_io_service(&self, _url: &Url) -> VortexResult<Option<Arc<dyn IoService>>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let reader = Arc::clone(&self.reader);
        Ok(Some(Arc::new(ReadAt2IoService::new(Arc::new(move || {
            Ok(Arc::new(FutureReadAt2::new(Arc::clone(&reader))))
        })))))
    }
}

#[test]
fn v2_uses_completion_io_and_v1_keeps_its_reader() -> anyhow::Result<()> {
    let runtime = CurrentThreadRuntime::new();
    let input = NamedTempFile::with_suffix(".vortex")?;
    runtime.block_on(async {
        let array =
            StructArray::try_from_iter([("value", buffer![1_i64, 2, 3, 4, 5, 6].into_array())])?;
        let mut file = async_fs::File::create(input.path()).await?;
        VortexSession::default()
            .with_handle(runtime.handle())
            .write_options()
            .write(&mut file, array.into_array().to_array_stream())
            .await?;
        VortexResult::Ok(())
    })?;
    let factory = Arc::new(Factory {
        reader: Arc::new(FileReadAt::open(input.path(), runtime.handle())?),
        calls: AtomicUsize::new(0),
    });
    set_scan_io_factory(Some(Arc::clone(&factory) as _));
    let db = Database::open_in_memory()?;
    vortex_duckdb::initialize(&db)?;
    let connection = db.connect()?;
    let query = format!(
        "SELECT CAST(sum(value) AS BIGINT) FROM vortex_scan('{}') WHERE value % 2 = 1",
        input.path().to_string_lossy().replace('\'', "''"),
    );
    for v2 in [false, true] {
        vortex_duckdb::set_scan_v2(v2);
        let result = connection.query(&query)?;
        let mut chunks = result.into_iter();
        let mut chunk = chunks.next().expect("aggregate row");
        assert_eq!(chunk.len(), 1);
        assert!(chunks.next().is_none());
        let vector = chunk.get_vector_mut(0);
        assert!(!vector.row_is_null(0));
        // SAFETY: SQL casts the non-null aggregate to BIGINT, and the result has one row.
        assert_eq!(unsafe { vector.as_slice_mut::<i64>(1) }[0], 9);
        assert_eq!(factory.calls.load(Ordering::Relaxed), usize::from(v2));
    }
    set_scan_io_factory(None);
    vortex_duckdb::set_scan_v2(false);
    Ok(())
}
