use std::path::PathBuf;
use std::sync::Arc;

use bitvec::macros::internal::funty::Fundamental;
use crossbeam_queue::SegQueue;
use vortex::dtype::FieldNames;
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{ExprRef, and, and_collect, lit, root, select};
use vortex::file::{VortexFile, VortexOpenOptions};

use crate::convert::{try_from_bound_expression, try_from_table_filter};
use crate::duckdb::{
    BindInput, BindResult, DataChunk, Expression, LogicalType, TableFunction, TableInitInput,
};
use crate::exporter::ArrayIteratorExporter;

pub struct VortexBindData {
    _first_file: VortexFile,
    filter_exprs: Vec<ExprRef>,
    file_paths: Vec<PathBuf>,
    column_names: Vec<String>,
    column_types: Vec<LogicalType>,
}

impl Clone for VortexBindData {
    /// `VortexBindData` is cloned in case of multiple scan nodes.
    fn clone(&self) -> Self {
        Self {
            _first_file: self._first_file.clone(),
            // filter_expr don't need to be cloned as they are consumed once in `init_global`.
            filter_exprs: vec![],
            file_paths: self.file_paths.clone(),
            column_names: self.column_names.clone(),
            column_types: self.column_types.clone(),
        }
    }
}

impl std::fmt::Debug for VortexBindData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexBindData")
            .field("file_paths", &self.file_paths)
            .field("column_names", &self.column_names)
            .field("column_types", &self.column_types)
            .field("filter_expr", &self.filter_exprs)
            .finish()
    }
}

pub struct VortexGlobalData {
    file_paths: SegQueue<PathBuf>,
    _is_first_file_processed: std::sync::atomic::AtomicBool,
    filter_expr: ExprRef,
    projection_expr: ExprRef,
}

pub struct VortexLocalData {
    exporter: Option<ArrayIteratorExporter>,
}

#[derive(Debug)]
pub struct VortexTableFunction;

/// Extracts the schema from a Vortex file.
fn extract_schema_from_vortex_file(
    file: &VortexFile,
) -> VortexResult<(Vec<String>, Vec<LogicalType>)> {
    let dtype = file.dtype();

    // For now, we assume the top-level type to be a struct.
    let struct_dtype = dtype
        .as_struct()
        .ok_or_else(|| vortex_err!("Vortex file must contain a struct array at the top level"))?;

    let mut column_names = Vec::new();
    let mut column_types = Vec::new();

    for (field_name, field_dtype) in struct_dtype.names().iter().zip(struct_dtype.fields()) {
        let logical_type = LogicalType::try_from(&field_dtype)
            .map_err(|e| vortex_err!("Failed to convert field '{}' type: {}", field_name, e))?;

        column_names.push(field_name.to_string());
        column_types.push(logical_type);
    }

    Ok((column_names, column_types))
}

/// Creates a projection expression based on the table initialization input.
fn create_projection_expr(init: &TableInitInput<VortexTableFunction>) -> ExprRef {
    let projection_ids = init.projection_ids().unwrap_or(&[]);
    let column_ids = init.column_ids();

    let projected_ids = projection_ids.iter().map(|p| column_ids[p.as_usize()]);
    select(
        projected_ids
            .map(|idx| {
                init.bind_data()
                    .column_names
                    .get(idx.as_usize())
                    .vortex_expect("prune idx in column names")
            })
            .map(|s| Arc::from(s.clone()))
            .collect::<FieldNames>(),
        root(),
    )
}

/// Creates a table filter expression from the table filter set.
fn create_table_filter_expr(
    init: &TableInitInput<VortexTableFunction>,
) -> VortexResult<Option<ExprRef>> {
    init
        .table_filter_set()
        .and_then(|filter| {
            filter
                .into_iter()
                .map(|(idx, ex)| {
                    let name = init
                        .bind_data()
                        .column_names
                        .get(idx.as_usize())
                        .vortex_expect("exists");
                    try_from_table_filter(&ex, name)
                })
                .reduce(|l, r| l?.zip(r?).map(|(l, r)| Ok(and(l, r))).transpose())
        })
        .transpose()?
        .flatten();
}

/// Creates a SegQueue populated with file paths from bind data.
fn create_file_paths_queue(bind_data: &VortexBindData) -> SegQueue<PathBuf> {
    let file_paths = SegQueue::new();
    for path in bind_data.file_paths.iter() {
        file_paths.push(path.clone());
    }
    file_paths
}

impl TableFunction for VortexTableFunction {
    type BindData = VortexBindData;
    type GlobalState = VortexGlobalData;
    type LocalState = VortexLocalData;

    const PROJECTION_PUSHDOWN: bool = true;
    const FILTER_PUSHDOWN: bool = true;
    const FILTER_PRUNE: bool = true;

    /// Input parameter types of the `vortex_scan` table function.
    ///
    // `vortex_scan` takes a single file glob parameter.
    fn parameters() -> Vec<LogicalType> {
        vec![LogicalType::varchar()]
    }

    fn bind(input: &BindInput, result: &mut BindResult) -> VortexResult<Self::BindData> {
        let file_glob_string = input
            .get_parameter(0)
            .ok_or_else(|| vortex_err!("Missing file glob parameter"))?;

        let paths = match glob::glob(file_glob_string.as_string().to_str()?) {
            Ok(paths) => paths,
            Err(e) => vortex_bail!("Failed to glob files: {}", e),
        };

        let mut file_paths = Vec::new();
        for path in paths {
            match path {
                Ok(path) => file_paths.push(path),
                Err(e) => vortex_bail!("Failed to glob files: {}", e),
            }
        }

        let file_path: String = file_paths[0].to_string_lossy().into_owned();
        let first_file = VortexOpenOptions::file()
            .open_blocking(&file_path)
            .map_err(|e| vortex_err!("Failed to open Vortex file: {}", e))?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (name, logical_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(name, logical_type);
        }

        Ok(VortexBindData {
            file_paths,
            _first_file: first_file,
            column_names,
            column_types,
            filter_exprs: vec![],
        })
    }

    fn scan(
        _bind_data: &Self::BindData,
        local_state: &mut Self::LocalState,
        global_state: &mut Self::GlobalState,
        chunk: &mut DataChunk,
    ) -> VortexResult<()> {
        if local_state.exporter.is_none() {
            // Retrieve a file path from the shared lock-free queue.
            if let Some(file_path) = global_state.file_paths.pop() {
                let file = VortexOpenOptions::file()
                    .open_blocking(&file_path)
                    .map_err(|e| vortex_err!("Failed to open Vortex file: {}", e))?;

                let array_iter = file
                    .scan()?
                    .with_projection(global_state.projection_expr.clone())
                    .with_filter(global_state.filter_expr.clone())
                    .into_array_iter()
                    .map_err(|e| vortex_err!("Failed to create array iterator: {}", e))?;

                local_state.exporter = Some(ArrayIteratorExporter::new(Box::new(array_iter)));
            } else {
                // If the exporter is None and there are no more files to process, signal that the scan finished.
                chunk.set_len(0);
                return Ok(());
            }
        }

        let Some(ref mut exporter) = local_state.exporter else {
            vortex_bail!("ArrayIteratorExporter is not set")
        };

        let is_data_left_to_scan = !exporter
            .export(chunk)
            .map_err(|e| vortex_err!("Failed to export data: {}", e))?;

        if !is_data_left_to_scan {
            local_state.exporter = None;
        }

        Ok(())
    }

    fn init_global(init_input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        let bind_data = init_input.bind_data();
        let file_paths = create_file_paths_queue(bind_data);
        let projection_expr = create_projection_expr(init_input);
        let filter_expr = create_table_filter_expr(init_input)?;

        let complex_filter_expr = and_collect(init_input.bind_data().filter_exprs.clone());
        let filter_expr = complex_filter_expr
            .into_iter()
            .chain(filter_expr)
            .reduce(and)
            .unwrap_or_else(|| lit(true));

        Ok(VortexGlobalData {
            file_paths,
            _is_first_file_processed: std::sync::atomic::AtomicBool::new(false),
            filter_expr,
            projection_expr,
        })
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        _global: &mut Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        Ok(VortexLocalData { exporter: None })
    }

    fn pushdown_complex_filter(
        bind_data: &mut Self::BindData,
        expr: &Expression,
    ) -> VortexResult<bool> {
        println!("expr {}", expr);
        let expr = try_from_bound_expression(expr)?;
        bind_data.filter_exprs.push(expr);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use duckdb::Connection;
    use tempfile::NamedTempFile;
    use vortex::IntoArray;
    use vortex::arrays::{BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray};
    use vortex::file::VortexWriteOptions;
    use vortex::scalar::Scalar;
    use vortex::validity::Validity;

    use super::*;
    use crate::duckdb::Database;

    fn database_connection() -> Connection {
        let db = Database::open_in_memory().unwrap();
        let connection = db.connect().unwrap();
        connection
            .register_table_function::<VortexTableFunction>(c"vortex_scan")
            .unwrap();
        connection
            .register_table_function::<VortexTableFunction>(c"read_vortex")
            .unwrap();
        unsafe { Connection::open_from_raw(db.as_ptr().cast()) }.unwrap()
    }

    fn create_temp_file() -> NamedTempFile {
        NamedTempFile::new().unwrap()
    }

    async fn write_single_column_vortex_file(
        field_name: &str,
        array: impl IntoArray,
    ) -> NamedTempFile {
        write_vortex_file([(field_name, array)].into_iter()).await
    }

    async fn write_vortex_file(
        iter: impl Iterator<Item = (impl AsRef<str>, impl IntoArray)>,
    ) -> NamedTempFile {
        let temp_file_path = create_temp_file();

        let struct_array = StructArray::try_from_iter(iter).unwrap();
        let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
        VortexWriteOptions::default()
            .write(file, struct_array.to_array_stream())
            .await
            .unwrap();

        temp_file_path
    }

    fn scan_vortex_file_single_row<T>(tmp_file: NamedTempFile, query: &str, col_idx: usize) -> T
    where
        T: duckdb::types::FromSql,
    {
        let conn = database_connection();
        conn.prepare(query)
            .unwrap()
            .query_row([tmp_file.path().to_string_lossy()], |row| row.get(col_idx))
            .unwrap()
    }

    fn scan_vortex_file<T>(
        tmp_file: NamedTempFile,
        query: &str,
        col_idx: usize,
    ) -> Result<Vec<T>, String>
    where
        T: duckdb::types::FromSql,
    {
        let conn = database_connection();
        conn.prepare(query)
            .unwrap()
            .query_and_then([tmp_file.path().to_string_lossy()], |row| row.get(col_idx))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())
    }

    async fn write_vortex_file_to_dir(
        dir: &std::path::Path,
        field_name: &str,
        array: impl IntoArray,
    ) -> NamedTempFile {
        let struct_array = StructArray::from_fields(&[(field_name, array.into_array())]).unwrap();
        let temp_file_path = tempfile::Builder::new()
            .suffix(".vortex")
            .tempfile_in(dir)
            .unwrap();

        let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
        VortexWriteOptions::default()
            .write(file, struct_array.to_array_stream())
            .await
            .unwrap();

        temp_file_path
    }

    #[test]
    fn test_scan_function_registration() {
        let conn = database_connection();
        let result: String = conn
            .prepare(
                "SELECT function_name FROM duckdb_functions() WHERE function_name = 'vortex_scan'",
            )
            .unwrap()
            .query_row([], |row| row.get(0))
            .unwrap();
        assert_eq!(&result, "vortex_scan");
    }

    #[tokio::test]
    async fn test_vortex_scan_strings() {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        let file = write_single_column_vortex_file("strings", strings).await;
        let result: String = scan_vortex_file_single_row(
            file,
            "SELECT string_agg(strings, ',') FROM vortex_scan(?)",
            0,
        );
        assert_eq!(result, "Hello,Hi,Hey");
    }

    #[tokio::test]
    async fn test_vortex_scan_integers() {
        let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
        let file = write_single_column_vortex_file("number", numbers).await;
        let sum: i64 =
            scan_vortex_file_single_row(file, "SELECT SUM(number) FROM vortex_scan(?)", 0);
        assert_eq!(sum, 138);
    }

    #[tokio::test]
    async fn test_vortex_scan_floats() {
        let values = PrimitiveArray::from_iter([1.5f64, -2.5, 0.0, 42.42]);
        let file = write_single_column_vortex_file("value", values).await;
        let count: i64 = scan_vortex_file_single_row(
            file,
            "SELECT COUNT(*) FROM vortex_scan(?) WHERE value > 0",
            0,
        );
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_vortex_scan_constant() {
        let constant = ConstantArray::new(Scalar::from(42i32), 100);
        let file = write_single_column_vortex_file("constant", constant).await;
        let value: i32 =
            scan_vortex_file_single_row(file, "SELECT constant FROM vortex_scan(?) LIMIT 1", 0);
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn test_vortex_scan_booleans() {
        let flags = vec![true, false, true, true, false];
        let flags_array = BoolArray::new(flags.into(), Validity::NonNullable);
        let file = write_single_column_vortex_file("flag", flags_array).await;
        let true_count: i64 = scan_vortex_file_single_row(
            file,
            "SELECT COUNT(*) FROM vortex_scan(?) WHERE flag = true",
            0,
        );
        assert_eq!(true_count, 3);
    }

    #[tokio::test]
    async fn test_vortex_multi_column() {
        let f1 = BoolArray::new(
            vec![true, false, true, true, false].into(),
            Validity::NonNullable,
        )
        .to_array();
        let f2 = (0..5).collect::<PrimitiveArray>().to_array();
        let f3 = (100..105).collect::<PrimitiveArray>().to_array();
        let file = write_vortex_file([("f1", f1), ("f2", f2), ("f3", f3)].into_iter()).await;

        let result: Vec<i32> = scan_vortex_file(
            file,
            "SELECT f2 FROM vortex_scan(?) WHERE f1 = true and f2 >= 2",
            0,
        )
        .unwrap();

        assert_eq!(result, vec![2, 3]);
    }

    #[tokio::test]
    async fn test_vortex_scan_multiple_files() {
        let tempdir = tempfile::tempdir().unwrap();

        let _file1 = write_vortex_file_to_dir(
            tempdir.path(),
            "numbers",
            PrimitiveArray::from_iter([1i32, 2, 3]),
        )
        .await;

        let _file2 = write_vortex_file_to_dir(
            tempdir.path(),
            "numbers",
            PrimitiveArray::from_iter([4i32, 5, 6]),
        )
        .await;

        // Create glob pattern to match all .vortex files in the temp directory.
        let glob_pattern = format!("{}/*.vortex", tempdir.path().display());

        // Scan both Vortex files.
        let conn = database_connection();
        let total_sum: i64 = conn
            .prepare("SELECT SUM(numbers) FROM vortex_scan(?)")
            .unwrap()
            .query_row([&glob_pattern], |row| row.get(0))
            .unwrap();

        assert_eq!(total_sum, 21);
    }

    #[tokio::test]
    async fn test_tpch_bench() {
        let conn = database_connection();
        let table = conn.execute(r#"
        create view hits as select * from read_vortex('/Users/joeisaacs/git/spiraldb/vortex/bench-vortex/data/clickbench_partitioned/vortex-file-compressed/hits_0.vortex');
        "#, []).unwrap();
        println!("assert table {}", table);

        let queries = r#"
    SELECT COUNT(*) FROM hits;
    SELECT COUNT(*) FROM hits WHERE "AdvEngineID" <> 0;
    SELECT SUM("AdvEngineID"), COUNT(*), AVG("ResolutionWidth") FROM hits;
    SELECT AVG("UserID") FROM hits;
    SELECT COUNT(DISTINCT "UserID") FROM hits;
    SELECT COUNT(DISTINCT "SearchPhrase") FROM hits;
    SELECT MIN("EventDate"), MAX("EventDate") FROM hits;
    SELECT "AdvEngineID", COUNT(*) FROM hits WHERE "AdvEngineID" <> 0 GROUP BY "AdvEngineID" ORDER BY COUNT(*) DESC;
    SELECT "RegionID", COUNT(DISTINCT "UserID") AS u FROM hits GROUP BY "RegionID" ORDER BY u DESC LIMIT 10;
    SELECT "RegionID", SUM("AdvEngineID"), COUNT(*) AS c, AVG("ResolutionWidth"), COUNT(DISTINCT "UserID") FROM hits GROUP BY "RegionID" ORDER BY c DESC LIMIT 10;
    SELECT "MobilePhoneModel", COUNT(DISTINCT "UserID") AS u FROM hits WHERE "MobilePhoneModel" <> '' GROUP BY "MobilePhoneModel" ORDER BY u DESC LIMIT 10;
    SELECT "MobilePhone", "MobilePhoneModel", COUNT(DISTINCT "UserID") AS u FROM hits WHERE "MobilePhoneModel" <> '' GROUP BY "MobilePhone", "MobilePhoneModel" ORDER BY u DESC LIMIT 10;
    SELECT "SearchPhrase", COUNT(*) AS c FROM hits WHERE "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY c DESC LIMIT 10;
    SELECT "SearchPhrase", COUNT(DISTINCT "UserID") AS u FROM hits WHERE "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY u DESC LIMIT 10;
    SELECT "SearchEngineID", "SearchPhrase", COUNT(*) AS c FROM hits WHERE "SearchPhrase" <> '' GROUP BY "SearchEngineID", "SearchPhrase" ORDER BY c DESC LIMIT 10;
    SELECT "UserID", COUNT(*) FROM hits GROUP BY "UserID" ORDER BY COUNT(*) DESC LIMIT 10;
    SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", "SearchPhrase" ORDER BY COUNT(*) DESC LIMIT 10;
    SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", "SearchPhrase" LIMIT 10;
    SELECT "UserID", extract(minute FROM "EventTime") AS m, "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", m, "SearchPhrase" ORDER BY COUNT(*) DESC LIMIT 10;
    SELECT "UserID" FROM hits WHERE "UserID" = 435090932899640449;
    SELECT COUNT(*) FROM hits WHERE "URL" LIKE '%google%';
    SELECT "SearchPhrase", MIN("URL"), COUNT(*) AS c FROM hits WHERE "URL" LIKE '%google%' AND "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY c DESC LIMIT 10;
    SELECT "SearchPhrase", MIN("URL"), MIN("Title"), COUNT(*) AS c, COUNT(DISTINCT "UserID") FROM hits WHERE "Title" LIKE '%Google%' AND "URL" NOT LIKE '%.google.%' AND "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY c DESC LIMIT 10;
    SELECT * FROM hits WHERE "URL" LIKE '%google%' ORDER BY "EventTime" LIMIT 10;
    SELECT "SearchPhrase" FROM hits WHERE "SearchPhrase" <> '' ORDER BY "EventTime" LIMIT 10;
    SELECT "SearchPhrase" FROM hits WHERE "SearchPhrase" <> '' ORDER BY "SearchPhrase" LIMIT 10;
    SELECT "SearchPhrase" FROM hits WHERE "SearchPhrase" <> '' ORDER BY "EventTime", "SearchPhrase" LIMIT 10;
    SELECT "CounterID", AVG(length("URL")) AS l, COUNT(*) AS c FROM hits WHERE "URL" <> '' GROUP BY "CounterID" HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;
    SELECT REGEXP_REPLACE("Referer", '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(length("Referer")) AS l, COUNT(*) AS c, MIN("Referer") FROM hits WHERE "Referer" <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;
    SELECT SUM("ResolutionWidth"), SUM("ResolutionWidth" + 1), SUM("ResolutionWidth" + 2), SUM("ResolutionWidth" + 3), SUM("ResolutionWidth" + 4), SUM("ResolutionWidth" + 5), SUM("ResolutionWidth" + 6), SUM("ResolutionWidth" + 7), SUM("ResolutionWidth" + 8), SUM("ResolutionWidth" + 9), SUM("ResolutionWidth" + 10), SUM("ResolutionWidth" + 11), SUM("ResolutionWidth" + 12), SUM("ResolutionWidth" + 13), SUM("ResolutionWidth" + 14), SUM("ResolutionWidth" + 15), SUM("ResolutionWidth" + 16), SUM("ResolutionWidth" + 17), SUM("ResolutionWidth" + 18), SUM("ResolutionWidth" + 19), SUM("ResolutionWidth" + 20), SUM("ResolutionWidth" + 21), SUM("ResolutionWidth" + 22), SUM("ResolutionWidth" + 23), SUM("ResolutionWidth" + 24), SUM("ResolutionWidth" + 25), SUM("ResolutionWidth" + 26), SUM("ResolutionWidth" + 27), SUM("ResolutionWidth" + 28), SUM("ResolutionWidth" + 29), SUM("ResolutionWidth" + 30), SUM("ResolutionWidth" + 31), SUM("ResolutionWidth" + 32), SUM("ResolutionWidth" + 33), SUM("ResolutionWidth" + 34), SUM("ResolutionWidth" + 35), SUM("ResolutionWidth" + 36), SUM("ResolutionWidth" + 37), SUM("ResolutionWidth" + 38), SUM("ResolutionWidth" + 39), SUM("ResolutionWidth" + 40), SUM("ResolutionWidth" + 41), SUM("ResolutionWidth" + 42), SUM("ResolutionWidth" + 43), SUM("ResolutionWidth" + 44), SUM("ResolutionWidth" + 45), SUM("ResolutionWidth" + 46), SUM("ResolutionWidth" + 47), SUM("ResolutionWidth" + 48), SUM("ResolutionWidth" + 49), SUM("ResolutionWidth" + 50), SUM("ResolutionWidth" + 51), SUM("ResolutionWidth" + 52), SUM("ResolutionWidth" + 53), SUM("ResolutionWidth" + 54), SUM("ResolutionWidth" + 55), SUM("ResolutionWidth" + 56), SUM("ResolutionWidth" + 57), SUM("ResolutionWidth" + 58), SUM("ResolutionWidth" + 59), SUM("ResolutionWidth" + 60), SUM("ResolutionWidth" + 61), SUM("ResolutionWidth" + 62), SUM("ResolutionWidth" + 63), SUM("ResolutionWidth" + 64), SUM("ResolutionWidth" + 65), SUM("ResolutionWidth" + 66), SUM("ResolutionWidth" + 67), SUM("ResolutionWidth" + 68), SUM("ResolutionWidth" + 69), SUM("ResolutionWidth" + 70), SUM("ResolutionWidth" + 71), SUM("ResolutionWidth" + 72), SUM("ResolutionWidth" + 73), SUM("ResolutionWidth" + 74), SUM("ResolutionWidth" + 75), SUM("ResolutionWidth" + 76), SUM("ResolutionWidth" + 77), SUM("ResolutionWidth" + 78), SUM("ResolutionWidth" + 79), SUM("ResolutionWidth" + 80), SUM("ResolutionWidth" + 81), SUM("ResolutionWidth" + 82), SUM("ResolutionWidth" + 83), SUM("ResolutionWidth" + 84), SUM("ResolutionWidth" + 85), SUM("ResolutionWidth" + 86), SUM("ResolutionWidth" + 87), SUM("ResolutionWidth" + 88), SUM("ResolutionWidth" + 89) FROM hits;
    SELECT "SearchEngineID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"), AVG("ResolutionWidth") FROM hits WHERE "SearchPhrase" <> '' GROUP BY "SearchEngineID", "ClientIP" ORDER BY c DESC LIMIT 10;
    SELECT "WatchID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"), AVG("ResolutionWidth") FROM hits WHERE "SearchPhrase" <> '' GROUP BY "WatchID", "ClientIP" ORDER BY c DESC LIMIT 10;
    SELECT "WatchID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"), AVG("ResolutionWidth") FROM hits GROUP BY "WatchID", "ClientIP" ORDER BY c DESC LIMIT 10;
    SELECT "URL", COUNT(*) AS c FROM hits GROUP BY "URL" ORDER BY c DESC LIMIT 10;
    SELECT 1, "URL", COUNT(*) AS c FROM hits GROUP BY 1, "URL" ORDER BY c DESC LIMIT 10;
    SELECT "ClientIP", "ClientIP" - 1, "ClientIP" - 2, "ClientIP" - 3, COUNT(*) AS c FROM hits GROUP BY "ClientIP", "ClientIP" - 1, "ClientIP" - 2, "ClientIP" - 3 ORDER BY c DESC LIMIT 10;
    SELECT "URL", COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31' AND "DontCountHits" = 0 AND "IsRefresh" = 0 AND "URL" <> '' GROUP BY "URL" ORDER BY PageViews DESC LIMIT 10;
    SELECT "Title", COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31' AND "DontCountHits" = 0 AND "IsRefresh" = 0 AND "Title" <> '' GROUP BY "Title" ORDER BY PageViews DESC LIMIT 10;
    SELECT "URL", COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31' AND "IsRefresh" = 0 AND "IsLink" <> 0 AND "IsDownload" = 0 GROUP BY "URL" ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;
    SELECT "TraficSourceID", "SearchEngineID", "AdvEngineID", CASE WHEN ("SearchEngineID" = 0 AND "AdvEngineID" = 0) THEN "Referer" ELSE '' END AS Src, "URL" AS Dst, COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31' AND "IsRefresh" = 0 GROUP BY "TraficSourceID", "SearchEngineID", "AdvEngineID", Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;
    SELECT "URLHash", "EventDate", COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31' AND "IsRefresh" = 0 AND "TraficSourceID" IN (-1, 6) AND "RefererHash" = 3594120000172545465 GROUP BY "URLHash", "EventDate" ORDER BY PageViews DESC LIMIT 10 OFFSET 100;
    SELECT "WindowClientWidth", "WindowClientHeight", COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-01' AND "EventDate" <= '2013-07-31' AND "IsRefresh" = 0 AND "DontCountHits" = 0 AND "URLHash" = 2868770270353813622 GROUP BY "WindowClientWidth", "WindowClientHeight" ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;
    SELECT DATE_TRUNC('minute', "EventTime") AS M, COUNT(*) AS PageViews FROM hits WHERE "CounterID" = 62 AND "EventDate" >= '2013-07-14' AND "EventDate" <= '2013-07-15' AND "IsRefresh" = 0 AND "DontCountHits" = 0 GROUP BY DATE_TRUNC('minute', "EventTime") ORDER BY DATE_TRUNC('minute', M) LIMIT 10 OFFSET 1000;
                "#;

        for (id, q) in queries
            .split(";")
            .filter(|s| !s.is_empty())
            .enumerate()
            .map(|(id, s)| (id, s.to_string() + ";"))
        {
            println!("running query {id}");
            println!("{}", conn.execute(q.as_str(), []).unwrap());
        }
    }
}
