use std::path::PathBuf;

use bitvec::macros::internal::funty::Fundamental;
use crossbeam_queue::SegQueue;
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{ExprRef, and, and_collect, lit};
use vortex::file::{VortexFile, VortexOpenOptions};

use crate::convert::{try_from_bound_expression, try_from_table_filter};
use crate::duckdb::{
    BindInput, BindResult, DataChunk, Expression, LogicalType, TableFunction, TableInitInput,
};
use crate::exporter::ArrayIteratorExporter;

#[derive(Clone)]
pub struct VortexBindData {
    first_file: VortexFile,
    filter_exprs: Vec<ExprRef>,
    file_paths: Vec<PathBuf>,
    column_names: Vec<String>,
    _column_types: Vec<LogicalType>,
}

impl std::fmt::Debug for VortexBindData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexBindData")
            .field("file_paths", &self.file_paths)
            .field("column_names", &self.column_names)
            .finish()
    }
}

impl PartialEq for VortexBindData {
    fn eq(&self, other: &Self) -> bool {
        self.file_paths == other.file_paths && self.column_names == other.column_names
    }
}

pub struct VortexGlobalData {
    file_paths: SegQueue<PathBuf>,
    _is_first_file_processed: std::sync::atomic::AtomicBool,
    filter_expr: ExprRef,
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

impl TableFunction for VortexTableFunction {
    type BindData = VortexBindData;
    type GlobalState = VortexGlobalData;
    type LocalState = VortexLocalData;

    const FILTER_PUSHDOWN: bool = true;

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

        let file_path: String = file_glob_string.as_string().to_string_lossy().into_owned();
        let first_file = VortexOpenOptions::file()
            .open_blocking(&file_path)
            .map_err(|e| vortex_err!("Failed to open Vortex file: {}", e))?;

        let (column_names, column_types) = extract_schema_from_vortex_file(&first_file)?;

        // Add result columns based on the extracted schema.
        for (name, logical_type) in column_names.iter().zip(&column_types) {
            result.add_result_column(name, logical_type);
        }

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

        Ok(VortexBindData {
            file_paths,
            first_file,
            column_names,
            _column_types: column_types,
            filter_exprs: vec![],
        })
    }

    fn scan(
        bind_data: &Self::BindData,
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

        let is_data_left_to_scan = exporter
            .export(chunk)
            .map_err(|e| vortex_err!("Failed to export data: {}", e))?;

        if !is_data_left_to_scan {
            local_state.exporter = None;
        }

        Ok(())
    }

    fn init_global(init: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        let bind_data = init.bind_data();
        let file_paths = SegQueue::new();

        // Skip the first file path, as the file is opened during bind.
        for path in bind_data.file_paths.iter() {
            file_paths.push(path.clone());
        }

        let complex_filter = and_collect(init.bind_data().filter_exprs.clone());

        let filter = init
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
                    .reduce(|l, r| Ok(and(l?, r?)))
            })
            .transpose()?;

        let filter_expr = complex_filter
            .into_iter()
            .chain(filter)
            .reduce(and)
            .unwrap_or_else(|| lit(true));

        Ok(VortexGlobalData {
            file_paths,
            _is_first_file_processed: std::sync::atomic::AtomicBool::new(false),
            filter_expr,
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
        unsafe { Connection::open_from_raw(db.as_ptr().cast()) }.unwrap()
    }

    fn create_temp_file() -> NamedTempFile {
        NamedTempFile::new().unwrap()
    }

    async fn write_vortex_file(field_name: &str, array: impl IntoArray) -> NamedTempFile {
        let temp_file_path = create_temp_file();

        let struct_array = StructArray::from_fields(&[(field_name, array.into_array())]).unwrap();
        let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
        VortexWriteOptions::default()
            .write(file, struct_array.to_array_stream())
            .await
            .unwrap();

        temp_file_path
    }

    fn scan_vortex_file<T>(tmp_file: NamedTempFile, query: &str) -> T
    where
        T: duckdb::types::FromSql,
    {
        let conn = database_connection();
        conn.prepare(query)
            .unwrap()
            .query_row([tmp_file.path().to_string_lossy()], |row| row.get(0))
            .unwrap()
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
    async fn test_vortex_bind_data_equality() {
        let temp_file =
            write_vortex_file("test_col", PrimitiveArray::from_iter([1i32, 2, 3])).await;

        let vortex_file1 = VortexOpenOptions::file()
            .open_blocking(temp_file.path())
            .unwrap();

        let vortex_file2 = VortexOpenOptions::file()
            .open_blocking(temp_file.path())
            .unwrap();

        let bind_data1 = VortexBindData {
            first_file: vortex_file1,
            file_paths: vec![temp_file.path().to_owned()],
            column_names: vec!["test_col".to_string()],
            _column_types: vec![],
            filter_exprs: vec![],
        };

        let bind_data2 = VortexBindData {
            first_file: vortex_file2,
            file_paths: vec![temp_file.path().to_owned()],
            column_names: vec!["test_col".to_string()],
            _column_types: vec![],
            filter_exprs: vec![],
        };

        // Compares file_paths and column_names.
        assert_eq!(bind_data1, bind_data2);
    }

    #[tokio::test]
    async fn test_vortex_bind_data_path_inequality() {
        let temp_file1 =
            write_vortex_file("test_col", PrimitiveArray::from_iter([1i32, 2, 3])).await;

        let temp_file2 =
            write_vortex_file("test_col", PrimitiveArray::from_iter([1i32, 2, 3])).await;

        let vortex_file1 = VortexOpenOptions::file()
            .open_blocking(temp_file1.path())
            .unwrap();

        let vortex_file2 = VortexOpenOptions::file()
            .open_blocking(temp_file2.path())
            .unwrap();

        let bind_data1 = VortexBindData {
            first_file: vortex_file1,
            file_paths: vec![temp_file1.path().to_owned()],
            column_names: vec!["test_col".to_string()],
            _column_types: vec![],
            filter_exprs: vec![],
        };

        let bind_data2 = VortexBindData {
            first_file: vortex_file2,
            file_paths: vec![temp_file2.path().to_owned()],
            column_names: vec!["test_col".to_string()],
            _column_types: vec![],
            filter_exprs: vec![],
        };

        // Compares file_paths and column_names.
        assert_ne!(bind_data1, bind_data2);
    }

    #[tokio::test]
    async fn test_vortex_scan_strings() {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        let file = write_vortex_file("strings", strings).await;
        let result: String =
            scan_vortex_file(file, "SELECT string_agg(strings, ',') FROM vortex_scan(?)");
        assert_eq!(result, "Hello,Hi,Hey");
    }

    #[tokio::test]
    async fn test_vortex_scan_integers() {
        let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
        let file = write_vortex_file("number", numbers).await;
        let sum: i64 = scan_vortex_file(file, "SELECT SUM(number) FROM vortex_scan(?)");
        assert_eq!(sum, 138);
    }

    #[tokio::test]
    async fn test_vortex_scan_floats() {
        let values = PrimitiveArray::from_iter([1.5f64, -2.5, 0.0, 42.42]);
        let file = write_vortex_file("value", values).await;
        let count: i64 =
            scan_vortex_file(file, "SELECT COUNT(*) FROM vortex_scan(?) WHERE value > 0");
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_vortex_scan_constant() {
        let constant = ConstantArray::new(Scalar::from(42i32), 100);
        let file = write_vortex_file("constant", constant).await;
        let value: i32 = scan_vortex_file(file, "SELECT constant FROM vortex_scan(?) LIMIT 1");
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn test_vortex_scan_booleans() {
        let flags = vec![true, false, true, true, false];
        let flags_array = BoolArray::new(flags.into(), Validity::NonNullable);
        let file = write_vortex_file("flag", flags_array).await;
        let true_count: i64 = scan_vortex_file(
            file,
            "SELECT COUNT(*) FROM vortex_scan(?) WHERE flag = true",
        );
        assert_eq!(true_count, 3);
    }
}
