// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains tests for the `vortex_scan` table function.

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;
    use tempfile::NamedTempFile;
    use vortex::IntoArray;
    use vortex::arrays::{BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray};
    use vortex::file::VortexWriteOptions;
    use vortex::scalar::Scalar;
    use vortex::validity::Validity;

    use crate::duckdb::{Connection, Database, QueryResultCell};

    fn database_connection() -> Connection {
        let db = Database::open_in_memory().unwrap();
        let connection = db.connect().unwrap();
        crate::register_table_functions(&connection).unwrap();
        connection
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
        T: TryFrom<QueryResultCell, Error = vortex::error::VortexError>,
    {
        let conn = database_connection();
        let file_path = tmp_file.path().to_string_lossy();
        let formatted_query = query.replace('?', &format!("'{file_path}'"));

        let result = conn.query(&formatted_query).unwrap();
        result.get::<T>(col_idx, 0).unwrap()
    }

    fn scan_vortex_file<T>(tmp_file: NamedTempFile, query: &str, col_idx: usize) -> Result<Vec<T>>
    where
        T: TryFrom<QueryResultCell, Error = vortex::error::VortexError>,
    {
        let conn = database_connection();
        let file_path = tmp_file.path().to_string_lossy();
        let formatted_query = query.replace('?', &format!("'{file_path}'"));

        let result = conn.query(&formatted_query)?;

        let mut values = Vec::new();
        for row_idx in 0..result.row_count()? {
            let value = result.get::<T>(col_idx, row_idx)?;
            values.push(value);
        }

        Ok(values)
    }

    async fn write_vortex_file_to_dir(
        dir: &Path,
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
        let result = conn
            .query(
                "SELECT function_name FROM duckdb_functions() WHERE function_name = 'vortex_scan'",
            )
            .unwrap();
        assert_eq!(&result.get::<String>(0, 0).unwrap(), "vortex_scan");
    }

    #[test]
    fn test_vortex_scan_strings() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
            write_single_column_vortex_file("strings", strings).await
        });

        let result: String = scan_vortex_file_single_row(
            file,
            "SELECT string_agg(strings, ',') FROM vortex_scan(?)",
            0,
        );

        assert_eq!(result, "Hello,Hi,Hey");
    }

    #[test]
    fn test_vortex_scan_strings_contains() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
            write_single_column_vortex_file("strings", strings).await
        });
        let result: String = scan_vortex_file_single_row(
            file,
            "SELECT string_agg(strings, ',') FROM vortex_scan(?) WHERE strings LIKE '%He%'",
            0,
        );
        assert_eq!(result, "Hello,Hey");
    }

    #[test]
    fn test_vortex_scan_integers() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
            write_single_column_vortex_file("number", numbers).await
        });
        let sum: i64 =
            scan_vortex_file_single_row(file, "SELECT SUM(number) FROM vortex_scan(?)", 0);
        assert_eq!(sum, 138);
    }

    #[test]
    fn test_vortex_scan_integers_in_list() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
            write_single_column_vortex_file("number", numbers).await
        });
        let sum: i64 = scan_vortex_file_single_row(
            file,
            "SELECT SUM(number) FROM vortex_scan(?) WHERE number in (1, 42, -5)",
            0,
        );
        assert_eq!(sum, 38);
    }

    #[test]
    fn test_vortex_scan_integers_between() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
            write_single_column_vortex_file("number", numbers).await
        });
        let sum: i64 = scan_vortex_file_single_row(
            file,
            "SELECT SUM(number) FROM vortex_scan(?) WHERE number > 0 and number < 100",
            0,
        );
        assert_eq!(sum, 43);
    }

    #[test]
    fn test_vortex_scan_floats() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let values = PrimitiveArray::from_iter([1.5f64, -2.5, 0.0, 42.42]);
            write_single_column_vortex_file("value", values).await
        });
        let count: i64 = scan_vortex_file_single_row(
            file,
            "SELECT COUNT(*) FROM vortex_scan(?) WHERE value > 0",
            0,
        );
        assert_eq!(count, 2);
    }

    #[test]
    fn test_vortex_scan_constant() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let constant = ConstantArray::new(Scalar::from(42i32), 100);
            write_single_column_vortex_file("constant", constant).await
        });
        let value: i32 =
            scan_vortex_file_single_row(file, "SELECT constant FROM vortex_scan(?) LIMIT 1", 0);
        assert_eq!(value, 42);
    }

    #[test]
    fn test_vortex_scan_booleans() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let flags = vec![true, false, true, true, false];
            let flags_array = BoolArray::new(flags.into(), Validity::NonNullable);
            write_single_column_vortex_file("flag", flags_array).await
        });
        let true_count: i64 = scan_vortex_file_single_row(
            file,
            "SELECT COUNT(*) FROM vortex_scan(?) WHERE flag = true",
            0,
        );
        assert_eq!(true_count, 3);
    }

    #[test]
    fn test_vortex_multi_column() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let file = runtime.block_on(async {
            let f1 = BoolArray::new(
                vec![true, false, true, true, false].into(),
                Validity::NonNullable,
            )
            .to_array();
            let f2 = (0..5).collect::<PrimitiveArray>().to_array();
            let f3 = (100..105).collect::<PrimitiveArray>().to_array();
            write_vortex_file([("f1", f1), ("f2", f2), ("f3", f3)].into_iter()).await
        });

        let result: Vec<i32> = scan_vortex_file(
            file,
            "SELECT f2 FROM vortex_scan(?) WHERE f1 = true and f2 >= 2",
            0,
        )
        .unwrap();

        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn test_vortex_scan_multiple_files() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (tempdir, _file1, _file2) = runtime.block_on(async {
            let tempdir = tempfile::tempdir().unwrap();

            let file1 = write_vortex_file_to_dir(
                tempdir.path(),
                "numbers",
                PrimitiveArray::from_iter([1i32, 2, 3]),
            )
            .await;

            let file2 = write_vortex_file_to_dir(
                tempdir.path(),
                "numbers",
                PrimitiveArray::from_iter([4i32, 5, 6]),
            )
            .await;

            (tempdir, file1, file2)
        });

        // Create glob pattern to match all .vortex files in the temp directory.
        let glob_pattern = format!("{}/*.vortex", tempdir.path().display());

        // Scan both Vortex files.
        let conn = database_connection();
        let result = conn
            .query(&format!(
                "SELECT SUM(numbers) FROM vortex_scan('{glob_pattern}')",
            ))
            .unwrap();

        let total_sum: i64 = result.get(0, 0).unwrap();

        assert_eq!(total_sum, 21);
    }

    #[test]
    fn test_write_file() {
        let conn = database_connection();
        let tempdir = tempfile::tempdir().unwrap();
        let file_path = format!("{}/test.vortex", tempdir.path().to_string_lossy());

        conn.query(&format!(
            "copy (select * as number from generate_series(10)) to '{file_path}' (FORMAT VORTEX);",
        ))
        .unwrap();

        let result = conn
            .query(&format!(
                "SELECT SUM(number) FROM vortex_scan('{file_path}')",
            ))
            .unwrap();
        let total_sum: i64 = result.get(0, 0).unwrap();

        assert_eq!(total_sum, 55);
    }
}
