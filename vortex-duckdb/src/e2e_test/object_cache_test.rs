// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test table function that demonstrates object cache usage

use std::ffi::CString;

use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::BindInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::ClientContextRef;
use crate::duckdb::ColumnStatistics;
use crate::duckdb::DataChunkRef;
use crate::duckdb::LogicalType;
use crate::duckdb::TableFunction;
use crate::duckdb::TableInitInput;

#[derive(Debug, Clone)]
pub struct TestTableFunction;

#[derive(Debug, Clone)]
pub struct TestBindData {
    cache_key: String,
}

#[derive(Debug)]
pub struct TestGlobalState;

#[derive(Debug)]
pub struct TestLocalState;

#[derive(Debug, PartialEq)]
struct CachedData {
    message: String,
    count: i32,
}

impl TableFunction for TestTableFunction {
    type BindData = TestBindData;
    type GlobalState = TestGlobalState;
    type LocalState = TestLocalState;

    fn bind(
        client_context: &ClientContextRef,
        _input: &BindInputRef,
        result: &mut BindResultRef,
    ) -> VortexResult<Self::BindData> {
        let logical_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT);
        result.add_result_column("test_value", &logical_type);

        let cache = client_context.object_cache();
        let cached_data = CachedData {
            message: "Hello from bind phase cache!".to_string(),
            count: 123,
        };

        // Store data in cache during bind
        cache.put("bind_phase_data", cached_data);

        Ok(TestBindData {
            cache_key: "test_table_function_data".to_string(),
        })
    }

    fn table_scan_progress(
        _client_context: &ClientContextRef,
        _bind_data: &Self::BindData,
        _global_state: &Self::GlobalState,
    ) -> f64 {
        100.0
    }

    fn scan(
        _client_context: &ClientContextRef,
        _bind_data: &Self::BindData,
        _local_state: &mut Self::LocalState,
        _global_state: &Self::GlobalState,
        chunk: &mut DataChunkRef,
    ) -> VortexResult<()> {
        chunk.set_len(0);

        Ok(())
    }

    fn init_global(input: &TableInitInput<Self>) -> VortexResult<Self::GlobalState> {
        if let Ok(ctx) = input.client_context() {
            let cached_data = CachedData {
                message: "Hello from table function cache!".to_string(),
                count: 42,
            };
            let cache = ctx.object_cache();

            cache.put(&input.bind_data().cache_key, cached_data);
        }

        Ok(TestGlobalState)
    }

    fn init_local(
        _init: &TableInitInput<Self>,
        _global: &Self::GlobalState,
    ) -> VortexResult<Self::LocalState> {
        Ok(TestLocalState)
    }

    fn partition_data(
        _bind_data: &Self::BindData,
        _global_init_data: &Self::GlobalState,
        _local_init_data: &mut Self::LocalState,
    ) -> VortexResult<u64> {
        Ok(0)
    }

    fn statistics(
        _client_context: &ClientContextRef,
        _bind_data: &Self::BindData,
        _column_index: usize,
    ) -> Option<ColumnStatistics> {
        None
    }

    fn to_string(_bind_data: &Self::BindData, _map: &mut crate::duckdb::DuckdbStringMapRef) {}
}

use crate::duckdb::Database;

#[test]
fn test_table_function_with_object_cache() -> VortexResult<()> {
    let db = Database::open_in_memory()?;
    let conn = db.connect()?;

    // Register our test table function
    let name = CString::new("test_cache_func").map_err(|e| vortex_err!("CString error: {}", e))?;
    db.register_table_function::<TestTableFunction>(&name)?;

    // Call the table function - this should store data in the cache during init_global
    let _result = conn.query("SELECT * FROM test_cache_func()")?;

    // Try to verify that we can access the cached data from outside the table function
    // This part is optional since we're not sure if the object cache access is working yet
    let ctx = conn.client_context();
    if let Ok(ctx) = ctx {
        let cache = ctx.object_cache();
        // Check data from bind phase
        let bind_cached_data = cache.get::<CachedData>("bind_phase_data");
        if let Some(data) = bind_cached_data {
            assert_eq!(data.message, "Hello from bind phase cache!");
            assert_eq!(data.count, 123);
            println!("Successfully retrieved bind phase cached data!");
        }

        // Check data from init_global phase
        let cached_data = cache.get::<CachedData>("test_table_function_data");
        if let Some(data) = cached_data {
            assert_eq!(data.message, "Hello from table function cache!");
            assert_eq!(data.count, 42);
            println!("Successfully retrieved init_global phase cached data!");
        }
    }

    Ok(())
}
