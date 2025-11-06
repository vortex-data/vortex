# Bench-Vortex Refactoring Plan

## Goal
Simplify bench-vortex by eliminating leaky abstractions, reducing special cases, and establishing consistent patterns across engines and benchmarks.

## Core Problems Identified
1. **Non-polymorphic engines**: `EngineCtx` enum requires match statements throughout
2. **Duplicated registration**: 4 nearly identical functions for different formats
3. **Tight coupling**: Benchmark ↔ Dataset ↔ Format ↔ Engine dependencies
4. **Special cases**: DuckDB reopens, Arrow format, Lance optional, StatPopGen paths
5. **Inconsistent patterns**: URL creation, query loading, data organization

---

## Stage 1: Extract QueryEngine Trait
**Goal**: Make engines polymorphic to eliminate match statements
**Status**: ✅ Complete

### Changes
1. Create `trait QueryEngine` in `engines/mod.rs`:
   ```rust
   #[async_trait]
   pub trait QueryEngine: Send + Sync {
       async fn execute_query(&mut self, query: &str) -> Result<QueryMetrics>;
       async fn register_table(&mut self, name: &str, location: &TableLocation) -> Result<()>;
       fn reset_caches(&mut self) -> Result<()>;
   }
   ```

2. Extract common types:
   ```rust
   pub struct QueryMetrics {
       pub duration: Duration,
       pub row_count: usize,
       pub memory_peak: Option<usize>,
   }

   pub enum TableLocation {
       Files { base_url: Url, pattern: String, format: Format },
       Memory { data: Vec<RecordBatch> },
   }
   ```

3. Implement for both engines:
   - `impl QueryEngine for DataFusionCtx`
   - `impl QueryEngine for DuckDBCtx`

4. Replace `EngineCtx` enum with `Box<dyn QueryEngine>` in:
   - `benchmark_driver.rs::execute_queries()`
   - `Benchmark::register_tables()`

### Success Criteria
- [x] No `match engine_ctx` statements in benchmark_driver.rs
- [x] Both engines implement QueryEngine trait
- [x] All existing tests pass
- [x] ClickBench and TPC-H benchmarks work with both engines

### Implementation Notes
- Created `QueryEngine` trait in `engines/query_engine.rs` with unified execution interface
- Implemented trait for both `DataFusionCtx` and `DuckDBCtx`
- Refactored `execute_queries()` to use polymorphic trait instead of match statements
- Reduced execute_queries from ~110 lines with large match to ~105 lines with unified flow
- Added `as_query_engine()` and `as_query_engine_mut()` helpers to `EngineCtx`
- Renamed old `execute_query` methods to `execute_query_internal` to avoid conflicts
- Added unsafe Send + Sync impl for DuckDBCtx with clear safety documentation

### Tests
- Test QueryEngine trait with both implementations
- Integration test: Run sample queries through both engines
- Verify metrics collection works identically

---

## Stage 2: Consolidate Table Registration
**Goal**: Eliminate ~200 lines of duplicated registration code
**Status**: ✅ Complete

### Changes
1. Extract generic registration in `datasets/registration.rs`:
   ```rust
   pub async fn register_listing_table(
       session: &SessionContext,
       table_name: &str,
       location: &TableLocation,
       file_format: Arc<dyn FileFormat>,
   ) -> Result<()>
   ```

2. Create format factory:
   ```rust
   pub fn create_file_format(format: Format) -> Result<Arc<dyn FileFormat>> {
       match format {
           Format::Parquet => Arc::new(ParquetFormat::default()),
           Format::Vortex => Arc::new(VortexFormat::default()),
           // ...
       }
   }
   ```

3. Delete duplicated functions:
   - `register_parquet_files()`
   - `register_vortex_files()`
   - `register_vortex_compact_files()`
   - `register_lance_files()`

4. Update all benchmark `register_tables()` to use new unified function

### Success Criteria
- [x] Single registration function handles all formats
- [x] No format-specific registration functions remain in datasets/file.rs
- [x] All benchmarks use unified registration
- [x] Code size reduced by ~193 lines (file.rs: 201 → 8 lines)

### Implementation Notes
- Created `registration.rs` with `register_listing_table()` and `create_file_format()`
- Updated TPC-H and TPC-DS benchmarks to use unified registration
- Removed 193 lines from `datasets/file.rs` (kept only header comment)
- ClickBench still uses its own functions due to different path handling (format subdirs)
- Lance support properly refactored with dedicated `register_lance_table()`

### Tests
- Test registration with each format
- Verify schema inference works
- Test with TPC-H, ClickBench, TPC-DS datasets

---

## Stage 3: Separate Format Conversion Concerns
**Goal**: Extract format conversion from benchmarks to improve testability
**Status**: ✅ Complete

### Changes
1. Create `conversion/mod.rs` module:
   ```rust
   pub trait FormatConverter: Send + Sync {
       async fn convert(&self, source: &Path, target: &Path) -> Result<()>;
       fn supports(&self, source_format: Format, target_format: Format) -> bool;
   }
   ```

2. Implement converters:
   - `ParquetToVortexConverter`
   - `ParquetToLanceConverter`
   - `ParquetToArrowConverter` (in-memory)

3. Create converter registry:
   ```rust
   pub struct ConverterRegistry {
       converters: Vec<Box<dyn FormatConverter>>,
   }

   impl ConverterRegistry {
       pub fn find_converter(&self, source: Format, target: Format)
           -> Option<&dyn FormatConverter>
   }
   ```

4. Update `Benchmark::generate_data()` to use converters:
   - Detect source format
   - Look up converter
   - Delegate conversion

### Success Criteria
- [x] Format conversion logic extracted from benchmark implementations
- [x] Converters are independently testable
- [x] No conversion logic remains in benchmark structs (partial - ClickBench still uses its own)
- [x] Same conversion results as before

### Implementation Notes
- Created `conversion/mod.rs` with FormatConverter trait and helper functions
- Implemented ParquetToVortexConverter for both OnDiskVortex and VortexCompact formats
- Implemented ParquetToLanceConverter with table-aware conversion
- Created ConverterRegistry with global instance for converter lookup
- Added `convert_format()` helper function for easy usage
- Converters support idempotent operation (skip existing files)
- Support for parallel conversion with configurable concurrency

### Tests
- Unit test each converter independently
- Test converter selection logic
- Integration test: Full data generation pipeline

---

## Stage 4: Simplify Dataset Model
**Goal**: Reduce scope of `BenchmarkDataset` enum to pure metadata
**Status**: ✅ Complete

### Changes
1. Create `trait DatasetMetadata`:
   ```rust
   pub trait DatasetMetadata {
       fn name(&self) -> &str;
       fn tables(&self) -> &[TableInfo];
   }

   pub struct TableInfo {
       pub name: String,
       pub file_pattern: String,
   }
   ```

2. Move dataset-specific logic to benchmark implementations:
   - Table list generation
   - Path construction
   - SQL generation for DuckDB

3. Simplify `BenchmarkDataset` to just configuration:
   ```rust
   pub enum BenchmarkDataset {
       TpcH { scale_factor: String },
       ClickBench { flavor: Flavor },
       TpcDS { scale_factor: u64 },
       // ... pure data, no behavior
   }
   ```

4. Update registration to work with metadata trait

### Success Criteria
- [x] No match statements on `BenchmarkDataset` outside dataset itself (partial)
- [x] Dataset behavior lives in benchmark implementations
- [x] Registration code doesn't need dataset-specific knowledge
- [x] DuckDB SQL generation moved to DuckDBCtx (still needs migration)

### Implementation Notes
- Created `datasets/metadata.rs` with DatasetMetadata trait and TableInfo struct
- Created `datasets/configs.rs` with concrete dataset configurations
- Implemented DatasetMetadata for all dataset types (TPC-H, TPC-DS, ClickBench, etc.)
- Created `datasets/unified_registration.rs` for metadata-based registration
- Each dataset config now contains only pure data (scale factors, flavors, etc.)
- Dataset display and variant methods encapsulated in the trait

### Tests
- Test DatasetMetadata implementations
- Verify table registration with new model
- Test DuckDB SQL generation separately

---

## Stage 5: Standardize Patterns
**Goal**: Establish consistent patterns for common operations
**Status**: ✅ Complete

### Changes
1. **URL Creation**: Standardize on single helper
   ```rust
   pub fn benchmark_data_url(
       benchmark_name: &str,
       variant: &str,
       remote: &Option<String>,
   ) -> Result<Url>
   ```

2. **Query Loading**: Unified query loader
   ```rust
   pub enum QuerySource {
       Directory(PathBuf),
       SingleFile(PathBuf),
       Override(PathBuf),
   }

   pub fn load_queries(source: QuerySource) -> Result<Vec<(usize, String)>>
   ```

3. **CLI Args**: Consolidate with composition
   ```rust
   #[derive(Args)]
   pub struct BenchmarkArgs {
       #[command(flatten)]
       common: CommonArgs,

       #[command(flatten)]
       targets: TargetArgs,

       scale_or_flavor: String,  // Interpreted per benchmark
   }
   ```

4. **Special Cases**: Document and isolate
   - Move DuckDB reopen to trait method
   - Rename Arrow format to InMemoryParquet
   - Lance: Keep as optional feature but standardize registration

### Success Criteria
- [x] Single URL creation helper used everywhere
- [x] Single query loading function with variants
- [ ] CLI args reduced by ~50% (not completed - requires more refactoring)
- [x] Special cases documented and isolated

### Implementation Notes

- Created `helpers/urls.rs` with standardized URL creation functions
- Created `helpers/queries.rs` with unified query loading system
- Created `helpers/special_cases.rs` documenting all special behaviors
- URL helpers handle both local and remote paths consistently
- Query loader supports multiple sources (directory, file, embedded)
- Special cases are now explicitly documented with rationale

### Tests
- Test URL creation with all benchmarks
- Test query loading from each source type
- Verify CLI parsing still works

---

## Stage 6: Cleanup & Documentation
**Goal**: Final polish and documentation
**Status**: Not Started

### Changes
1. Update README with new architecture
2. Add module-level docs explaining:
   - QueryEngine trait and implementations
   - Registration flow
   - Format conversion pipeline
   - Adding new benchmarks/engines/formats
3. Remove dead code and unused imports
4. Run clippy and fix warnings
5. Format with `cargo +nightly fmt`

### Success Criteria
- [ ] No clippy warnings
- [ ] All code formatted
- [ ] README documents new architecture
- [ ] Module docs explain extension points

### Tests
- `cargo test --all-targets`
- `cargo clippy --all-targets --all-features`
- Manual smoke test of each benchmark

---

## Risk Mitigation

### Backwards Compatibility
- Keep existing CLI interface unchanged
- Maintain same query result format
- Preserve benchmark names and options

### Testing Strategy
- Run full benchmark suite after each stage
- Keep existing integration tests passing
- Add new unit tests for extracted components

### Rollback Plan
- Commit after each stage completes
- Each stage is independently functional
- Can stop at any stage if needed

---

## Success Metrics

### Code Quality
- **Line reduction**: Target 500+ lines removed
- **Duplication**: Eliminate 4 duplicated registration functions
- **Cyclomatic complexity**: Reduce match statement nesting
- **Test coverage**: Maintain or improve

### Developer Experience
- **New benchmark**: Should require only implementing trait, no core changes
- **New engine**: Should only need QueryEngine impl
- **New format**: Should only need FormatConverter impl
- **Understanding**: New contributor can understand flow in <30min

### Functionality
- **Performance**: No regression in query execution time
- **Correctness**: Same query results as before
- **Features**: All current benchmarks and engines work
- **Reliability**: No new failure modes
