# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## `vortex-mask` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-mask-v0.21.1...vortex-mask-v0.22.0) - 2025-01-21

### Added
- add vortex-mask crate (#2019)

### Fixed
- update setup instructions (rye -> uv) (#1176)
- fix docs badge in readme ([#753](https://github.com/spiraldb/vortex/pull/753))

### Other
- link docs from README (#1521)
- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))
- very small README.md fixes
- More README.md improvements ([#1084](https://github.com/spiraldb/vortex/pull/1084))
- Update README.md ([#1055](https://github.com/spiraldb/vortex/pull/1055))
- minor addition to README ([#1030](https://github.com/spiraldb/vortex/pull/1030))
- updated README ([#876](https://github.com/spiraldb/vortex/pull/876))
- release to Test PyPI on each push to version tags (#760)
- Run ETE benchmarks with MiMalloc and leave a note encouraging its usage ([#399](https://github.com/spiraldb/vortex/pull/399))
- README updates ([#394](https://github.com/spiraldb/vortex/pull/394))
- Download flatc instead of building it from source ([#374](https://github.com/spiraldb/vortex/pull/374))
- Update README.md ([#337](https://github.com/spiraldb/vortex/pull/337))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Add note to readme about git submodules and zig version ([#176](https://github.com/spiraldb/vortex/pull/176))
- acknowledgments ([#171](https://github.com/spiraldb/vortex/pull/171))
- Update README.md ([#168](https://github.com/spiraldb/vortex/pull/168))
- More README updates ([#140](https://github.com/spiraldb/vortex/pull/140))
- Update README.md
- readme improvements ([#137](https://github.com/spiraldb/vortex/pull/137))
- README ([#102](https://github.com/spiraldb/vortex/pull/102))
- Root project is vortex-array ([#67](https://github.com/spiraldb/vortex/pull/67))
- Add minimal description to readme and fixup cargo metadata ([#30](https://github.com/spiraldb/vortex/pull/30))
- Add Readme

## `vortex-datafusion` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.21.1...vortex-datafusion-v0.22.0) - 2025-01-21

### Added
- feature flag ExecDriver implementations ([#2005](https://github.com/spiraldb/vortex/pull/2005))
- PyVortex with V2 IO (#1977)
- Unify `LazyDType` into `StructDType` (#1826)
- File row splits (#1709)

### Other
- Split by layout chunks of fields used in query ([#2022](https://github.com/spiraldb/vortex/pull/2022))
- Opportunistically compute cardinality stat for constant array  ([#2018](https://github.com/spiraldb/vortex/pull/2018))
- Add a split by constant to the datafusion opener ([#1995](https://github.com/spiraldb/vortex/pull/1995))
- Remove all old IO code (#1989)
- Only persist certain stats when serializing arrays in FlatLayout ([#1984](https://github.com/spiraldb/vortex/pull/1984))
- Split conjunct into scanner and apply one by one ([#1963](https://github.com/spiraldb/vortex/pull/1963))
- Combine scan and take ([#1975](https://github.com/spiraldb/vortex/pull/1975))
- Vortex Layouts DataFusion Statistics ([#1967](https://github.com/spiraldb/vortex/pull/1967))
- Added a ScanBuilder & dont evaluate a project with an empty `row_mask` in scan ([#1960](https://github.com/spiraldb/vortex/pull/1960))
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- A data fusion inspired traversal API for expressions ([#1828](https://github.com/spiraldb/vortex/pull/1828))
- Export DType fields as public ([#1833](https://github.com/spiraldb/vortex/pull/1833))
- Move RowFilter, PruningPredicate, and expr_project to vortex-expr ([#1820](https://github.com/spiraldb/vortex/pull/1820))
- Into arrow with hint ([#1730](https://github.com/spiraldb/vortex/pull/1730))
- ContextRef = Arc<Context> ([#1802](https://github.com/spiraldb/vortex/pull/1802))
- *(deps)* update datafusion to v44 (major) (#1770)
- Pull out message cache from layout reader ([#1773](https://github.com/spiraldb/vortex/pull/1773))
- Pull dtype out of the message cache ([#1771](https://github.com/spiraldb/vortex/pull/1771))
- update rust-toolchain (#1751)
- vortex-buffer (#1742)
- Use vortex-buffer over bytes::Bytes ([#1713](https://github.com/spiraldb/vortex/pull/1713))

## `vortex-zigzag` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.21.1...vortex-zigzag-v0.22.0) - 2025-01-21

### Added
- compare array metadata against checked in goldenfiles (#1812)

### Fixed
- Trim down large arrays in tests and exclude slow projects from miri (#1871)
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- prefer into_buffer_mut() to into_buffer().into_mut() (#1784)
- Improve alp decode ([#1764](https://github.com/spiraldb/vortex/pull/1764))
- vortex-buffer (#1742)

## `vortex-sampling-compressor` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.21.1...vortex-sampling-compressor-v0.22.0) - 2025-01-21

### Added
- re-enable ALP-RD as top-level float compressor (#1725)
- BitPackedCompressor allows signed arrays (#1699)

### Fixed
- FlexBuffer serialization ambiguity for binary/string/list(u8) (#1859)
- sampling compressor overflow with large parameters (#1737)
- BitPackedArray enforces can only be built over non-negative values (#1705)

### Other
- remove roaring int/bool & run end bool arrays (#2020)
- Support reusing dictionary when encoding values ([#2008](https://github.com/spiraldb/vortex/pull/2008))
- ContextRef = Arc<Context> ([#1802](https://github.com/spiraldb/vortex/pull/1802))
- vortex-buffer (#1742)

## `vortex-runend` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.21.1...vortex-runend-v0.22.0) - 2025-01-21

### Added
- compare array metadata against checked in goldenfiles (#1812)
- add BinaryNumericFn for array arithmetic (#1640)

### Fixed
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- Optimize RunEnd array filter for sparse masks (#1969)
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Use Validity::from when building validities ([#1793](https://github.com/spiraldb/vortex/pull/1793))
- vortex-buffer (#1742)

## `vortex-fsst` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.21.1...vortex-fsst-v0.22.0) - 2025-01-21

### Added
- Use buffers for opaque data in VarBin and VarBinView (#1935)
- Hold views in a buffer instead of an array (#1933)
- compare array metadata against checked in goldenfiles (#1812)

### Fixed
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Arc InnerArrayData and add a WeakArrayData ([#1930](https://github.com/spiraldb/vortex/pull/1930))
- Faster FSST decompression ([#1769](https://github.com/spiraldb/vortex/pull/1769))
- Remove FSST copy ([#1757](https://github.com/spiraldb/vortex/pull/1757))
- vortex-buffer (#1742)
- Add debug assertions to ComputeFn results ([#1716](https://github.com/spiraldb/vortex/pull/1716))

## `vortex-scan` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-scan-v0.21.1...vortex-scan-v0.22.0) - 2025-01-21

### Fixed
- update setup instructions (rye -> uv) (#1176)
- fix docs badge in readme ([#753](https://github.com/spiraldb/vortex/pull/753))

### Other
- Split by layout chunks of fields used in query ([#2022](https://github.com/spiraldb/vortex/pull/2022))
- Filter empty from scan batches ([#1996](https://github.com/spiraldb/vortex/pull/1996))
- Some simplifications to pruning and exprs ([#1990](https://github.com/spiraldb/vortex/pull/1990))
- Split conjunct into scanner and apply one by one ([#1963](https://github.com/spiraldb/vortex/pull/1963))
- Added a ScanBuilder & dont evaluate a project with an empty `row_mask` in scan ([#1960](https://github.com/spiraldb/vortex/pull/1960))
- Fix clickbench ([#1956](https://github.com/spiraldb/vortex/pull/1956))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Add take implementation ([#1955](https://github.com/spiraldb/vortex/pull/1955))
- assorted cleanups (#1953)
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- I/O Driver ([#1897](https://github.com/spiraldb/vortex/pull/1897))
- Scaffolding for layout stats ([#1885](https://github.com/spiraldb/vortex/pull/1885))
- Async Layouts ([#1866](https://github.com/spiraldb/vortex/pull/1866))
- Fix vortex-dtype feature dependency ([#1853](https://github.com/spiraldb/vortex/pull/1853))
- Vortex Layouts - scanner ([#1849](https://github.com/spiraldb/vortex/pull/1849))
- link docs from README (#1521)
- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))
- very small README.md fixes
- More README.md improvements ([#1084](https://github.com/spiraldb/vortex/pull/1084))
- Update README.md ([#1055](https://github.com/spiraldb/vortex/pull/1055))
- minor addition to README ([#1030](https://github.com/spiraldb/vortex/pull/1030))
- updated README ([#876](https://github.com/spiraldb/vortex/pull/876))
- release to Test PyPI on each push to version tags (#760)
- Run ETE benchmarks with MiMalloc and leave a note encouraging its usage ([#399](https://github.com/spiraldb/vortex/pull/399))
- README updates ([#394](https://github.com/spiraldb/vortex/pull/394))
- Download flatc instead of building it from source ([#374](https://github.com/spiraldb/vortex/pull/374))
- Update README.md ([#337](https://github.com/spiraldb/vortex/pull/337))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Add note to readme about git submodules and zig version ([#176](https://github.com/spiraldb/vortex/pull/176))
- acknowledgments ([#171](https://github.com/spiraldb/vortex/pull/171))
- Update README.md ([#168](https://github.com/spiraldb/vortex/pull/168))
- More README updates ([#140](https://github.com/spiraldb/vortex/pull/140))
- Update README.md
- readme improvements ([#137](https://github.com/spiraldb/vortex/pull/137))
- README ([#102](https://github.com/spiraldb/vortex/pull/102))
- Root project is vortex-array ([#67](https://github.com/spiraldb/vortex/pull/67))
- Add minimal description to readme and fixup cargo metadata ([#30](https://github.com/spiraldb/vortex/pull/30))
- Add Readme

## `vortex-layout` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-layout-v0.21.1...vortex-layout-v0.22.0) - 2025-01-21

### Added
- StructDType `from_iter` (#2003)
- Unify `LazyDType` into `StructDType` (#1826)
- new expression builders (#1829)

### Fixed
- don't eval expr on empty chunk mask (#1998)
- comments (#1923)
- update setup instructions (rye -> uv) (#1176)
- fix docs badge in readme ([#753](https://github.com/spiraldb/vortex/pull/753))

### Other
- Use FuturesUnordered when order of try_join_all is not relevant ([#2026](https://github.com/spiraldb/vortex/pull/2026))
- Split by layout chunks of fields used in query ([#2022](https://github.com/spiraldb/vortex/pull/2022))
- Avoid double validation of file layout flatbuffers ([#2014](https://github.com/spiraldb/vortex/pull/2014))
- Nicer pack expr convenience function ([#2002](https://github.com/spiraldb/vortex/pull/2002))
- Add a split by constant to the datafusion opener ([#1995](https://github.com/spiraldb/vortex/pull/1995))
- Filter empty from scan batches ([#1996](https://github.com/spiraldb/vortex/pull/1996))
- Cache pruning mask ([#1994](https://github.com/spiraldb/vortex/pull/1994))
- Only persist certain stats when serializing arrays in FlatLayout ([#1984](https://github.com/spiraldb/vortex/pull/1984))
- Use prune with get item, not column ([#1976](https://github.com/spiraldb/vortex/pull/1976))
- Gourmet changes (#1972)
- Vortex Layouts DataFusion Statistics ([#1967](https://github.com/spiraldb/vortex/pull/1967))
- Fix clickbench ([#1956](https://github.com/spiraldb/vortex/pull/1956))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Check partition count in expression partition logic ([#1944](https://github.com/spiraldb/vortex/pull/1944))
- Only construct StructReader field_lookup when there's more than 80 fields (#1945)
- Cache partitioned expressions in StructLayoutReader ([#1947](https://github.com/spiraldb/vortex/pull/1947))
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- nit in layouts/chunked/reader.rs (#1926)
- Remove field from struct layout ([#1919](https://github.com/spiraldb/vortex/pull/1919))
- Vortex Layouts - Some Cleanup ([#1917](https://github.com/spiraldb/vortex/pull/1917))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- Vortex Layouts - Drivers ([#1914](https://github.com/spiraldb/vortex/pull/1914))
- Fix the ident splitting into expr partition ([#1913](https://github.com/spiraldb/vortex/pull/1913))
- Struct layout eval with sub-expression slicing and push down ([#1893](https://github.com/spiraldb/vortex/pull/1893))
- I/O Driver ([#1897](https://github.com/spiraldb/vortex/pull/1897))
- Scaffolding for layout stats ([#1885](https://github.com/spiraldb/vortex/pull/1885))
- Segment Alignment ([#1883](https://github.com/spiraldb/vortex/pull/1883))
- Use flatbuffers::follow instead of init_from_table ([#1878](https://github.com/spiraldb/vortex/pull/1878))
- Async Layouts ([#1866](https://github.com/spiraldb/vortex/pull/1866))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Add SplitBy to VortexOpenOptions ([#1858](https://github.com/spiraldb/vortex/pull/1858))
- Vortex Layouts - scanner ([#1849](https://github.com/spiraldb/vortex/pull/1849))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- Vortex Layouts File V2 ([#1830](https://github.com/spiraldb/vortex/pull/1830))
- Arc layout scan ([#1825](https://github.com/spiraldb/vortex/pull/1825))
- Vortex Layouts - chunk pruning ([#1824](https://github.com/spiraldb/vortex/pull/1824))
- Vortex Layouts - Chunked ([#1819](https://github.com/spiraldb/vortex/pull/1819))
- Vortex Layouts - Chunked ([#1814](https://github.com/spiraldb/vortex/pull/1814))
- Initial Vortex Layouts ([#1805](https://github.com/spiraldb/vortex/pull/1805))
- link docs from README (#1521)
- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))
- very small README.md fixes
- More README.md improvements ([#1084](https://github.com/spiraldb/vortex/pull/1084))
- Update README.md ([#1055](https://github.com/spiraldb/vortex/pull/1055))
- minor addition to README ([#1030](https://github.com/spiraldb/vortex/pull/1030))
- updated README ([#876](https://github.com/spiraldb/vortex/pull/876))
- release to Test PyPI on each push to version tags (#760)
- Run ETE benchmarks with MiMalloc and leave a note encouraging its usage ([#399](https://github.com/spiraldb/vortex/pull/399))
- README updates ([#394](https://github.com/spiraldb/vortex/pull/394))
- Download flatc instead of building it from source ([#374](https://github.com/spiraldb/vortex/pull/374))
- Update README.md ([#337](https://github.com/spiraldb/vortex/pull/337))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Add note to readme about git submodules and zig version ([#176](https://github.com/spiraldb/vortex/pull/176))
- acknowledgments ([#171](https://github.com/spiraldb/vortex/pull/171))
- Update README.md ([#168](https://github.com/spiraldb/vortex/pull/168))
- More README updates ([#140](https://github.com/spiraldb/vortex/pull/140))
- Update README.md
- readme improvements ([#137](https://github.com/spiraldb/vortex/pull/137))
- README ([#102](https://github.com/spiraldb/vortex/pull/102))
- Root project is vortex-array ([#67](https://github.com/spiraldb/vortex/pull/67))
- Add minimal description to readme and fixup cargo metadata ([#30](https://github.com/spiraldb/vortex/pull/30))
- Add Readme

## `vortex-ipc` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-ipc-v0.21.1...vortex-ipc-v0.22.0) - 2025-01-21

### Added
- Unify `LazyDType` into `StructDType` (#1826)

### Other
- Move array data into IPC ([#1892](https://github.com/spiraldb/vortex/pull/1892))
- Segment Alignment ([#1883](https://github.com/spiraldb/vortex/pull/1883))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Arrays have multiple buffers ([#1743](https://github.com/spiraldb/vortex/pull/1743))
- *(deps)* update rust crate itertools to 0.14.0 (#1763)
- ContextRef = Arc<Context> ([#1802](https://github.com/spiraldb/vortex/pull/1802))
- move TODOs out of public doc strings (#1780)
- Zero-copy BufMessageReader ([#1753](https://github.com/spiraldb/vortex/pull/1753))
- vortex-buffer (#1742)
- Message Codec ([#1692](https://github.com/spiraldb/vortex/pull/1692))

## `vortex-io` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-io-v0.21.1...vortex-io-v0.22.0) - 2025-01-21

### Other
- TokioFile doesn't zero memory before reading into it ([#2013](https://github.com/spiraldb/vortex/pull/2013))
- Port v1 tests to v2 ([#1980](https://github.com/spiraldb/vortex/pull/1980))
- Gourmet changes (#1972)
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- vortex-buffer (#1742)
- Use vortex-buffer over bytes::Bytes ([#1713](https://github.com/spiraldb/vortex/pull/1713))
- move IoBuf to vortex-io (#1714)
- Message Codec ([#1692](https://github.com/spiraldb/vortex/pull/1692))
- test for repeated columns in a projection (#1691)

## `vortex-file` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-file-v0.21.1...vortex-file-v0.22.0) - 2025-01-21

### Added
- feature flag ExecDriver implementations ([#2005](https://github.com/spiraldb/vortex/pull/2005))
- PyVortex with V2 IO (#1977)
- Unify `LazyDType` into `StructDType` (#1826)
- new expression builders (#1829)
- File row splits (#1709)

### Fixed
- Remove Scan::build function (#1964)
- FilterMask::from_indices requires a vec (broken after filtermask changes) (#1959)
- binary_numeric is now correct (+ tests!) (#1721)
- also need to update ps_loc (#1792)
- InitialRead will refetch data to capture schema + layout (#1787)
- Message cache can be mutated with a shared reference (#1781)

### Other
- Use FuturesUnordered when order of try_join_all is not relevant ([#2026](https://github.com/spiraldb/vortex/pull/2026))
- Add execution concurrency == 2 * io concurrency ([#2023](https://github.com/spiraldb/vortex/pull/2023))
- Split by layout chunks of fields used in query ([#2022](https://github.com/spiraldb/vortex/pull/2022))
- Avoid double validation of file layout flatbuffers ([#2014](https://github.com/spiraldb/vortex/pull/2014))
- Filter empty from scan batches ([#1996](https://github.com/spiraldb/vortex/pull/1996))
- Some simplifications to pruning and exprs ([#1990](https://github.com/spiraldb/vortex/pull/1990))
- Remove all old IO code (#1989)
- Only persist certain stats when serializing arrays in FlatLayout ([#1984](https://github.com/spiraldb/vortex/pull/1984))
- Port v1 tests to v2 ([#1980](https://github.com/spiraldb/vortex/pull/1980))
- Split conjunct into scanner and apply one by one ([#1963](https://github.com/spiraldb/vortex/pull/1963))
- Combine scan and take ([#1975](https://github.com/spiraldb/vortex/pull/1975))
- Gourmet changes (#1972)
- Vortex Layouts DataFusion Statistics ([#1967](https://github.com/spiraldb/vortex/pull/1967))
- Added a ScanBuilder & dont evaluate a project with an empty `row_mask` in scan ([#1960](https://github.com/spiraldb/vortex/pull/1960))
- Fix clickbench ([#1956](https://github.com/spiraldb/vortex/pull/1956))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Add take implementation ([#1955](https://github.com/spiraldb/vortex/pull/1955))
- Cache coalesced segments ([#1949](https://github.com/spiraldb/vortex/pull/1949))
- assorted cleanups (#1953)
- Cache partitioned expressions in StructLayoutReader ([#1947](https://github.com/spiraldb/vortex/pull/1947))
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- Support opening Vortex files without I/O ([#1920](https://github.com/spiraldb/vortex/pull/1920))
- Vortex Layouts - Some Cleanup ([#1917](https://github.com/spiraldb/vortex/pull/1917))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- Vortex Layouts - Drivers ([#1914](https://github.com/spiraldb/vortex/pull/1914))
- I/O Driver ([#1897](https://github.com/spiraldb/vortex/pull/1897))
- Rename Vortex file options ([#1891](https://github.com/spiraldb/vortex/pull/1891))
- Scaffolding for layout stats ([#1885](https://github.com/spiraldb/vortex/pull/1885))
- Segment Alignment ([#1883](https://github.com/spiraldb/vortex/pull/1883))
- Fix splits to read empty struct/chunked arrays. ([#1876](https://github.com/spiraldb/vortex/pull/1876))
- Async Layouts ([#1866](https://github.com/spiraldb/vortex/pull/1866))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Add SplitBy to VortexOpenOptions ([#1858](https://github.com/spiraldb/vortex/pull/1858))
- Add vortex-expr GetItem, and update Select ([#1836](https://github.com/spiraldb/vortex/pull/1836))
- Vortex Layouts - scanner ([#1849](https://github.com/spiraldb/vortex/pull/1849))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- A data fusion inspired traversal API for expressions ([#1828](https://github.com/spiraldb/vortex/pull/1828))
- Export DType fields as public ([#1833](https://github.com/spiraldb/vortex/pull/1833))
- Vortex Layouts File V2 ([#1830](https://github.com/spiraldb/vortex/pull/1830))
- Move RowFilter, PruningPredicate, and expr_project to vortex-expr ([#1820](https://github.com/spiraldb/vortex/pull/1820))
- ContextRef = Arc<Context> ([#1802](https://github.com/spiraldb/vortex/pull/1802))
- Rename recordbatchreader to camel base ([#1801](https://github.com/spiraldb/vortex/pull/1801))
- Add a `from_iter_slow` method to ListArray to allow convenient creation of lists ([#1778](https://github.com/spiraldb/vortex/pull/1778))
- move TODOs out of public doc strings (#1780)
- Pull out message cache from layout reader ([#1773](https://github.com/spiraldb/vortex/pull/1773))
- Pull dtype out of the message cache ([#1771](https://github.com/spiraldb/vortex/pull/1771))
- Zero-copy BufMessageReader ([#1753](https://github.com/spiraldb/vortex/pull/1753))
- vortex-buffer (#1742)
- Use vortex-buffer over bytes::Bytes ([#1713](https://github.com/spiraldb/vortex/pull/1713))
- Message Codec ([#1692](https://github.com/spiraldb/vortex/pull/1692))
- test for repeated columns in a projection (#1691)

## `vortex-expr` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.21.1...vortex-expr-v0.22.0) - 2025-01-21

### Added
- StructDType `from_iter` (#2003)
- PyVortex with V2 IO (#1977)
- teach VortexExpr to dtype (#1811)
- conjunctive & negative normal forms (#1857)
- implement Pack expression (#1810)
- new expression builders (#1829)

### Fixed
- `like` expr eval and children methods (#1937)

### Other
- Return empty conjunct list if filter is true ([#2027](https://github.com/spiraldb/vortex/pull/2027))
- Split by layout chunks of fields used in query ([#2022](https://github.com/spiraldb/vortex/pull/2022))
- Move doc string to pub fn ([#2017](https://github.com/spiraldb/vortex/pull/2017))
- Clean up expr analysis immediate accesses ([#1999](https://github.com/spiraldb/vortex/pull/1999))
- Nicer pack expr convenience function ([#2002](https://github.com/spiraldb/vortex/pull/2002))
- Update simplify and add docs ([#2000](https://github.com/spiraldb/vortex/pull/2000))
- Some simplifications to pruning and exprs ([#1990](https://github.com/spiraldb/vortex/pull/1990))
- Remove all old IO code (#1989)
- LikeFn to take owned ([#1974](https://github.com/spiraldb/vortex/pull/1974))
- Simplify FoldDown return ([#1991](https://github.com/spiraldb/vortex/pull/1991))
- Use prune with get item, not column ([#1976](https://github.com/spiraldb/vortex/pull/1976))
- Check partition count in expression partition logic ([#1944](https://github.com/spiraldb/vortex/pull/1944))
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- Arc InnerArrayData and add a WeakArrayData ([#1930](https://github.com/spiraldb/vortex/pull/1930))
- Remove select with typed_simplify pass ([#1929](https://github.com/spiraldb/vortex/pull/1929))
- Shared identity expression ([#1916](https://github.com/spiraldb/vortex/pull/1916))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- Fix the ident splitting into expr partition ([#1913](https://github.com/spiraldb/vortex/pull/1913))
- Struct layout eval with sub-expression slicing and push down ([#1893](https://github.com/spiraldb/vortex/pull/1893))
- :Index to Field::Name expr transform (#1894)
- Rescope expression then the identity is a struct dtype ([#1887](https://github.com/spiraldb/vortex/pull/1887))
- Add Hash to vortex-expr and therefore vortex-scalar ([#1869](https://github.com/spiraldb/vortex/pull/1869))
- Add vortex-expr GetItem, and update Select ([#1836](https://github.com/spiraldb/vortex/pull/1836))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- A data fusion inspired traversal API for expressions ([#1828](https://github.com/spiraldb/vortex/pull/1828))
- Add DynEq trait for VortexExpr and implement PartialEq for VortexExpr ([#1837](https://github.com/spiraldb/vortex/pull/1837))
- Export DType fields as public ([#1833](https://github.com/spiraldb/vortex/pull/1833))
- Vortex Layouts - chunk pruning ([#1824](https://github.com/spiraldb/vortex/pull/1824))
- Move RowFilter, PruningPredicate, and expr_project to vortex-expr ([#1820](https://github.com/spiraldb/vortex/pull/1820))
- vortex-buffer (#1742)

## `vortex-dict` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.21.1...vortex-dict-v0.22.0) - 2025-01-21

### Added
- Use buffers for opaque data in VarBin and VarBinView (#1935)
- compare array metadata against checked in goldenfiles (#1812)
- add BinaryNumericFn for array arithmetic (#1640)

### Fixed
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Support reusing dictionary when encoding values ([#2008](https://github.com/spiraldb/vortex/pull/2008))
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- LikeFn to take owned ([#1974](https://github.com/spiraldb/vortex/pull/1974))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- vortex-buffer (#1742)

## `vortex-datetime-parts` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.21.1...vortex-datetime-parts-v0.22.0) - 2025-01-21

### Added
- teach DateTimePartsArray to cast (#1946)
- compare array metadata against checked in goldenfiles (#1812)

### Fixed
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Converting LogicalValidity into Validity requires passing Nullability ([#1834](https://github.com/spiraldb/vortex/pull/1834))
- vortex-buffer (#1742)

## `vortex-bytebool` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.21.1...vortex-bytebool-v0.22.0) - 2025-01-21

### Added
- compare array metadata against checked in goldenfiles (#1812)

### Fixed
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- Arrays have multiple buffers ([#1743](https://github.com/spiraldb/vortex/pull/1743))
- vortex-buffer (#1742)

## `vortex-fastlanes` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.21.1...vortex-fastlanes-v0.22.0) - 2025-01-21

### Added
- optimize FoRArray compare (#1656)
- compare array metadata against checked in goldenfiles (#1812)
- BitPackedCompressor allows signed arrays (#1699)

### Fixed
- Trim down large arrays in tests and exclude slow projects from miri (#1871)
- binary_numeric is now correct (+ tests!) (#1721)
- Signed bitpacked arrays handle searching for negative values (#1800)
- BitPackedArray ptype changed by compute funcs (#1724)
- BitPackedArray enforces can only be built over non-negative values (#1705)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- FoR compare handles correctly encodes compared to constant into compressed space ([#1840](https://github.com/spiraldb/vortex/pull/1840))
- Arrays have multiple buffers ([#1743](https://github.com/spiraldb/vortex/pull/1743))
- Store PValue in FoR metadata ([#1768](https://github.com/spiraldb/vortex/pull/1768))
- Improve alp decode ([#1764](https://github.com/spiraldb/vortex/pull/1764))
- make miri tests much faster (#1756)
- vortex-buffer (#1742)
- use PrimitiveArray::patch in bitpacking take (#1690)

## `vortex-scalar` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.21.1...vortex-scalar-v0.22.0) - 2025-01-21

### Added
- Unify `LazyDType` into `StructDType` (#1826)
- impl subtract for primitive_scalar (#1592)
- add BinaryNumericFn for array arithmetic (#1640)

### Fixed
- FlexBuffer serialization ambiguity for binary/string/list(u8) (#1859)
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Some simplifications to pruning and exprs ([#1990](https://github.com/spiraldb/vortex/pull/1990))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- Add Hash to vortex-expr and therefore vortex-scalar ([#1869](https://github.com/spiraldb/vortex/pull/1869))
- Implement Display for ListScalar ([#1850](https://github.com/spiraldb/vortex/pull/1850))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- Add DynEq trait for VortexExpr and implement PartialEq for VortexExpr ([#1837](https://github.com/spiraldb/vortex/pull/1837))
- Export DType fields as public ([#1833](https://github.com/spiraldb/vortex/pull/1833))
- Store PValue in FoR metadata ([#1768](https://github.com/spiraldb/vortex/pull/1768))
- vortex-buffer (#1742)
- Use vortex-buffer over bytes::Bytes ([#1713](https://github.com/spiraldb/vortex/pull/1713))
- Add debug assertions to ComputeFn results ([#1716](https://github.com/spiraldb/vortex/pull/1716))
- Added a list builder ([#1711](https://github.com/spiraldb/vortex/pull/1711))

## `vortex-flatbuffers` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.21.1...vortex-flatbuffers-v0.22.0) - 2025-01-21

### Other
- Move array data into IPC ([#1892](https://github.com/spiraldb/vortex/pull/1892))
- Segment Alignment ([#1883](https://github.com/spiraldb/vortex/pull/1883))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Vortex Layouts File V2 ([#1830](https://github.com/spiraldb/vortex/pull/1830))
- *(deps)* update rust crate itertools to 0.14.0 (#1763)
- Initial Vortex Layouts ([#1805](https://github.com/spiraldb/vortex/pull/1805))
- vortex-buffer (#1742)
- Message Codec ([#1692](https://github.com/spiraldb/vortex/pull/1692))

## `vortex-dtype` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.21.1...vortex-dtype-v0.22.0) - 2025-01-21

### Added
- StructDType `from_iter` (#2003)
- tests for ext dtype equality (#1962)
- Don't allocate another DType in `DType::eq_ignore_nullability` (#1948)
- Unify `LazyDType` into `StructDType` (#1826)
- File row splits (#1709)
- add BinaryNumericFn for array arithmetic (#1640)

### Fixed
- ExtDType equality (#1961)

### Other
- Split by layout chunks of fields used in query ([#2022](https://github.com/spiraldb/vortex/pull/2022))
- Vortex Layouts DataFusion Statistics ([#1967](https://github.com/spiraldb/vortex/pull/1967))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- :Index to Field::Name expr transform (#1894)
- Split StructDType from DType file and restrict ViewedDType visibility to the crate ([#1890](https://github.com/spiraldb/vortex/pull/1890))
- Use flatbuffers::follow instead of init_from_table ([#1878](https://github.com/spiraldb/vortex/pull/1878))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Vortex Layouts - scanner ([#1849](https://github.com/spiraldb/vortex/pull/1849))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- Export DType fields as public ([#1833](https://github.com/spiraldb/vortex/pull/1833))
- Support list in the fuzzer (only for implemented actions) ([#1735](https://github.com/spiraldb/vortex/pull/1735))
- Added a list builder ([#1711](https://github.com/spiraldb/vortex/pull/1711))

## `vortex-datetime-dtype` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.21.1...vortex-datetime-dtype-v0.22.0) - 2025-01-21

### Added
- compare array metadata against checked in goldenfiles (#1812)

## `vortex-error` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.21.1...vortex-error-v0.22.0) - 2025-01-21

### Other
- update Cargo.toml dependencies

## `vortex-buffer` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.21.1...vortex-buffer-v0.22.0) - 2025-01-21

### Other
- A few fixes ([#1973](https://github.com/spiraldb/vortex/pull/1973))
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- Improve vortex-buffer Debug repr ([#1901](https://github.com/spiraldb/vortex/pull/1901))
- Split StructDType from DType file and restrict ViewedDType visibility to the crate ([#1890](https://github.com/spiraldb/vortex/pull/1890))
- Segment Alignment ([#1883](https://github.com/spiraldb/vortex/pull/1883))
- Add Hash to vortex-expr and therefore vortex-scalar ([#1869](https://github.com/spiraldb/vortex/pull/1869))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Trunc buffer in debug display ([#1809](https://github.com/spiraldb/vortex/pull/1809))
- Improve alp decode ([#1764](https://github.com/spiraldb/vortex/pull/1764))
- Do more vortex-buffer operations in terms of T instead of u8 ([#1766](https://github.com/spiraldb/vortex/pull/1766))
- Remove FSST copy ([#1757](https://github.com/spiraldb/vortex/pull/1757))
- Zero-copy BufMessageReader ([#1753](https://github.com/spiraldb/vortex/pull/1753))
- vortex-buffer (#1742)
- Use vortex-buffer over bytes::Bytes ([#1713](https://github.com/spiraldb/vortex/pull/1713))
- move IoBuf to vortex-io (#1714)
- Message Codec ([#1692](https://github.com/spiraldb/vortex/pull/1692))

## `vortex-array` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.21.1...vortex-array-v0.22.0) - 2025-01-21

### Added
- StructDType `from_iter` (#2003)
- teach DateTimePartsArray to cast (#1946)
- Use buffers for opaque data in VarBin and VarBinView (#1935)
- Hold views in a buffer instead of an array (#1933)
- teach VortexExpr to dtype (#1811)
- eagerly compute IsConstant stat (#1838)
- Unify `LazyDType` into `StructDType` (#1826)
- compare array metadata against checked in goldenfiles (#1812)
- impl subtract for primitive_scalar (#1592)
- add BinaryNumericFn for array arithmetic (#1640)

### Fixed
- Slicing empty chunked arrays (#1870)
- FlexBuffer serialization ambiguity for binary/string/list(u8) (#1859)
- binary_numeric is now correct (+ tests!) (#1721)
- list_to_arrow uses proper PType (#1776)
- BitPackedArray ptype changed by compute funcs (#1724)
- better child names (#1722)
- BitPackedArray enforces can only be built over non-negative values (#1705)

### Other
- remove roaring int/bool & run end bool arrays (#2020)
- Do not force compute true count for FilterMask::try_from ([#2024](https://github.com/spiraldb/vortex/pull/2024))
- Support reusing dictionary when encoding values ([#2008](https://github.com/spiraldb/vortex/pull/2008))
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Fix develop conflict ([#1993](https://github.com/spiraldb/vortex/pull/1993))
- LikeFn to take owned ([#1974](https://github.com/spiraldb/vortex/pull/1974))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Only persist certain stats when serializing arrays in FlatLayout ([#1984](https://github.com/spiraldb/vortex/pull/1984))
- Un-arc'd ViewedArrayData ([#1978](https://github.com/spiraldb/vortex/pull/1978))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- Split conjunct into scanner and apply one by one ([#1963](https://github.com/spiraldb/vortex/pull/1963))
- A few fixes ([#1973](https://github.com/spiraldb/vortex/pull/1973))
- Vortex Layouts DataFusion Statistics ([#1967](https://github.com/spiraldb/vortex/pull/1967))
- Arc viewed buffers ([#1970](https://github.com/spiraldb/vortex/pull/1970))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- assorted cleanups (#1953)
- Cache partitioned expressions in StructLayoutReader ([#1947](https://github.com/spiraldb/vortex/pull/1947))
- Cutover to Vortex Layouts ([#1899](https://github.com/spiraldb/vortex/pull/1899))
- Arc InnerArrayData and add a WeakArrayData ([#1930](https://github.com/spiraldb/vortex/pull/1930))
- Remove Field from vortex-expr, replace with FieldName ([#1915](https://github.com/spiraldb/vortex/pull/1915))
- Return nice vorex-y error when trying to compare arrays with Struct dtype ([#1912](https://github.com/spiraldb/vortex/pull/1912))
- Fix ArrayData::into_array_iter ([#1902](https://github.com/spiraldb/vortex/pull/1902))
- Segment Alignment ([#1883](https://github.com/spiraldb/vortex/pull/1883))
- use individual field access for maybe_null_field_by_idx (#1881)
- Fix splits to read empty struct/chunked arrays. ([#1876](https://github.com/spiraldb/vortex/pull/1876))
- Use flatbuffers::follow instead of init_from_table ([#1878](https://github.com/spiraldb/vortex/pull/1878))
- Const-alignment for flatbuffers ([#1868](https://github.com/spiraldb/vortex/pull/1868))
- Vortex Layouts - scanner ([#1849](https://github.com/spiraldb/vortex/pull/1849))
- Struct `field` renamed to `maybe_null_field` ([#1846](https://github.com/spiraldb/vortex/pull/1846))
- Add DynEq trait for VortexExpr and implement PartialEq for VortexExpr ([#1837](https://github.com/spiraldb/vortex/pull/1837))
- Converting LogicalValidity into Validity requires passing Nullability ([#1834](https://github.com/spiraldb/vortex/pull/1834))
- Export DType fields as public ([#1833](https://github.com/spiraldb/vortex/pull/1833))
- Arrays have multiple buffers ([#1743](https://github.com/spiraldb/vortex/pull/1743))
- patch bools (#1760)
- prefer downcast_array_ref (#1786)
- Into arrow with hint ([#1730](https://github.com/spiraldb/vortex/pull/1730))
- ContextRef = Arc<Context> ([#1802](https://github.com/spiraldb/vortex/pull/1802))
- Support list in the fuzzer (only for implemented actions) ([#1735](https://github.com/spiraldb/vortex/pull/1735))
- Revert "Convert into arrow without going via `into_canonical`" ([#1797](https://github.com/spiraldb/vortex/pull/1797))
- Use Validity::from when building validities ([#1793](https://github.com/spiraldb/vortex/pull/1793))
- Add a `from_iter_slow` method to ListArray to allow convenient creation of lists ([#1778](https://github.com/spiraldb/vortex/pull/1778))
- Convert into arrow without going via `into_canonical` ([#1736](https://github.com/spiraldb/vortex/pull/1736))
- :into_canonical extend with trusted len iter (#1767)
- Improve alp decode ([#1764](https://github.com/spiraldb/vortex/pull/1764))
- Zero-copy BufMessageReader ([#1753](https://github.com/spiraldb/vortex/pull/1753))
- update rust-toolchain (#1751)
- vortex-buffer (#1742)
- Added list to the fuzzer ([#1712](https://github.com/spiraldb/vortex/pull/1712))
- Remove null array usage from list view ([#1728](https://github.com/spiraldb/vortex/pull/1728))
- Remove static from NativePType ([#1731](https://github.com/spiraldb/vortex/pull/1731))
- Rename recordbatch to record_batch ([#1727](https://github.com/spiraldb/vortex/pull/1727))
- Into canonical (for list) should use dtype of elements, not inferred type of elements. ([#1726](https://github.com/spiraldb/vortex/pull/1726))
- Use vortex-buffer over bytes::Bytes ([#1713](https://github.com/spiraldb/vortex/pull/1713))
- Fix builder bugs ([#1718](https://github.com/spiraldb/vortex/pull/1718))
- Named child arrays ([#1710](https://github.com/spiraldb/vortex/pull/1710))
- Add debug assertions to ComputeFn results ([#1716](https://github.com/spiraldb/vortex/pull/1716))
- Added a list builder ([#1711](https://github.com/spiraldb/vortex/pull/1711))
- Message Codec ([#1692](https://github.com/spiraldb/vortex/pull/1692))
- use PrimitiveArray::patch in bitpacking take (#1690)

## `vortex-alp` - [0.22.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.21.1...vortex-alp-v0.22.0) - 2025-01-21

### Added
- compare array metadata against checked in goldenfiles (#1812)
- consume right_parts buffer in alp-rd decompression (#1785)

### Fixed
- ALP-RD scalar_at respects nullability (#1864)
- binary_numeric is now correct (+ tests!) (#1721)

### Other
- Add buffers to TryFrom parts ([#1992](https://github.com/spiraldb/vortex/pull/1992))
- Remove allocations when constructing array data ([#1986](https://github.com/spiraldb/vortex/pull/1986))
- Add ValidateVTable ([#1979](https://github.com/spiraldb/vortex/pull/1979))
- FilterMask Optimizations ([#1950](https://github.com/spiraldb/vortex/pull/1950))
- Converting LogicalValidity into Validity requires passing Nullability ([#1834](https://github.com/spiraldb/vortex/pull/1834))
- ALP roundtrips all null arrays ([#1794](https://github.com/spiraldb/vortex/pull/1794))
- Improve alp decode ([#1764](https://github.com/spiraldb/vortex/pull/1764))
- vortex-buffer (#1742)

## `vortex` - [0.22.0](https://github.com/spiraldb/vortex/compare/0.21.1...0.22.0) - 2025-01-21

### Added
- feature flag ExecDriver implementations ([#2005](https://github.com/spiraldb/vortex/pull/2005))

### Other
- remove roaring int/bool & run end bool arrays (#2020)
- Vortex Layouts - Some Cleanup ([#1917](https://github.com/spiraldb/vortex/pull/1917))
- Namespace encodings inside vortex ([#1803](https://github.com/spiraldb/vortex/pull/1803))

##

`vortex-datafusion` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.20.0...vortex-datafusion-v0.21.1) -
2024-12-16

### Added

- Concurrent schema discovery for DataFusion (#1568)
- add into_arrow to IntoCanonicalVTable (#1604)
- Add uncompressed size stat for Vortex/Datafusion (#1512)
- add clickbench benchmark (#1304)
- Layout metadata reader and column statistics (#1455)

### Fixed

- Consistent metadata table column names for DataFusion stats (#1577)
- Fix repartitioning regression (#1564)
- Correct DataFusion repartitioning (#1554)
- support stats for ExtensionArray in vortex-file (#1547)

### Other

- Cache initial reads in `VortexFormat` ([#1633](https://github.com/spiraldb/vortex/pull/1633))
- Propagate size from datafusion to VortexReadBuilder ([#1636](https://github.com/spiraldb/vortex/pull/1636))
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Fix partition count assertion  ([#1597](https://github.com/spiraldb/vortex/pull/1597))
- Pass typed flat buffers into layout builders ([#1563](https://github.com/spiraldb/vortex/pull/1563))
- DF repartition by file sizes instead of number of files ([#1572](https://github.com/spiraldb/vortex/pull/1572))
- simplify MetadataFetcher to function (#1569)
- cleanup in vortex-ipc & vortex-file (#1553)
- cleanups to support wasm32-wasip1 target (#1528)
- Like expression supports negated and case_sensitive arguments ([#1537](https://github.com/spiraldb/vortex/pull/1537))
- Add file grouping-based repartitioning for DataFusion ([#1531](https://github.com/spiraldb/vortex/pull/1531))
- Add LIKE operator ([#1525](https://github.com/spiraldb/vortex/pull/1525))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Report stats to DataFusion ([#1506](https://github.com/spiraldb/vortex/pull/1506))

##

`vortex-roaring` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.20.0...vortex-roaring-v0.21.1) -
2024-12-16

### Other

- Stats are stored as Vec of tuples instead of enummap ([#1658](https://github.com/spiraldb/vortex/pull/1658))
- cleanups to support wasm32-wasip1 target (#1528)
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-zigzag` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.20.0...vortex-zigzag-v0.21.1) -
2024-12-16

### Other

- impl take_fn and filter_fn for ZigZag ([#1665](https://github.com/spiraldb/vortex/pull/1665))
- Zigzag encode/decode reuses the underlying data vec ([#1638](https://github.com/spiraldb/vortex/pull/1638))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-runend-bool` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.20.0...vortex-runend-bool-v0.21.1) -
2024-12-16

### Fixed

- RunEndBool array take respects validity (#1684)
- RunEndBool scalar_at respects array's nullability (#1683)
- use search_sorted_usize when searching for indices (#1566)
- Support slicing RunEndBool arrays to 0 elements (#1511)

### Other

- Stats are stored as Vec of tuples instead of enummap ([#1658](https://github.com/spiraldb/vortex/pull/1658))
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- cleanups to support wasm32-wasip1 target (#1528)
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-runend` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.20.0...vortex-runend-v0.21.1) -
2024-12-16

### Fixed

- RunEnd array correctly encodes all null arrays (#1675)
- RunEnd encoding doesn't skip first value when encoding nullable arrays (#1674)
- use search_sorted_usize when searching for indices (#1566)
- Support slicing RunEndBool arrays to 0 elements (#1511)
- Fix slicing bug in RunEndArray (#1497)

### Other

- RunEnd compare produces canonical array (#1668)
- RunEnd fill_null correctly reconstructs itself (#1666)
- Run end fill null ([#1660](https://github.com/spiraldb/vortex/pull/1660))
- Remove validity from run-end array ([#1630](https://github.com/spiraldb/vortex/pull/1630))
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- cleanups to support wasm32-wasip1 target (#1528)
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-fsst` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.20.0...vortex-fsst-v0.21.1) -
2024-12-16

### Other

- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Ensure patches don't turn arrays nullable ([#1565](https://github.com/spiraldb/vortex/pull/1565))
- cleanups to support wasm32-wasip1 target (#1528)
- Narrow indices types during compression  ([#1558](https://github.com/spiraldb/vortex/pull/1558))
- Remove with_dyn and ArrayDef ([#1503](https://github.com/spiraldb/vortex/pull/1503))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-sampling-compressor` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.20.0...vortex-sampling-compressor-v0.21.1) -
2024-12-16

### Added

- faster Patches::take & use patches in alp-rd & sparse (#1628)
- consistently compress validity (#1544)

### Fixed

- SparseArray scalar_at was broken due to strict PValue PartialOrd (#1575)

### Other

- Cheaper `maybe_from` ([#1677](https://github.com/spiraldb/vortex/pull/1677))
- Remove validity from run-end array ([#1630](https://github.com/spiraldb/vortex/pull/1630))
- Fix regression in search_sorted when Patches replaced
  SparseArray ([#1624](https://github.com/spiraldb/vortex/pull/1624))
- Cannot compress like logs are debug level ([#1620](https://github.com/spiraldb/vortex/pull/1620))
- Patches Utility ([#1601](https://github.com/spiraldb/vortex/pull/1601))
- cleanups to support wasm32-wasip1 target (#1528)
- Add `maybe_from` function to help downcast ArrayData into a specific encoded array without potentially capturing a
  backtrace ([#1560](https://github.com/spiraldb/vortex/pull/1560))
- Narrow indices types during compression  ([#1558](https://github.com/spiraldb/vortex/pull/1558))
- ArrayNBytes includes size of arrays metadata ([#1549](https://github.com/spiraldb/vortex/pull/1549))
- Revert "feat: consistently compress validity" ([#1551](https://github.com/spiraldb/vortex/pull/1551))
- Added a list compressor ([#1536](https://github.com/spiraldb/vortex/pull/1536))
- Add more click bench things ([#1530](https://github.com/spiraldb/vortex/pull/1530))
- Remove with_dyn and ArrayDef ([#1503](https://github.com/spiraldb/vortex/pull/1503))

##

`vortex-ipc` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-ipc-v0.20.0...vortex-ipc-v0.21.1) - 2024-12-16

### Added

- add into_arrow to IntoCanonicalVTable (#1604)

### Other

- IPC Message clean up ([#1686](https://github.com/spiraldb/vortex/pull/1686))
- Cheaper `maybe_from` ([#1677](https://github.com/spiraldb/vortex/pull/1677))
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- cleanup in vortex-ipc & vortex-file (#1553)
- cleanups to support wasm32-wasip1 target (#1528)
- Add `maybe_from` function to help downcast ArrayData into a specific encoded array without potentially capturing a
  backtrace ([#1560](https://github.com/spiraldb/vortex/pull/1560))

## `vortex-io` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-io-v0.20.0...vortex-io-v0.21.1) - 2024-12-16

### Added

- coalesce multiple reads together and don't block on io if there's values available (#1466)
- Layout metadata reader and column statistics (#1455)

### Fixed

- properly gate things by features & test for that (#1494)
- regression for ObjectStoreReadAt (#1483)

### Other

- use cargo-hack and build all valid feature combos (#1653)
- actually run with wasm32-unknown-unknown (#1648)
- Hide underlying channel in the Dispatcher (#1585)
- cleanups to support wasm32-wasip1 target (#1528)

##

`vortex-file` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-file-v0.20.0...vortex-file-v0.21.1) -
2024-12-16

### Added

- Add fill_null compute function (#1590)
- coalesce multiple reads together and don't block on io if there's values available (#1466)
- Add uncompressed size stat for Vortex/Datafusion (#1512)
- prune layouts based on stats (#1485)
- Layout metadata reader and column statistics (#1455)

### Fixed

- Fix length of empty struct arrays (#1673)
- support stats for ExtensionArray in vortex-file (#1547)

### Other

- IPC Message clean up ([#1686](https://github.com/spiraldb/vortex/pull/1686))
- random refactoring/renaming (#1669)
- Cache initial reads in `VortexFormat` ([#1633](https://github.com/spiraldb/vortex/pull/1633))
- Reading stats tables reuses schemas ([#1637](https://github.com/spiraldb/vortex/pull/1637))
- some LayoutReader cleanups (#1623)
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- `vortex-file` crate-level docs adjustment ([#1610](https://github.com/spiraldb/vortex/pull/1610))
- Use FilterMask in RowMask ([#1616](https://github.com/spiraldb/vortex/pull/1616))
- Fix up InnerScalarValue ([#1613](https://github.com/spiraldb/vortex/pull/1613))
- Pass typed flat buffers into layout builders ([#1563](https://github.com/spiraldb/vortex/pull/1563))
- simplify MetadataFetcher to function (#1569)
- cleanup in vortex-ipc & vortex-file (#1553)
- cleanups to support wasm32-wasip1 target (#1528)
- Store present stats as a bitset in metadata of chunked layout and remove inline dtype
  layout ([#1555](https://github.com/spiraldb/vortex/pull/1555))
- Remove dead chunked reader ([#1552](https://github.com/spiraldb/vortex/pull/1552))
- Like expression supports negated and case_sensitive arguments ([#1537](https://github.com/spiraldb/vortex/pull/1537))
- Add LIKE operator ([#1525](https://github.com/spiraldb/vortex/pull/1525))
- Use filter in RowMask::evaluate ([#1515](https://github.com/spiraldb/vortex/pull/1515))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- RowMask uses ConstantArray for all valid and all invalid selections and remove unsafe RowMask
  constructor ([#1495](https://github.com/spiraldb/vortex/pull/1495))
- Remove uses of with_dyn for validity ([#1487](https://github.com/spiraldb/vortex/pull/1487))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-expr` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.20.0...vortex-expr-v0.21.1) -
2024-12-16

### Other

- Reading stats tables reuses schemas ([#1637](https://github.com/spiraldb/vortex/pull/1637))
- Like expression supports negated and case_sensitive arguments ([#1537](https://github.com/spiraldb/vortex/pull/1537))
- Add LIKE operator ([#1525](https://github.com/spiraldb/vortex/pull/1525))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))

##

`vortex-dict` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.20.0...vortex-dict-v0.21.1) -
2024-12-16

### Added

- Add fill_null compute function (#1590)

### Other

- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- cleanups to support wasm32-wasip1 target (#1528)
- Like expression supports negated and case_sensitive arguments ([#1537](https://github.com/spiraldb/vortex/pull/1537))
- Add LIKE operator ([#1525](https://github.com/spiraldb/vortex/pull/1525))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))
- Remove uses of with_dyn for validity ([#1487](https://github.com/spiraldb/vortex/pull/1487))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-datetime-parts` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.20.0...vortex-datetime-parts-v0.21.1) -
2024-12-16

### Fixed

- support stats for ExtensionArray in vortex-file (#1547)

### Other

- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Fix date time parts scalar_at to cast ([#1584](https://github.com/spiraldb/vortex/pull/1584))
- Narrow indices types during compression  ([#1558](https://github.com/spiraldb/vortex/pull/1558))
- Use filter in RowMask::evaluate ([#1515](https://github.com/spiraldb/vortex/pull/1515))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Remove uses of with_dyn for validity ([#1487](https://github.com/spiraldb/vortex/pull/1487))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-bytebool` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.20.0...vortex-bytebool-v0.21.1) -
2024-12-16

### Other

- Stats are stored as Vec of tuples instead of enummap ([#1658](https://github.com/spiraldb/vortex/pull/1658))
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-fastlanes` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.20.0...vortex-fastlanes-v0.21.1) -
2024-12-16

### Added

- Faster bitpacked filter & take (#1667)

### Other

- Fix regression in search_sorted when Patches replaced
  SparseArray ([#1624](https://github.com/spiraldb/vortex/pull/1624))
- Skip FoR decompression if min and scalar are 0 ([#1618](https://github.com/spiraldb/vortex/pull/1618))
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Patches Utility ([#1601](https://github.com/spiraldb/vortex/pull/1601))
- Fix search sorted casting ([#1579](https://github.com/spiraldb/vortex/pull/1579))
- cleanups to support wasm32-wasip1 target (#1528)
- Add `maybe_from` function to help downcast ArrayData into a specific encoded array without potentially capturing a
  backtrace ([#1560](https://github.com/spiraldb/vortex/pull/1560))
- Remove with_dyn and ArrayDef ([#1503](https://github.com/spiraldb/vortex/pull/1503))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Remove uses of with_dyn for validity ([#1487](https://github.com/spiraldb/vortex/pull/1487))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-scalar` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.20.0...vortex-scalar-v0.21.1) -
2024-12-16

### Fixed

- ScalarValue flatbuffer serde doesn't perform redundant copy (#1635)
- SparseArray scalar_at was broken due to strict PValue PartialOrd (#1575)

### Other

- Some small fixes ([#1631](https://github.com/spiraldb/vortex/pull/1631))
- Fix up InnerScalarValue ([#1613](https://github.com/spiraldb/vortex/pull/1613))
- cleanups to support wasm32-wasip1 target (#1528)
- Narrow indices types during compression  ([#1558](https://github.com/spiraldb/vortex/pull/1558))
- Array Builders ([#1543](https://github.com/spiraldb/vortex/pull/1543))

##

`vortex-flatbuffers` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.20.0...vortex-flatbuffers-v0.21.1) -
2024-12-16

### Other

- IPC Message clean up ([#1686](https://github.com/spiraldb/vortex/pull/1686))

##

`vortex-error` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-error-v0.20.0...vortex-error-v0.21.1) -
2024-12-16

### Added

- faster Patches::take & use patches in alp-rd & sparse (#1628)

### Other

- cleanups to support wasm32-wasip1 target (#1528)

##

`vortex-dtype` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.20.0...vortex-dtype-v0.21.1) -
2024-12-16

### Added

- patches uses a map in some cases (#1626)

### Other

- IPC Message clean up ([#1686](https://github.com/spiraldb/vortex/pull/1686))
- Move PrimitiveBuilder constraints to where clause ([#1634](https://github.com/spiraldb/vortex/pull/1634))
- Reading stats tables reuses schemas ([#1637](https://github.com/spiraldb/vortex/pull/1637))
- Revert "feat: patches uses a map in some cases" ([#1629](https://github.com/spiraldb/vortex/pull/1629))
- fuzzer reference take implementation respects generated values nullability (#1586)
- cleanups to support wasm32-wasip1 target (#1528)
- Narrow indices types during compression  ([#1558](https://github.com/spiraldb/vortex/pull/1558))
- Add ArrowPrimitiveType ([#1540](https://github.com/spiraldb/vortex/pull/1540))
- Add lists to vortex ([#1524](https://github.com/spiraldb/vortex/pull/1524))

##

`vortex-buffer` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.20.0...vortex-buffer-v0.21.1) -
2024-12-16

### Added

- Layout metadata reader and column statistics (#1455)

##

`vortex-array` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-array-v0.20.0...vortex-array-v0.21.1) -
2024-12-16

### Added

- faster Patches::take & use patches in alp-rd & sparse (#1628)
- patches uses a map in some cases (#1626)
- add into_arrow to IntoCanonicalVTable (#1604)
- Add fill_null compute function (#1590)

### Fixed

- RunEndBool array take respects validity (#1684)
- Fix length of empty struct arrays (#1673)
- Consistent metadata table column names for DataFusion stats (#1577)
- SparseArray scalar_at was broken due to strict PValue PartialOrd (#1575)
- use search_sorted_usize when searching for indices (#1566)
- support stats for ExtensionArray in vortex-file (#1547)
- properly gate things by features & test for that (#1494)

### Other

- IPC Message clean up ([#1686](https://github.com/spiraldb/vortex/pull/1686))
- Cheaper `maybe_from` ([#1677](https://github.com/spiraldb/vortex/pull/1677))
- simplify StatsSet (#1672)
- ChunkedArray stats compute handles ordered stats and doesn't eagerly merge chunk
  stats ([#1652](https://github.com/spiraldb/vortex/pull/1652))
- Run end fill null ([#1660](https://github.com/spiraldb/vortex/pull/1660))
- Stats are stored as Vec of tuples instead of enummap ([#1658](https://github.com/spiraldb/vortex/pull/1658))
- VarBin to arrow conversion uses reinterpret cast to convert offsets to
  I32/64 ([#1644](https://github.com/spiraldb/vortex/pull/1644))
- Add missing overrides for into_arrow delegation for
  VarBinArray ([#1642](https://github.com/spiraldb/vortex/pull/1642))
- Move PrimitiveBuilder constraints to where clause ([#1634](https://github.com/spiraldb/vortex/pull/1634))
- Reading stats tables reuses schemas ([#1637](https://github.com/spiraldb/vortex/pull/1637))
- Remove validity from run-end array ([#1630](https://github.com/spiraldb/vortex/pull/1630))
- Some small fixes ([#1631](https://github.com/spiraldb/vortex/pull/1631))
- Revert "feat: patches uses a map in some cases" ([#1629](https://github.com/spiraldb/vortex/pull/1629))
- Fix regression in search_sorted when Patches replaced
  SparseArray ([#1624](https://github.com/spiraldb/vortex/pull/1624))
- search_sorted_many handles failed downcasts and use search_sorted_usize_many in Patches::take (#1621)
- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Use FilterMask in RowMask ([#1616](https://github.com/spiraldb/vortex/pull/1616))
- Patches Utility ([#1601](https://github.com/spiraldb/vortex/pull/1601))
- Fix up InnerScalarValue ([#1613](https://github.com/spiraldb/vortex/pull/1613))
- Count into_canonical invocations ([#1615](https://github.com/spiraldb/vortex/pull/1615))
- Deref scalar buffer once ([#1608](https://github.com/spiraldb/vortex/pull/1608))
- fill_null fallback will not infinitely recurse (#1607)
- Fallback fill_null to canonical array ([#1600](https://github.com/spiraldb/vortex/pull/1600))
- Fix search sorted casting ([#1579](https://github.com/spiraldb/vortex/pull/1579))
- implement binary_boolean for chunked encoding ([#1532](https://github.com/spiraldb/vortex/pull/1532))
- Fuzz filter implementation respects data validity ([#1573](https://github.com/spiraldb/vortex/pull/1573))
- Ensure patches don't turn arrays nullable ([#1565](https://github.com/spiraldb/vortex/pull/1565))
- cleanups to support wasm32-wasip1 target (#1528)
- Add `maybe_from` function to help downcast ArrayData into a specific encoded array without potentially capturing a
  backtrace ([#1560](https://github.com/spiraldb/vortex/pull/1560))
- Narrow indices types during compression  ([#1558](https://github.com/spiraldb/vortex/pull/1558))
- Store present stats as a bitset in metadata of chunked layout and remove inline dtype
  layout ([#1555](https://github.com/spiraldb/vortex/pull/1555))
- ArrayNBytes includes size of arrays metadata ([#1549](https://github.com/spiraldb/vortex/pull/1549))
- Array Builders ([#1543](https://github.com/spiraldb/vortex/pull/1543))
- Like expression supports negated and case_sensitive arguments ([#1537](https://github.com/spiraldb/vortex/pull/1537))
- Add ArrowPrimitiveType ([#1540](https://github.com/spiraldb/vortex/pull/1540))
- Add lists to vortex ([#1524](https://github.com/spiraldb/vortex/pull/1524))
- Add LIKE operator ([#1525](https://github.com/spiraldb/vortex/pull/1525))
- Use filter in RowMask::evaluate ([#1515](https://github.com/spiraldb/vortex/pull/1515))
- Remove with_dyn and ArrayDef ([#1503](https://github.com/spiraldb/vortex/pull/1503))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Short-circuit BoolArray null count ([#1509](https://github.com/spiraldb/vortex/pull/1509))
- Move invert from BoolArrayTrait to InvertFn ([#1490](https://github.com/spiraldb/vortex/pull/1490))
- Remove uses of with_dyn for validity ([#1487](https://github.com/spiraldb/vortex/pull/1487))
- Make BinaryBooleanFn consistent with CompareFn ([#1488](https://github.com/spiraldb/vortex/pull/1488))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

##

`vortex-alp` - [0.21.1](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.20.0...vortex-alp-v0.21.1) - 2024-12-16

### Added

- faster Patches::take & use patches in alp-rd & sparse (#1628)

### Other

- add an unsafe take_unchecked to TakeFn for bounds check ellision (#1611)
- Patches Utility ([#1601](https://github.com/spiraldb/vortex/pull/1601))
- Speed up ALP decompress ([#1614](https://github.com/spiraldb/vortex/pull/1614))
- Remove ALP compare ([#1603](https://github.com/spiraldb/vortex/pull/1603))
- cleanups to support wasm32-wasip1 target (#1528)
- Add `maybe_from` function to help downcast ArrayData into a specific encoded array without potentially capturing a
  backtrace ([#1560](https://github.com/spiraldb/vortex/pull/1560))
- ALP Mask clone ([#1522](https://github.com/spiraldb/vortex/pull/1522))
- Variants VTable ([#1501](https://github.com/spiraldb/vortex/pull/1501))
- Remove uses of with_dyn for validity ([#1487](https://github.com/spiraldb/vortex/pull/1487))
- Flatten unary compute mod ([#1489](https://github.com/spiraldb/vortex/pull/1489))

## `vortex` - [0.21.1](https://github.com/spiraldb/vortex/compare/0.20.0...0.21.1) - 2024-12-16

### Other

- cleanups to support wasm32-wasip1 target (#1528)

##

`vortex-datafusion` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.19.0...vortex-datafusion-v0.20.0) -
2024-11-26

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- :read_selection uses immutable reference ([#1295](https://github.com/spiraldb/vortex/pull/1295))
- Move dispatcher into vortex io from vortex file ([#1385](https://github.com/spiraldb/vortex/pull/1385))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Change Datafusion integration to FileFormat instead of a
  TableProvider ([#1364](https://github.com/spiraldb/vortex/pull/1364))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-zigzag` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.19.0...vortex-zigzag-v0.20.0) -
2024-11-26

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))

##

`vortex-runend-bool` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.19.0...vortex-runend-bool-v0.20.0) -
2024-11-26

### Added

- run end bool compressor ([#1355](https://github.com/spiraldb/vortex/pull/1355))

### Fixed

- RunEndBool stats and slice accounts for offsets ([#1428](https://github.com/spiraldb/vortex/pull/1428))
- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-runend` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.19.0...vortex-runend-v0.20.0) -
2024-11-26

### Added

- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- SearchSorted Many Side ([#1427](https://github.com/spiraldb/vortex/pull/1427))
- CompareFn VTable ([#1426](https://github.com/spiraldb/vortex/pull/1426))
- Remove MaybeCompare and arrow-compatible compare impls ([#1418](https://github.com/spiraldb/vortex/pull/1418))
- Search sorted usize ([#1410](https://github.com/spiraldb/vortex/pull/1410))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- RunEnd compare preserves offset ([#1387](https://github.com/spiraldb/vortex/pull/1387))
- Fix RunEndArray filter ([#1380](https://github.com/spiraldb/vortex/pull/1380))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Support RunEnd array with bool values ([#1365](https://github.com/spiraldb/vortex/pull/1365))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Implement filter for RunEnd array ([#1342](https://github.com/spiraldb/vortex/pull/1342))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Push-down compare function for Dictionary and RunEnd ([#1339](https://github.com/spiraldb/vortex/pull/1339))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-roaring` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.19.0...vortex-roaring-v0.20.0) -
2024-11-26

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))
- Bool arrays with one value and rest being nulls are not
  constant ([#1360](https://github.com/spiraldb/vortex/pull/1360))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-fsst` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.19.0...vortex-fsst-v0.20.0) -
2024-11-26

### Added

- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- CompareFn VTable ([#1426](https://github.com/spiraldb/vortex/pull/1426))
- Remove MaybeCompare and arrow-compatible compare impls ([#1418](https://github.com/spiraldb/vortex/pull/1418))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))

##

`vortex-sampling-compressor` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.19.0...vortex-sampling-compressor-v0.20.0) -
2024-11-26

### Added

- split computations of stats for VarBin & VarBinView ([#1457](https://github.com/spiraldb/vortex/pull/1457))
- run end bool compressor ([#1355](https://github.com/spiraldb/vortex/pull/1355))
- add stat for uncompressed size in bytes ([#1315](https://github.com/spiraldb/vortex/pull/1315))

### Fixed

- FSST compress-like child indices ([#1480](https://github.com/spiraldb/vortex/pull/1480))
- compress_noci benchmark broken on develop ([#1450](https://github.com/spiraldb/vortex/pull/1450))
- CompressionTrees diverge from the actual array children  ([#1430](https://github.com/spiraldb/vortex/pull/1430))
- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Hash and PartialEq EncodingId only by the numeric value ([#1391](https://github.com/spiraldb/vortex/pull/1391))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Support RunEnd array with bool values ([#1365](https://github.com/spiraldb/vortex/pull/1365))

##

`vortex-ipc` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-ipc-v0.19.0...vortex-ipc-v0.20.0) - 2024-11-26

### Added

- VortexFileWriter is Send ([#1479](https://github.com/spiraldb/vortex/pull/1479))
- eliminate VortexRead, replace with struct VortexBufReader ([#1349](https://github.com/spiraldb/vortex/pull/1349))

### Fixed

- restore reading of inline dtype layout ([#1442](https://github.com/spiraldb/vortex/pull/1442))

### Other

- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Move ArrayData into a module ([#1370](https://github.com/spiraldb/vortex/pull/1370))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))

## `vortex-io` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-io-v0.19.0...vortex-io-v0.20.0) - 2024-11-26

### Added

- implement SizeLimitedStream for backpressure ([#1477](https://github.com/spiraldb/vortex/pull/1477))
- add optional instrumentation to readers ([#1431](https://github.com/spiraldb/vortex/pull/1431))
- eliminate VortexRead, replace with struct VortexBufReader ([#1349](https://github.com/spiraldb/vortex/pull/1349))

### Fixed

- allocate aligned buffers for VortexReadAt impls ([#1456](https://github.com/spiraldb/vortex/pull/1456))

### Other

- Make the VortexReadAt::size method return an io::Result ([#1471](https://github.com/spiraldb/vortex/pull/1471))
- Update name of thread and set max blocking threads to once ([#1419](https://github.com/spiraldb/vortex/pull/1419))
- Move dispatcher into vortex io from vortex file ([#1385](https://github.com/spiraldb/vortex/pull/1385))

##

`vortex-file` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-file-v0.19.0...vortex-file-v0.20.0) -
2024-11-26

### Added

- VortexFileWriter is Send ([#1479](https://github.com/spiraldb/vortex/pull/1479))
- support Identity in pruner ([#1441](https://github.com/spiraldb/vortex/pull/1441))
- RowFilter is a valid pruning predicate ([#1438](https://github.com/spiraldb/vortex/pull/1438))
- teach PruningPredicate to evaluate itself against a stats
  table ([#1436](https://github.com/spiraldb/vortex/pull/1436))
- add optional instrumentation to readers ([#1431](https://github.com/spiraldb/vortex/pull/1431))
- teach ChunkedLayout how to read metadata ([#1399](https://github.com/spiraldb/vortex/pull/1399))
- don't write leading/trailing zero histograms into file ([#1372](https://github.com/spiraldb/vortex/pull/1372))
- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))
- eliminate VortexRead, replace with struct VortexBufReader ([#1349](https://github.com/spiraldb/vortex/pull/1349))

### Fixed

- allocate aligned buffers for VortexReadAt impls ([#1456](https://github.com/spiraldb/vortex/pull/1456))
- restore reading of inline dtype layout ([#1442](https://github.com/spiraldb/vortex/pull/1442))
- required stats are relations not maps ([#1432](https://github.com/spiraldb/vortex/pull/1432))
- Stop producing empty row masks in chunked reader ([#1429](https://github.com/spiraldb/vortex/pull/1429))
- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Make the VortexReadAt::size method return an io::Result ([#1471](https://github.com/spiraldb/vortex/pull/1471))
- Remove array iterators ([#1451](https://github.com/spiraldb/vortex/pull/1451))
- :read_selection uses immutable reference ([#1295](https://github.com/spiraldb/vortex/pull/1295))
- introduce not_prunable ([#1435](https://github.com/spiraldb/vortex/pull/1435))
- test filter conditions interacting with chunks ([#1400](https://github.com/spiraldb/vortex/pull/1400))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Reset ColumnarBatchReader state when short circuiting ([#1386](https://github.com/spiraldb/vortex/pull/1386))
- Move dispatcher into vortex io from vortex file ([#1385](https://github.com/spiraldb/vortex/pull/1385))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Change Datafusion integration to FileFormat instead of a
  TableProvider ([#1364](https://github.com/spiraldb/vortex/pull/1364))
- Use SplitIterator in layout tests ([#1363](https://github.com/spiraldb/vortex/pull/1363))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- RowMasks use bitmasks instead of bitmaps ([#1346](https://github.com/spiraldb/vortex/pull/1346))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-expr` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.19.0...vortex-expr-v0.20.0) -
2024-11-26

### Added

- support Identity in pruner ([#1441](https://github.com/spiraldb/vortex/pull/1441))

### Other

- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Change Datafusion integration to FileFormat instead of a
  TableProvider ([#1364](https://github.com/spiraldb/vortex/pull/1364))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-dict` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.19.0...vortex-dict-v0.20.0) -
2024-11-26

### Added

- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))

### Fixed

- CompressionTrees diverge from the actual array children  ([#1430](https://github.com/spiraldb/vortex/pull/1430))
- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Implement NBytes only using visitor ([#1449](https://github.com/spiraldb/vortex/pull/1449))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- CompareFn VTable ([#1426](https://github.com/spiraldb/vortex/pull/1426))
- Remove MaybeCompare and arrow-compatible compare impls ([#1418](https://github.com/spiraldb/vortex/pull/1418))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Push-down compare function for Dictionary and RunEnd ([#1339](https://github.com/spiraldb/vortex/pull/1339))

##

`vortex-datetime-parts` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.19.0...vortex-datetime-parts-v0.20.0) -
2024-11-26

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-bytebool` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.19.0...vortex-bytebool-v0.20.0) -
2024-11-26

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- FillForward VTable ([#1405](https://github.com/spiraldb/vortex/pull/1405))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-fastlanes` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.19.0...vortex-fastlanes-v0.20.0) -
2024-11-26

### Added

- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))

### Fixed

- CompressionTrees diverge from the actual array children  ([#1430](https://github.com/spiraldb/vortex/pull/1430))
- BitPackedArray filter correctly stores fully unpacked chunks ([#1393](https://github.com/spiraldb/vortex/pull/1393))
- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Implement NBytes only using visitor ([#1449](https://github.com/spiraldb/vortex/pull/1449))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- SearchSorted Many Side ([#1427](https://github.com/spiraldb/vortex/pull/1427))
- SearchSorted VTable ([#1414](https://github.com/spiraldb/vortex/pull/1414))
- Search sorted usize ([#1410](https://github.com/spiraldb/vortex/pull/1410))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- Hash and PartialEq EncodingId only by the numeric value ([#1391](https://github.com/spiraldb/vortex/pull/1391))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Implement FilterFn for BitPackedArray ([#1356](https://github.com/spiraldb/vortex/pull/1356))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-scalar` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.19.0...vortex-scalar-v0.20.0) -
2024-11-26

### Other

- PValue PartialEq uses NativePType aware equality ([#1374](https://github.com/spiraldb/vortex/pull/1374))

##

`vortex-flatbuffers` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.19.0...vortex-flatbuffers-v0.20.0) -
2024-11-26

### Added

- add stat for uncompressed size in bytes ([#1315](https://github.com/spiraldb/vortex/pull/1315))

##

`vortex-dtype` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.19.0...vortex-dtype-v0.20.0) -
2024-11-26

### Other

- :read_selection uses immutable reference ([#1295](https://github.com/spiraldb/vortex/pull/1295))

##

`vortex-buffer` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.19.0...vortex-buffer-v0.20.0) -
2024-11-26

### Added

- VortexFileWriter is Send ([#1479](https://github.com/spiraldb/vortex/pull/1479))

### Other

- Always zero-copy from VortexBuffer to ArrowBuffer ([#1348](https://github.com/spiraldb/vortex/pull/1348))

##

`vortex-array` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.19.0...vortex-array-v0.20.0) -
2024-11-26

### Added

- split computations of stats for VarBin & VarBinView ([#1457](https://github.com/spiraldb/vortex/pull/1457))
- don't write leading/trailing zero histograms into file ([#1372](https://github.com/spiraldb/vortex/pull/1372))
- add stat for uncompressed size in bytes ([#1315](https://github.com/spiraldb/vortex/pull/1315))
- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))

### Fixed

- CompressionTrees diverge from the actual array children  ([#1430](https://github.com/spiraldb/vortex/pull/1430))
- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))
- Bool arrays with one value and rest being nulls are not
  constant ([#1360](https://github.com/spiraldb/vortex/pull/1360))

### Other

- Use array len as denominator for selectivity ([#1468](https://github.com/spiraldb/vortex/pull/1468))
- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove array iterators ([#1451](https://github.com/spiraldb/vortex/pull/1451))
- Implement NBytes only using visitor ([#1449](https://github.com/spiraldb/vortex/pull/1449))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- SearchSorted Many Side ([#1427](https://github.com/spiraldb/vortex/pull/1427))
- CompareFn VTable ([#1426](https://github.com/spiraldb/vortex/pull/1426))
- Subtract scalar VTable ([#1422](https://github.com/spiraldb/vortex/pull/1422))
- Remove MaybeCompare and arrow-compatible compare impls ([#1418](https://github.com/spiraldb/vortex/pull/1418))
- SearchSorted VTable ([#1414](https://github.com/spiraldb/vortex/pull/1414))
- Binary Boolean VTable ([#1407](https://github.com/spiraldb/vortex/pull/1407))
- Search sorted usize ([#1410](https://github.com/spiraldb/vortex/pull/1410))
- FillForward VTable ([#1405](https://github.com/spiraldb/vortex/pull/1405))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- test filter conditions interacting with chunks ([#1400](https://github.com/spiraldb/vortex/pull/1400))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- Cast VTable ([#1397](https://github.com/spiraldb/vortex/pull/1397))
- Fix metadata printing ([#1392](https://github.com/spiraldb/vortex/pull/1392))
- Hash and PartialEq EncodingId only by the numeric value ([#1391](https://github.com/spiraldb/vortex/pull/1391))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Load metadata in ViewedArrayData ([#1383](https://github.com/spiraldb/vortex/pull/1383))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Move ArrayData into a module ([#1370](https://github.com/spiraldb/vortex/pull/1370))
- Change Datafusion integration to FileFormat instead of a
  TableProvider ([#1364](https://github.com/spiraldb/vortex/pull/1364))
- Support RunEnd array with bool values ([#1365](https://github.com/spiraldb/vortex/pull/1365))
- Implement FilterFn for BitPackedArray ([#1356](https://github.com/spiraldb/vortex/pull/1356))
- Enable Clippy redundant clone check ([#1361](https://github.com/spiraldb/vortex/pull/1361))
- Avoid unnecessary backtrace generation ([#1353](https://github.com/spiraldb/vortex/pull/1353))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Implement VarBinView take using Buffer selection ([#1344](https://github.com/spiraldb/vortex/pull/1344))
- Fix compression benchmarks ([#1345](https://github.com/spiraldb/vortex/pull/1345))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Push-down compare function for Dictionary and RunEnd ([#1339](https://github.com/spiraldb/vortex/pull/1339))
- Remove primitive compare impl ([#1337](https://github.com/spiraldb/vortex/pull/1337))
- Use arrow scalars for cmp where possible ([#1334](https://github.com/spiraldb/vortex/pull/1334))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

##

`vortex-alp` - [0.20.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.19.0...vortex-alp-v0.20.0) - 2024-11-26

### Added

- cache FilterMask iterators ([#1351](https://github.com/spiraldb/vortex/pull/1351))

### Fixed

- Remove redundant len/is_empty implementations on typed array
  structs ([#1384](https://github.com/spiraldb/vortex/pull/1384))

### Other

- Validity VTable ([#1454](https://github.com/spiraldb/vortex/pull/1454))
- Remove array iterators ([#1451](https://github.com/spiraldb/vortex/pull/1451))
- Remove ArrayCompute ([#1446](https://github.com/spiraldb/vortex/pull/1446))
- Visitor VTable ([#1445](https://github.com/spiraldb/vortex/pull/1445))
- StatsCompute VTable ([#1434](https://github.com/spiraldb/vortex/pull/1434))
- CompareFn VTable ([#1426](https://github.com/spiraldb/vortex/pull/1426))
- Remove MaybeCompare and arrow-compatible compare impls ([#1418](https://github.com/spiraldb/vortex/pull/1418))
- ScalarAt VTable ([#1404](https://github.com/spiraldb/vortex/pull/1404))
- Take VTable ([#1401](https://github.com/spiraldb/vortex/pull/1401))
- Slice VTable ([#1398](https://github.com/spiraldb/vortex/pull/1398))
- Hash and PartialEq EncodingId only by the numeric value ([#1391](https://github.com/spiraldb/vortex/pull/1391))
- FilterFn vtable ([#1390](https://github.com/spiraldb/vortex/pull/1390))
- Remove TypedArray, make InnerArrayData non-pub ([#1378](https://github.com/spiraldb/vortex/pull/1378))
- Add helper function to unpack constant scalar out of array ([#1373](https://github.com/spiraldb/vortex/pull/1373))
- Remove as_primitive ([#1376](https://github.com/spiraldb/vortex/pull/1376))
- Filter mask ([#1327](https://github.com/spiraldb/vortex/pull/1327))
- Add TakeOptions to skip bounds checking ([#1343](https://github.com/spiraldb/vortex/pull/1343))
- Use enum map for stats instead of HashMap ([#1341](https://github.com/spiraldb/vortex/pull/1341))
- Remove BoolArray::from_vec ([#1332](https://github.com/spiraldb/vortex/pull/1332))

## `vortex` - [0.20.0](https://github.com/spiraldb/vortex/compare/0.19.0...0.20.0) - 2024-11-26

### Other

- :read_selection uses immutable reference ([#1295](https://github.com/spiraldb/vortex/pull/1295))
- Move dispatcher into vortex io from vortex file ([#1385](https://github.com/spiraldb/vortex/pull/1385))

##

`vortex-ipc` - [0.19.0](https://github.com/spiraldb/vortex/compare/vortex-ipc-v0.18.1...vortex-ipc-v0.19.0) - 2024-11-15

### Added

- return Bytes from readers ([#1330](https://github.com/spiraldb/vortex/pull/1330))

## `vortex-io` - [0.19.0](https://github.com/spiraldb/vortex/compare/vortex-io-v0.18.1...vortex-io-v0.19.0) - 2024-11-15

### Added

- return Bytes from readers ([#1330](https://github.com/spiraldb/vortex/pull/1330))

##

`vortex-file` - [0.19.0](https://github.com/spiraldb/vortex/compare/vortex-file-v0.18.1...vortex-file-v0.19.0) -
2024-11-15

### Added

- return Bytes from readers ([#1330](https://github.com/spiraldb/vortex/pull/1330))

### Other

- Rename Array -> ArrayData ([#1316](https://github.com/spiraldb/vortex/pull/1316))
- Restore perfoamnce of filter bitmask to bitmap conversion for dense
  rowmasks ([#1302](https://github.com/spiraldb/vortex/pull/1302))
- Reuse the IoDispatcher across DataFusion instances ([#1299](https://github.com/spiraldb/vortex/pull/1299))
- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-file` - [0.18.1](https://github.com/spiraldb/vortex/compare/vortex-file-v0.18.0...vortex-file-v0.18.1) -
2024-11-15

### Other

- Rename Array -> ArrayData ([#1316](https://github.com/spiraldb/vortex/pull/1316))
- Restore perfoamnce of filter bitmask to bitmap conversion for dense
  rowmasks ([#1302](https://github.com/spiraldb/vortex/pull/1302))
- Reuse the IoDispatcher across DataFusion instances ([#1299](https://github.com/spiraldb/vortex/pull/1299))
- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-flatbuffers` - [0.18.1](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.18.0...vortex-flatbuffers-v0.18.1) -
2024-11-15

### Fixed

- use correct feature flags for vortex-ipc -> vortex-fbs dep ([#1328](https://github.com/spiraldb/vortex/pull/1328))

##

`vortex-file` - [0.18.0](https://github.com/spiraldb/vortex/compare/vortex-file-v0.17.0...vortex-file-v0.18.0) -
2024-11-15

### Other

- Rename Array -> ArrayData ([#1316](https://github.com/spiraldb/vortex/pull/1316))
- Restore perfoamnce of filter bitmask to bitmap conversion for dense
  rowmasks ([#1302](https://github.com/spiraldb/vortex/pull/1302))
- Reuse the IoDispatcher across DataFusion instances ([#1299](https://github.com/spiraldb/vortex/pull/1299))
- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-datetime-parts` - [0.18.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.17.0...vortex-datetime-parts-v0.18.0) -
2024-11-15

### Other

- Canonicalize constant array with extension dtype ([#1322](https://github.com/spiraldb/vortex/pull/1322))

##

`vortex-array` - [0.18.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.17.0...vortex-array-v0.18.0) -
2024-11-15

### Other

- Canonicalize constant array with extension dtype ([#1322](https://github.com/spiraldb/vortex/pull/1322))

## `vortex-file` - [0.17.0](https://github.com/spiraldb/vortex/releases/tag/vortex-file-v0.17.0) - 2024-11-15

### Other

- Rename Array -> ArrayData ([#1316](https://github.com/spiraldb/vortex/pull/1316))
- Restore perfoamnce of filter bitmask to bitmap conversion for dense
  rowmasks ([#1302](https://github.com/spiraldb/vortex/pull/1302))
- Reuse the IoDispatcher across DataFusion instances ([#1299](https://github.com/spiraldb/vortex/pull/1299))
- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-datafusion` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.16.0...vortex-datafusion-v0.17.0) -
2024-11-15

### Other

- Shuffle datafusion provider ([#1312](https://github.com/spiraldb/vortex/pull/1312))
- Reuse the IoDispatcher across DataFusion instances ([#1299](https://github.com/spiraldb/vortex/pull/1299))
- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))
- introduce ExprRef, teach expressions new_ref ([#1258](https://github.com/spiraldb/vortex/pull/1258))

##

`vortex-zigzag` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.16.0...vortex-zigzag-v0.17.0) -
2024-11-15

### Added

- split computation of primitive statistics ([#1306](https://github.com/spiraldb/vortex/pull/1306))
- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-runend-bool` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.16.0...vortex-runend-bool-v0.17.0) -
2024-11-15

### Added

- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-runend` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.16.0...vortex-runend-v0.17.0) -
2024-11-15

### Added

- split computation of primitive statistics ([#1306](https://github.com/spiraldb/vortex/pull/1306))
- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-roaring` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.16.0...vortex-roaring-v0.17.0) -
2024-11-15

### Added

- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-sampling-compressor` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.16.0...vortex-sampling-compressor-v0.17.0) -
2024-11-15

### Added

- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

## `vortex-io` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-io-v0.16.0...vortex-io-v0.17.0) - 2024-11-15

### Fixed

- update setup instructions (rye -> uv) ([#1176](https://github.com/spiraldb/vortex/pull/1176))
- fix docs badge in readme ([#753](https://github.com/spiraldb/vortex/pull/753))

### Other

- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))
- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))
- very small README.md fixes
- More README.md improvements ([#1084](https://github.com/spiraldb/vortex/pull/1084))
- Update README.md ([#1055](https://github.com/spiraldb/vortex/pull/1055))
- minor addition to README ([#1030](https://github.com/spiraldb/vortex/pull/1030))
- updated README ([#876](https://github.com/spiraldb/vortex/pull/876))
- release to Test PyPI on each push to version tags ([#760](https://github.com/spiraldb/vortex/pull/760))
- Run ETE benchmarks with MiMalloc and leave a note encouraging its
  usage ([#399](https://github.com/spiraldb/vortex/pull/399))
- README updates ([#394](https://github.com/spiraldb/vortex/pull/394))
- Download flatc instead of building it from source ([#374](https://github.com/spiraldb/vortex/pull/374))
- Update README.md ([#337](https://github.com/spiraldb/vortex/pull/337))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Add note to readme about git submodules and zig version ([#176](https://github.com/spiraldb/vortex/pull/176))
- acknowledgments ([#171](https://github.com/spiraldb/vortex/pull/171))
- Update README.md ([#168](https://github.com/spiraldb/vortex/pull/168))
- More README updates ([#140](https://github.com/spiraldb/vortex/pull/140))
- Update README.md
- readme improvements ([#137](https://github.com/spiraldb/vortex/pull/137))
- README ([#102](https://github.com/spiraldb/vortex/pull/102))
- Root project is vortex-array ([#67](https://github.com/spiraldb/vortex/pull/67))
- Add minimal description to readme and fixup cargo metadata ([#30](https://github.com/spiraldb/vortex/pull/30))
- Add Readme

##

`vortex-file` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-file-v0.16.0...vortex-file-v0.17.0) -
2024-11-15

### Other

- Restore perfoamnce of filter bitmask to bitmap conversion for dense
  rowmasks ([#1302](https://github.com/spiraldb/vortex/pull/1302))
- Reuse the IoDispatcher across DataFusion instances ([#1299](https://github.com/spiraldb/vortex/pull/1299))
- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-expr` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.16.0...vortex-expr-v0.17.0) -
2024-11-15

### Added

- teach VortexExpr to Display ([#1293](https://github.com/spiraldb/vortex/pull/1293))

### Other

- introduce ExprRef, teach expressions new_ref ([#1258](https://github.com/spiraldb/vortex/pull/1258))
- Use itertools format for VortexExpr Display ([#1294](https://github.com/spiraldb/vortex/pull/1294))

##

`vortex-dict` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.16.0...vortex-dict-v0.17.0) -
2024-11-15

### Added

- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-datetime-parts` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.16.0...vortex-datetime-parts-v0.17.0) -
2024-11-15

### Added

- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-fastlanes` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.16.0...vortex-fastlanes-v0.17.0) -
2024-11-15

### Added

- split computation of primitive statistics ([#1306](https://github.com/spiraldb/vortex/pull/1306))
- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))

##

`vortex-scalar` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.16.0...vortex-scalar-v0.17.0) -
2024-11-15

### Added

- split computation of primitive statistics ([#1306](https://github.com/spiraldb/vortex/pull/1306))

##

`vortex-flatbuffers` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.16.0...vortex-flatbuffers-v0.17.0) -
2024-11-15

### Other

- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-dtype` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.16.0...vortex-dtype-v0.17.0) -
2024-11-15

### Added

- split computation of primitive statistics ([#1306](https://github.com/spiraldb/vortex/pull/1306))

### Other

- Extract RowMask creation and filtering to separate struct ([#1272](https://github.com/spiraldb/vortex/pull/1272))

##

`vortex-array` - [0.17.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.16.0...vortex-array-v0.17.0) -
2024-11-15

### Added

- split computation of primitive statistics ([#1306](https://github.com/spiraldb/vortex/pull/1306))
- stats implementations for more array types ([#1305](https://github.com/spiraldb/vortex/pull/1305))
- run VortexFileArrayStream on dedicated IoDispatcher ([#1232](https://github.com/spiraldb/vortex/pull/1232))

### Other

- Faster boolean stats ([#1301](https://github.com/spiraldb/vortex/pull/1301))
- Extract RowMask creation and filtering to separate struct ([#1272](https://github.com/spiraldb/vortex/pull/1272))

## `vortex` - [0.17.0](https://github.com/spiraldb/vortex/compare/0.16.0...0.17.0) - 2024-11-15

### Other

- replace vortex-serde with 3 crates ([#1296](https://github.com/spiraldb/vortex/pull/1296))

##

`vortex-serde` - [0.16.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.15.2...vortex-serde-v0.16.0) -
2024-11-13

### Added

- kleene_or, kleene_and, and restore SQL semantics ([#1284](https://github.com/spiraldb/vortex/pull/1284))

### Fixed

- clarify and vs and_kleene in stream.rs and filtering.rs ([#1289](https://github.com/spiraldb/vortex/pull/1289))

##

`vortex-expr` - [0.16.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.15.2...vortex-expr-v0.16.0) -
2024-11-13

### Added

- kleene_or, kleene_and, and restore SQL semantics ([#1284](https://github.com/spiraldb/vortex/pull/1284))

##

`vortex-array` - [0.16.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.15.2...vortex-array-v0.16.0) -
2024-11-13

### Added

- kleene_or, kleene_and, and restore SQL semantics ([#1284](https://github.com/spiraldb/vortex/pull/1284))

##

`vortex-datafusion` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.14.0...vortex-datafusion-v0.15.2) -
2024-11-13

### Added

- [**breaking**] standardize file format names & stop wrapping Footer contents in
  messages ([#1275](https://github.com/spiraldb/vortex/pull/1275))

### Other

- *(deps)* update datafusion to v43 (major) ([#1261](https://github.com/spiraldb/vortex/pull/1261))
- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- some documentation of the layout reading system ([#1225](https://github.com/spiraldb/vortex/pull/1225))
- Replace usages of lazy_static with LazyLock ([#1214](https://github.com/spiraldb/vortex/pull/1214))

##

`vortex-serde` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.14.0...vortex-serde-v0.15.2) -
2024-11-13

### Added

- [**breaking**] standardize file format names & stop wrapping Footer contents in
  messages ([#1275](https://github.com/spiraldb/vortex/pull/1275))
- do not read fully masked-out layouts ([#1251](https://github.com/spiraldb/vortex/pull/1251))
- prefer take to filter for very sparse masks ([#1249](https://github.com/spiraldb/vortex/pull/1249))
- teach LayoutBatchStream to filter by indices ([#1242](https://github.com/spiraldb/vortex/pull/1242))
- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Fixed

- fix NotEq case in anticipation of non-literals ([#1260](https://github.com/spiraldb/vortex/pull/1260))

### Other

- Trim dev arrow dependencies to individual packages ([#1259](https://github.com/spiraldb/vortex/pull/1259))
- rework LayoutBatchStream for legibility ([#1245](https://github.com/spiraldb/vortex/pull/1245))
- rename ByteBufferReader to ArrayMessageReader and move messages
  around ([#1254](https://github.com/spiraldb/vortex/pull/1254))
- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- Add Not vortex expr and use it in pruning expression
  conversion ([#1213](https://github.com/spiraldb/vortex/pull/1213))
- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))
- some documentation of the layout reading system ([#1225](https://github.com/spiraldb/vortex/pull/1225))
- move Buffer enum to private inner struct ([#1216](https://github.com/spiraldb/vortex/pull/1216))
- move Array enum out of public interface ([#1212](https://github.com/spiraldb/vortex/pull/1212))
- Replace usages of lazy_static with LazyLock ([#1214](https://github.com/spiraldb/vortex/pull/1214))

##

`vortex-zigzag` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.14.0...vortex-zigzag-v0.15.2) -
2024-11-13

### Added

- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))

##

`vortex-sampling-compressor` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.14.0...vortex-sampling-compressor-v0.15.2) -
2024-11-13

### Added

- eagerly compute pruning stats during compression ([#1252](https://github.com/spiraldb/vortex/pull/1252))
- propagate statistics through compression ([#1236](https://github.com/spiraldb/vortex/pull/1236))
- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- split out sampling compressor ([#1262](https://github.com/spiraldb/vortex/pull/1262))
- port random access benchmark to layouts ([#1246](https://github.com/spiraldb/vortex/pull/1246))
- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))
- Replace usages of lazy_static with LazyLock ([#1214](https://github.com/spiraldb/vortex/pull/1214))

##

`vortex-runend-bool` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.14.0...vortex-runend-bool-v0.15.2) -
2024-11-13

### Added

- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Add Not vortex expr and use it in pruning expression
  conversion ([#1213](https://github.com/spiraldb/vortex/pull/1213))

##

`vortex-runend` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.14.0...vortex-runend-v0.15.2) -
2024-11-13

### Added

- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

##

`vortex-roaring` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.14.0...vortex-roaring-v0.15.2) -
2024-11-13

### Added

- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- Add Not vortex expr and use it in pruning expression
  conversion ([#1213](https://github.com/spiraldb/vortex/pull/1213))

##

`vortex-expr` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.14.0...vortex-expr-v0.15.2) -
2024-11-13

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- Add Not vortex expr and use it in pruning expression
  conversion ([#1213](https://github.com/spiraldb/vortex/pull/1213))

##

`vortex-dict` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.14.0...vortex-dict-v0.15.2) -
2024-11-13

### Added

- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- cleanup dict encoding logic ([#1231](https://github.com/spiraldb/vortex/pull/1231))
- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))

##

`vortex-datetime-parts` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.14.0...vortex-datetime-parts-v0.15.2) -
2024-11-13

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))

##

`vortex-bytebool` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.14.0...vortex-bytebool-v0.15.2) -
2024-11-13

### Added

- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- Add Not vortex expr and use it in pruning expression
  conversion ([#1213](https://github.com/spiraldb/vortex/pull/1213))
- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))

##

`vortex-fastlanes` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.14.0...vortex-fastlanes-v0.15.2) -
2024-11-13

### Added

- propagate statistics through compression ([#1236](https://github.com/spiraldb/vortex/pull/1236))
- Support patching bool arrays, patch primitive array validity and use patching when canonicalizing sparse
  arrays ([#1218](https://github.com/spiraldb/vortex/pull/1218))
- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))

##

`vortex-scalar` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.14.0...vortex-scalar-v0.15.2) -
2024-11-13

### Added

- propagate statistics through compression ([#1236](https://github.com/spiraldb/vortex/pull/1236))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))

##

`vortex-flatbuffers` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.14.0...vortex-flatbuffers-v0.15.2) -
2024-11-13

### Added

- [**breaking**] standardize file format names & stop wrapping Footer contents in
  messages ([#1275](https://github.com/spiraldb/vortex/pull/1275))

### Other

- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))
- some documentation of the layout reading system ([#1225](https://github.com/spiraldb/vortex/pull/1225))

##

`vortex-error` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-error-v0.14.0...vortex-error-v0.15.2) -
2024-11-13

### Other

- Annotate wrapped error in Context error type as #[source] ([#1265](https://github.com/spiraldb/vortex/pull/1265))
- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))

##

`vortex-datetime-dtype` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.14.0...vortex-datetime-dtype-v0.15.2) -
2024-11-13

### Other

- Replace usages of lazy_static with LazyLock ([#1214](https://github.com/spiraldb/vortex/pull/1214))

##

`vortex-buffer` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.14.0...vortex-buffer-v0.15.2) -
2024-11-13

### Other

- move Buffer enum to private inner struct ([#1216](https://github.com/spiraldb/vortex/pull/1216))

## `vortex-array` - [0.15.2](https://github.com/spiraldb/vortex/compare/0.14.0...0.15.2) - 2024-11-13

### Added

- propagate statistics through compression ([#1236](https://github.com/spiraldb/vortex/pull/1236))
- Support patching bool arrays, patch primitive array validity and use patching when canonicalizing sparse
  arrays ([#1218](https://github.com/spiraldb/vortex/pull/1218))
- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Fixed

- SparseArray canonicalize needs to correctly handle validity ([#1234](https://github.com/spiraldb/vortex/pull/1234))

### Other

- Remove unused dependencies ([#1256](https://github.com/spiraldb/vortex/pull/1256))
- Add Not vortex expr and use it in pruning expression
  conversion ([#1213](https://github.com/spiraldb/vortex/pull/1213))
- cleanup dict encoding logic ([#1231](https://github.com/spiraldb/vortex/pull/1231))
- Enable more lints about unused lifetimes and unnecessary
  prefixes ([#1233](https://github.com/spiraldb/vortex/pull/1233))
- Correctly define validity of sparse arrays with non null fill ([#1217](https://github.com/spiraldb/vortex/pull/1217))
- move Array enum out of public interface ([#1212](https://github.com/spiraldb/vortex/pull/1212))
- Replace usages of lazy_static with LazyLock ([#1214](https://github.com/spiraldb/vortex/pull/1214))

##

`vortex-alp` - [0.15.2](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.14.0...vortex-alp-v0.15.2) - 2024-11-13

### Added

- Support patching bool arrays, patch primitive array validity and use patching when canonicalizing sparse
  arrays ([#1218](https://github.com/spiraldb/vortex/pull/1218))
- teach PrimitiveArrayTrait iterate_primitive_array! and RowMask
  from_index_array ([#1241](https://github.com/spiraldb/vortex/pull/1241))

### Other

- Trim dev arrow dependencies to individual packages ([#1259](https://github.com/spiraldb/vortex/pull/1259))

##

`vortex-datafusion` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.13.1...vortex-datafusion-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))
- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))

### Other

- Filter pushdown over layouts ([#1124](https://github.com/spiraldb/vortex/pull/1124))

##

`vortex-serde` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.13.1...vortex-serde-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))
- store min, max, null count, and true count in column metadata ([#1164](https://github.com/spiraldb/vortex/pull/1164))

### Other

- Filter pushdown over layouts ([#1124](https://github.com/spiraldb/vortex/pull/1124))

##

`vortex-zigzag` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.13.1...vortex-zigzag-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-sampling-compressor` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.13.1...vortex-sampling-compressor-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-runend-bool` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.13.1...vortex-runend-bool-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-runend` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.13.1...vortex-runend-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-roaring` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.13.1...vortex-roaring-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

### Other

- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))

##

`vortex-fsst` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.13.1...vortex-fsst-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

### Other

- split CI into separate moldy jobs ([#1192](https://github.com/spiraldb/vortex/pull/1192))

##

`vortex-expr` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.13.1...vortex-expr-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

### Other

- Filter pushdown over layouts ([#1124](https://github.com/spiraldb/vortex/pull/1124))

##

`vortex-dict` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.13.1...vortex-dict-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-datetime-parts` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.13.1...vortex-datetime-parts-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))
- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))

##

`vortex-bytebool` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.13.1...vortex-bytebool-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-fastlanes` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.13.1...vortex-fastlanes-v0.14.0) -
2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-scalar` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.13.1...vortex-scalar-v0.14.0) -
2024-11-04

### Added

- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))
- store min, max, null count, and true count in column metadata ([#1164](https://github.com/spiraldb/vortex/pull/1164))

##

`vortex-proto` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-proto-v0.13.1...vortex-proto-v0.14.0) -
2024-11-04

### Added

- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))

##

`vortex-flatbuffers` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.13.1...vortex-flatbuffers-v0.14.0) -
2024-11-04

### Added

- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))

### Other

- Filter pushdown over layouts ([#1124](https://github.com/spiraldb/vortex/pull/1124))

##

`vortex-error` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.13.1...vortex-error-v0.14.0) -
2024-11-04

### Added

- docs for `vortex-error` crate ([#1184](https://github.com/spiraldb/vortex/pull/1184))

##

`vortex-dtype` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.13.1...vortex-dtype-v0.14.0) -
2024-11-04

### Added

- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))

### Other

- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))

##

`vortex-datetime-dtype` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.13.1...vortex-datetime-dtype-v0.14.0) -
2024-11-04

### Added

- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))

### Other

- improve datetime-dtype unit tests ([#1183](https://github.com/spiraldb/vortex/pull/1183))

##

`vortex-buffer` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.13.1...vortex-buffer-v0.14.0) -
2024-11-04

### Added

- store min, max, null count, and true count in column metadata ([#1164](https://github.com/spiraldb/vortex/pull/1164))

## `vortex-array` - [0.14.0](https://github.com/spiraldb/vortex/compare/0.13.1...0.14.0) - 2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))
- specify the storage_dtype in ExtDType ([#1007](https://github.com/spiraldb/vortex/pull/1007))
- store min, max, null count, and true count in column metadata ([#1164](https://github.com/spiraldb/vortex/pull/1164))

### Other

- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))

##

`vortex-alp` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.13.1...vortex-alp-v0.14.0) - 2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

##

`vortex-all` - [0.14.0](https://github.com/spiraldb/vortex/compare/vortex-all-v0.13.1...vortex-all-v0.14.0) - 2024-11-04

### Added

- it's a bird, it's a plane, it's vortex-all! ([#1140](https://github.com/spiraldb/vortex/pull/1140))

### Fixed

- update setup instructions (rye -> uv) ([#1176](https://github.com/spiraldb/vortex/pull/1176))
- fix docs badge in readme ([#753](https://github.com/spiraldb/vortex/pull/753))

### Other

- deny missing_docs on vortex-dtype ([#1182](https://github.com/spiraldb/vortex/pull/1182))
- very small README.md fixes
- More README.md improvements ([#1084](https://github.com/spiraldb/vortex/pull/1084))
- Update README.md ([#1055](https://github.com/spiraldb/vortex/pull/1055))
- minor addition to README ([#1030](https://github.com/spiraldb/vortex/pull/1030))
- updated README ([#876](https://github.com/spiraldb/vortex/pull/876))
- release to Test PyPI on each push to version tags ([#760](https://github.com/spiraldb/vortex/pull/760))
- Run ETE benchmarks with MiMalloc and leave a note encouraging its
  usage ([#399](https://github.com/spiraldb/vortex/pull/399))
- README updates ([#394](https://github.com/spiraldb/vortex/pull/394))
- Download flatc instead of building it from source ([#374](https://github.com/spiraldb/vortex/pull/374))
- Update README.md ([#337](https://github.com/spiraldb/vortex/pull/337))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Add note to readme about git submodules and zig version ([#176](https://github.com/spiraldb/vortex/pull/176))
- acknowledgments ([#171](https://github.com/spiraldb/vortex/pull/171))
- Update README.md ([#168](https://github.com/spiraldb/vortex/pull/168))
- More README updates ([#140](https://github.com/spiraldb/vortex/pull/140))
- Update README.md
- readme improvements ([#137](https://github.com/spiraldb/vortex/pull/137))
- README ([#102](https://github.com/spiraldb/vortex/pull/102))
- Root project is vortex-array ([#67](https://github.com/spiraldb/vortex/pull/67))
- Add minimal description to readme and fixup cargo metadata ([#30](https://github.com/spiraldb/vortex/pull/30))
- Add Readme

##

`vortex-serde` - [0.13.1](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.13.0...vortex-serde-v0.13.1) -
2024-10-31

### Fixed

- specify features in vortex-serde & test default features ([#1168](https://github.com/spiraldb/vortex/pull/1168))

##

`vortex-zigzag` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.12.0...vortex-zigzag-v0.13.0) -
2024-10-29

### Added

- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))

##

`vortex-runend-bool` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.12.0...vortex-runend-bool-v0.13.0) -
2024-10-29

### Added

- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- Faster RunEndBool decompression, plus metadata cleanup ([#981](https://github.com/spiraldb/vortex/pull/981))

##

`vortex-runend` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.12.0...vortex-runend-v0.13.0) -
2024-10-29

### Added

- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- RunEnd ends array max is length of the array ([#1017](https://github.com/spiraldb/vortex/pull/1017))
- Clean up runend metadata & stats ([#1011](https://github.com/spiraldb/vortex/pull/1011))

##

`vortex-roaring` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.12.0...vortex-roaring-v0.13.0) -
2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Fixed

- RoaringInt `can_compress` was erroneously always `None` ([#1004](https://github.com/spiraldb/vortex/pull/1004))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- RoaringBool bitmaps needs to manually append trailing false values when
  canonicalizing ([#988](https://github.com/spiraldb/vortex/pull/988))

##

`vortex-fsst` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.12.0...vortex-fsst-v0.13.0) -
2024-10-29

### Added

- FSSTArray::into_canonical directly build VarBinView ([#1161](https://github.com/spiraldb/vortex/pull/1161))
- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))

##

`vortex-dict` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.12.0...vortex-dict-v0.13.0) -
2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))
- specialized IntoCanonical for DictArray utf8/binary ([#1146](https://github.com/spiraldb/vortex/pull/1146))
- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- increase dict decompression throughput ([#1032](https://github.com/spiraldb/vortex/pull/1032))
- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- faster Dict logical validity ([#1034](https://github.com/spiraldb/vortex/pull/1034))
- improved objective function in sampling compressor ([#1000](https://github.com/spiraldb/vortex/pull/1000))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- Implement filter for dict ([#1099](https://github.com/spiraldb/vortex/pull/1099))
- Use foldhash in dict encoding ([#980](https://github.com/spiraldb/vortex/pull/980))

##

`vortex-datetime-parts` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.12.0...vortex-datetime-parts-v0.13.0) -
2024-10-29

### Added

- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))

##

`vortex-sampling-compressor` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.12.0...vortex-sampling-compressor-v0.13.0) -
2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))
- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- vortex.dataset.Dataset: deep integration with Polars & DuckDB ([#1089](https://github.com/spiraldb/vortex/pull/1089))
- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- add ChunkedCompressor which compresses chunk n+1 like chunk n ([#996](https://github.com/spiraldb/vortex/pull/996))
- improved objective function in sampling compressor ([#1000](https://github.com/spiraldb/vortex/pull/1000))

### Fixed

- dict compressor supports varbinview ([#1118](https://github.com/spiraldb/vortex/pull/1118))
- disable roaring compressors ([#1076](https://github.com/spiraldb/vortex/pull/1076))
- Vortex (de)compress benchmarks read/write a Layout ([#1024](https://github.com/spiraldb/vortex/pull/1024))
- RoaringInt `can_compress` was erroneously always `None` ([#1004](https://github.com/spiraldb/vortex/pull/1004))

### Other

- More docs ([#1104](https://github.com/spiraldb/vortex/pull/1104))
- Trim BitPackedMetadata to only required values ([#1046](https://github.com/spiraldb/vortex/pull/1046))

##

`vortex-serde` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.12.0...vortex-serde-v0.13.0) -
2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))
- teach PyVortex to read from object storage (attempt two) ([#1151](https://github.com/spiraldb/vortex/pull/1151))
- faster `take` for `BitPackedArray` and `SparseArray` ([#1133](https://github.com/spiraldb/vortex/pull/1133))
- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- vortex.dataset.Dataset: deep integration with Polars & DuckDB ([#1089](https://github.com/spiraldb/vortex/pull/1089))
- Unify FlatLayout with other layouts and add metadata to their flatbuffer
  representation ([#1077](https://github.com/spiraldb/vortex/pull/1077))
- improved objective function in sampling compressor ([#1000](https://github.com/spiraldb/vortex/pull/1000))

### Fixed

- RelativeLayoutCache must hold disk dtype ([#1051](https://github.com/spiraldb/vortex/pull/1051))
- from_fields is fallible ([#1054](https://github.com/spiraldb/vortex/pull/1054))

### Other

- VortexRecordBatchReader is generic over runtime ([#1120](https://github.com/spiraldb/vortex/pull/1120))
- Layouts have buffers and there's self describing schema layout ([#1098](https://github.com/spiraldb/vortex/pull/1098))
- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))
- Error when projecting projected dtype ([#1063](https://github.com/spiraldb/vortex/pull/1063))
- Delegate dtype deserialization and projection to layouts ([#1060](https://github.com/spiraldb/vortex/pull/1060))
- Correctly resolve filter column references when reading column
  layouts ([#1058](https://github.com/spiraldb/vortex/pull/1058))
- Add Select expression and reorganize vortex-expr crate ([#1049](https://github.com/spiraldb/vortex/pull/1049))
- make write logic slightly more simple ([#1026](https://github.com/spiraldb/vortex/pull/1026))
- Fix writing of Chunked Struct arrays ([#1020](https://github.com/spiraldb/vortex/pull/1020))

##

`vortex-expr` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.12.0...vortex-expr-v0.13.0) -
2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))

### Other

- Add Select expression and reorganize vortex-expr crate ([#1049](https://github.com/spiraldb/vortex/pull/1049))

##

`vortex-datafusion` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.12.0...vortex-datafusion-v0.13.0) -
2024-10-29

### Added

- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Fixed

- from_fields is fallible ([#1054](https://github.com/spiraldb/vortex/pull/1054))

### Other

- move infer_schema and infer_data_type into vortex-dtype ([#1081](https://github.com/spiraldb/vortex/pull/1081))
- Add Select expression and reorganize vortex-expr crate ([#1049](https://github.com/spiraldb/vortex/pull/1049))

##

`vortex-bytebool` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.12.0...vortex-bytebool-v0.13.0) -
2024-10-29

### Added

- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))

##

`vortex-fastlanes` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.12.0...vortex-fastlanes-v0.13.0) -
2024-10-29

### Added

- faster `take` for `BitPackedArray` and `SparseArray` ([#1133](https://github.com/spiraldb/vortex/pull/1133))
- canonicalize `indices` in `take` if sufficiently large ([#1036](https://github.com/spiraldb/vortex/pull/1036))
- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- improved objective function in sampling compressor ([#1000](https://github.com/spiraldb/vortex/pull/1000))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Resolve suppressed clippy warning in BitPacked::take ([#1135](https://github.com/spiraldb/vortex/pull/1135))
- clean up stale comments ([#1134](https://github.com/spiraldb/vortex/pull/1134))
- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))
- Trim BitPackedMetadata to only required values ([#1046](https://github.com/spiraldb/vortex/pull/1046))

##

`vortex-scalar` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.12.0...vortex-scalar-v0.13.0) -
2024-10-29

### Added

- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- teach PyArray scalar_at ([#1095](https://github.com/spiraldb/vortex/pull/1095))
- trim metadatas (part 2) ([#1028](https://github.com/spiraldb/vortex/pull/1028))
- introduce ScalarType, the trait for scalar-y Rust types ([#1008](https://github.com/spiraldb/vortex/pull/1008))
- proto matches serde: f16 serializes as an unsigned integer ([#992](https://github.com/spiraldb/vortex/pull/992))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Fixed

- rely on ScalarValue Display in Scalar Display ([#978](https://github.com/spiraldb/vortex/pull/978))
- teach protobuf how to deserialize f16 ([#991](https://github.com/spiraldb/vortex/pull/991))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- StructScalar stores borrowed arcs ([#1073](https://github.com/spiraldb/vortex/pull/1073))
- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))

##

`vortex-flatbuffers` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.12.0...vortex-flatbuffers-v0.13.0) -
2024-10-29

### Added

- Unify FlatLayout with other layouts and add metadata to their flatbuffer
  representation ([#1077](https://github.com/spiraldb/vortex/pull/1077))

### Other

- Layouts have buffers and there's self describing schema layout ([#1098](https://github.com/spiraldb/vortex/pull/1098))
- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))

##

`vortex-error` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.12.0...vortex-error-v0.13.0) -
2024-10-29

### Added

- teach PyVortex to read from object storage (attempt two) ([#1151](https://github.com/spiraldb/vortex/pull/1151))

##

`vortex-dtype` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.12.0...vortex-dtype-v0.13.0) -
2024-10-29

### Fixed

- RelativeLayoutCache must hold disk dtype ([#1051](https://github.com/spiraldb/vortex/pull/1051))
- RoaringInt `can_compress` was erroneously always `None` ([#1004](https://github.com/spiraldb/vortex/pull/1004))

### Other

- More docs ([#1104](https://github.com/spiraldb/vortex/pull/1104))
- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))
- Delegate dtype deserialization and projection to layouts ([#1060](https://github.com/spiraldb/vortex/pull/1060))

##

`vortex-datetime-dtype` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.12.0...vortex-datetime-dtype-v0.13.0) -
2024-10-29

### Other

- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))

##

`vortex-buffer` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.12.0...vortex-buffer-v0.13.0) -
2024-10-29

### Added

- FSSTArray::into_canonical directly build VarBinView ([#1161](https://github.com/spiraldb/vortex/pull/1161))

### Fixed

- even empty slices must be aligned properly ([#1112](https://github.com/spiraldb/vortex/pull/1112))

### Other

- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))

## `vortex-array` - [0.13.0](https://github.com/spiraldb/vortex/compare/0.12.0...0.13.0) - 2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))
- faster `take` for `BitPackedArray` and `SparseArray` ([#1133](https://github.com/spiraldb/vortex/pull/1133))
- German strings, attempt 3 ([#1082](https://github.com/spiraldb/vortex/pull/1082))
- print diagnostic for function implementations ([#1103](https://github.com/spiraldb/vortex/pull/1103))
- BoolArray::take is faster ([#1035](https://github.com/spiraldb/vortex/pull/1035))
- better error message when an array encoding id is unknown ([#997](https://github.com/spiraldb/vortex/pull/997))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Fixed

- VarBinArray into_canonical dtype
  erasure ([#1143](https://github.com/spiraldb/vortex/pull/1143)) ([#1145](https://github.com/spiraldb/vortex/pull/1145))
- support non-Primitive encodings for views ([#1123](https://github.com/spiraldb/vortex/pull/1123))
- canonicalize null ConstantArray to VarBinViewArray ([#1122](https://github.com/spiraldb/vortex/pull/1122))
- even empty slices must be aligned properly ([#1112](https://github.com/spiraldb/vortex/pull/1112))
- from_fields is fallible ([#1054](https://github.com/spiraldb/vortex/pull/1054))
- StructArray::try_from([]) is an error ([#1053](https://github.com/spiraldb/vortex/pull/1053))
- RoaringInt `can_compress` was erroneously always `None` ([#1004](https://github.com/spiraldb/vortex/pull/1004))
- teach pyvortex all our encodings ([#998](https://github.com/spiraldb/vortex/pull/998))
- BitWidthFreq must be u64/usize ([#974](https://github.com/spiraldb/vortex/pull/974))

### Other

- VarBinViewArray take preserves nullability  ([#1157](https://github.com/spiraldb/vortex/pull/1157))
- :from_iter_bin creates non nullable array ([#1150](https://github.com/spiraldb/vortex/pull/1150))
- Register VarBinView compare fn ([#1130](https://github.com/spiraldb/vortex/pull/1130))
- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- More docs ([#1104](https://github.com/spiraldb/vortex/pull/1104))
- move infer_schema and infer_data_type into vortex-dtype ([#1081](https://github.com/spiraldb/vortex/pull/1081))
- limit number of single char variables to 2 ([#1066](https://github.com/spiraldb/vortex/pull/1066))
- Register ConstantArray filter function ([#1057](https://github.com/spiraldb/vortex/pull/1057))
- Use BooleanBuffer when canonicalizing Constant ([#1056](https://github.com/spiraldb/vortex/pull/1056))
- Fix infinite loop in constant array compare ([#1016](https://github.com/spiraldb/vortex/pull/1016))
- Slightly nicer unknown encodings error ([#1002](https://github.com/spiraldb/vortex/pull/1002))
- Don't clone dtype on PrimitiveArray::cast ([#972](https://github.com/spiraldb/vortex/pull/972))

##

`vortex-alp` - [0.13.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.12.0...vortex-alp-v0.13.0) - 2024-10-29

### Added

- use hashbrown::hashmap everywhere (modestly faster decompress) ([#1160](https://github.com/spiraldb/vortex/pull/1160))
- print diagnostic for function implementations ([#1103](https://github.com/spiraldb/vortex/pull/1103))
- improved objective function in sampling compressor ([#1000](https://github.com/spiraldb/vortex/pull/1000))
- teach *Metadata and ScalarValue to Display ([#975](https://github.com/spiraldb/vortex/pull/975))

### Other

- Even more docs ([#1105](https://github.com/spiraldb/vortex/pull/1105))
- Trim BitPackedMetadata to only required values ([#1046](https://github.com/spiraldb/vortex/pull/1046))

##

`vortex-runend-bool` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.11.0...vortex-runend-bool-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-bytebool` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.11.0...vortex-bytebool-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-zigzag` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.11.0...vortex-zigzag-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-runend` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.11.0...vortex-runend-v0.12.0) -
2024-10-03

### Added

- SparseArray uses ScalarValue instead of Scalar ([#955](https://github.com/spiraldb/vortex/pull/955))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-roaring` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.11.0...vortex-roaring-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-fsst` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.11.0...vortex-fsst-v0.12.0) -
2024-10-03

### Added

- teach FSSTArray to compress the offsets of its codes ([#952](https://github.com/spiraldb/vortex/pull/952))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-dict` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.11.0...vortex-dict-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-datetime-parts` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.11.0...vortex-datetime-parts-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- cast error in compress_noci benchmark ([#971](https://github.com/spiraldb/vortex/pull/971))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-sampling-compressor` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.11.0...vortex-sampling-compressor-v0.12.0) -
2024-10-03

### Added

- cost of for is ~0 ([#967](https://github.com/spiraldb/vortex/pull/967))
- implement ALP-RD compression ([#947](https://github.com/spiraldb/vortex/pull/947))
- teach FSSTArray to compress the offsets of its codes ([#952](https://github.com/spiraldb/vortex/pull/952))
- teach DeltaArray slice and scalar_at ([#927](https://github.com/spiraldb/vortex/pull/927))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-serde` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.11.0...vortex-serde-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- increase the measurement_time of benches of serde ([#941](https://github.com/spiraldb/vortex/pull/941))
- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-schema` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-schema-v0.11.0...vortex-schema-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-expr` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.11.0...vortex-expr-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-datafusion` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.11.0...vortex-datafusion-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-fastlanes` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.11.0...vortex-fastlanes-v0.12.0) -
2024-10-03

### Added

- implement ALP-RD compression ([#947](https://github.com/spiraldb/vortex/pull/947))
- SparseArray uses ScalarValue instead of Scalar ([#955](https://github.com/spiraldb/vortex/pull/955))
- teach DeltaArray slice and scalar_at ([#927](https://github.com/spiraldb/vortex/pull/927))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-scalar` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.11.0...vortex-scalar-v0.12.0) -
2024-10-03

### Added

- teach ScalarValue and PValue is_instance_of ([#958](https://github.com/spiraldb/vortex/pull/958))
- SparseArray uses ScalarValue instead of Scalar ([#955](https://github.com/spiraldb/vortex/pull/955))
- slim down vortex-array metadata  ([#951](https://github.com/spiraldb/vortex/pull/951))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-proto` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-proto-v0.11.0...vortex-proto-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-flatbuffers` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.11.0...vortex-flatbuffers-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-error` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.11.0...vortex-error-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-dtype` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.11.0...vortex-dtype-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-datetime-dtype` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.11.0...vortex-datetime-dtype-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-buffer` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.11.0...vortex-buffer-v0.12.0) -
2024-10-03

### Added

- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

## `vortex-array` - [0.12.0](https://github.com/spiraldb/vortex/compare/0.11.0...0.12.0) - 2024-10-03

### Added

- implement ALP-RD compression ([#947](https://github.com/spiraldb/vortex/pull/947))
- teach ScalarValue and PValue is_instance_of ([#958](https://github.com/spiraldb/vortex/pull/958))
- SparseArray uses ScalarValue instead of Scalar ([#955](https://github.com/spiraldb/vortex/pull/955))
- BoolMetadata stores bit offset in 8 bits instead of 64 ([#956](https://github.com/spiraldb/vortex/pull/956))
- slim down vortex-array metadata  ([#951](https://github.com/spiraldb/vortex/pull/951))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- cast error in compress_noci benchmark ([#971](https://github.com/spiraldb/vortex/pull/971))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))
- Update README.md in vortex-array
- Include README.md files in crates and link top level readme into vortex-array
  crate ([#934](https://github.com/spiraldb/vortex/pull/934))
- Converting Arrow to Vortex should create Array and not ArrayData ([#931](https://github.com/spiraldb/vortex/pull/931))

##

`vortex-alp` - [0.12.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.11.0...vortex-alp-v0.12.0) - 2024-10-03

### Added

- implement ALP-RD compression ([#947](https://github.com/spiraldb/vortex/pull/947))
- SparseArray uses ScalarValue instead of Scalar ([#955](https://github.com/spiraldb/vortex/pull/955))
- enable cargo lints ([#948](https://github.com/spiraldb/vortex/pull/948))

### Fixed

- trim array metadatas + fix validity handling (part 1) ([#966](https://github.com/spiraldb/vortex/pull/966))
- make PrimitiveArray cast properly handle validity/nullability ([#968](https://github.com/spiraldb/vortex/pull/968))
- edge case in filling ALP encoded child on patches ([#939](https://github.com/spiraldb/vortex/pull/939))

### Other

- Include README in Cargo.toml ([#936](https://github.com/spiraldb/vortex/pull/936))

##

`vortex-runend-bool` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.10.1...vortex-runend-bool-v0.11.0) -
2024-09-26

### Added

- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

##

`vortex-bytebool` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.10.1...vortex-bytebool-v0.11.0) -
2024-09-26

### Added

- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

##

`vortex-runend` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.10.1...vortex-runend-v0.11.0) -
2024-09-26

### Added

- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

##

`vortex-roaring` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.10.1...vortex-roaring-v0.11.0) -
2024-09-26

### Other

- Update croaring-sys to 4.1.4 and remove workarounds for
  croaring/660 ([#898](https://github.com/spiraldb/vortex/pull/898))

##

`vortex-sampling-compressor` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.10.1...vortex-sampling-compressor-v0.11.0) -
2024-09-26

### Added

- sampling compressor is now seeded ([#917](https://github.com/spiraldb/vortex/pull/917))

##

`vortex-fastlanes` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.10.1...vortex-fastlanes-v0.11.0) -
2024-09-26

### Added

- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

### Fixed

- BitPackedArray must be unsigned ([#930](https://github.com/spiraldb/vortex/pull/930))

### Other

- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))

##

`vortex-serde` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.10.1...vortex-serde-v0.11.0) -
2024-09-26

### Added

- update IPC format to hold buffer_index ([#903](https://github.com/spiraldb/vortex/pull/903))

### Other

- Naive interleaved filtering and data reading ([#918](https://github.com/spiraldb/vortex/pull/918))
- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))

##

`vortex-schema` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-schema-v0.10.1...vortex-schema-v0.11.0) -
2024-09-26

### Other

- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))

##

`vortex-expr` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.10.1...vortex-expr-v0.11.0) -
2024-09-26

### Other

- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))

##

`vortex-datafusion` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.10.1...vortex-datafusion-v0.11.0) -
2024-09-26

### Added

- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

### Other

- VortexScanExec stats are computed only once ([#914](https://github.com/spiraldb/vortex/pull/914))
- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))
- VortexScanExec reports statistics to datafusion ([#909](https://github.com/spiraldb/vortex/pull/909))

##

`vortex-scalar` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.10.1...vortex-scalar-v0.11.0) -
2024-09-26

### Other

- Teach StructTrait how to project fields ([#910](https://github.com/spiraldb/vortex/pull/910))

##

`vortex-flatbuffers` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.10.1...vortex-flatbuffers-v0.11.0) -
2024-09-26

### Added

- update IPC format to hold buffer_index ([#903](https://github.com/spiraldb/vortex/pull/903))

##

`vortex-dtype` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.10.1...vortex-dtype-v0.11.0) -
2024-09-26

### Other

- Naive interleaved filtering and data reading ([#918](https://github.com/spiraldb/vortex/pull/918))
- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))

## `vortex-array` - [0.11.0](https://github.com/spiraldb/vortex/compare/0.10.1...0.11.0) - 2024-09-26

### Added

- update IPC format to hold buffer_index ([#903](https://github.com/spiraldb/vortex/pull/903))
- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

### Other

- Refactoring some IO-related code ([#846](https://github.com/spiraldb/vortex/pull/846))
- Teach StructTrait how to project fields ([#910](https://github.com/spiraldb/vortex/pull/910))

##

`vortex-alp` - [0.11.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.10.1...vortex-alp-v0.11.0) - 2024-09-26

### Added

- improve ALP exponent selection ([#921](https://github.com/spiraldb/vortex/pull/921))
- ArrayView::child will throw if encoding not found ([#886](https://github.com/spiraldb/vortex/pull/886))

### Other

- faster ALP encode ([#924](https://github.com/spiraldb/vortex/pull/924))

##

`vortex-serde` - [0.10.1](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.10.0...vortex-serde-v0.10.1) -
2024-09-20

### Added

- track compressed size & compare to parquet(zstd)? & canonical ([#882](https://github.com/spiraldb/vortex/pull/882))

##

`vortex-runend-bool` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.9.0...vortex-runend-bool-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-bytebool` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.9.0...vortex-bytebool-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

##

`vortex-zigzag` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.9.0...vortex-zigzag-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-runend` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.9.0...vortex-runend-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))
- Fix take on sliced RunEnd array ([#859](https://github.com/spiraldb/vortex/pull/859))

##

`vortex-roaring` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.9.0...vortex-roaring-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Compute stats for RoaringBoolArray ([#874](https://github.com/spiraldb/vortex/pull/874))

##

`vortex-fsst` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.9.0...vortex-fsst-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- make miri tests fast again (take 2) ([#884](https://github.com/spiraldb/vortex/pull/884))
- also run the compress benchmarks ([#841](https://github.com/spiraldb/vortex/pull/841))
- Use sliced_bytes of VarBinArray when iterating over bytes() ([#867](https://github.com/spiraldb/vortex/pull/867))
- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-dict` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.9.0...vortex-dict-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-datetime-parts` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.9.0...vortex-datetime-parts-v0.10.0) -
2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-sampling-compressor` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.9.0...vortex-sampling-compressor-v0.10.0) -
2024-09-20

### Added

- use Buffer for BitPackedArray ([#862](https://github.com/spiraldb/vortex/pull/862))

##

`vortex-fastlanes` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.9.0...vortex-fastlanes-v0.10.0) -
2024-09-20

### Added

- add back ptype check for BitPackedArray ([#872](https://github.com/spiraldb/vortex/pull/872))
- use Buffer for BitPackedArray ([#862](https://github.com/spiraldb/vortex/pull/862))

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-datafusion` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.9.0...vortex-datafusion-v0.10.0) -
2024-09-20

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-buffer` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.9.0...vortex-buffer-v0.10.0) -
2024-09-20

### Added

- use Buffer for BitPackedArray ([#862](https://github.com/spiraldb/vortex/pull/862))

## `vortex-array` - [0.10.0](https://github.com/spiraldb/vortex/compare/0.9.0...0.10.0) - 2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))
- teach compute_as_cast and get_as_cast to handle null-only arrays ([#881](https://github.com/spiraldb/vortex/pull/881))

### Other

- Remove clone when creating ArrayData to run validation ([#888](https://github.com/spiraldb/vortex/pull/888))
- Don't validate offset buffers when converting them to arrow ([#887](https://github.com/spiraldb/vortex/pull/887))
- Add doc to bytes and sliced_bytes methods of VarBinArray ([#869](https://github.com/spiraldb/vortex/pull/869))
- Use sliced_bytes of VarBinArray when iterating over bytes() ([#867](https://github.com/spiraldb/vortex/pull/867))
- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-alp` - [0.10.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.9.0...vortex-alp-v0.10.0) - 2024-09-20

### Fixed

- ID collision between vortex.ext and fastlanes.delta ([#878](https://github.com/spiraldb/vortex/pull/878))

### Other

- Make entry point compute functions accept generic arguments ([#861](https://github.com/spiraldb/vortex/pull/861))

##

`vortex-serde` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.8.0...vortex-serde-v0.9.0) -
2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Fixed

- vortex-serde benchmarks depend on the ipc feature in arrow ([#849](https://github.com/spiraldb/vortex/pull/849))

### Other

- Simplify/idiomize the way arrays return `&Array` ([#826](https://github.com/spiraldb/vortex/pull/826))
- Reorder row filters ([#825](https://github.com/spiraldb/vortex/pull/825))
- Introduce a new `vortex-schema` crate ([#819](https://github.com/spiraldb/vortex/pull/819))
- Convert pruning filters to express whether the block should be pruned and not whether it should
  stay ([#800](https://github.com/spiraldb/vortex/pull/800))
- Fix ChunkedArray find_chunk_idx for empty chunks ([#802](https://github.com/spiraldb/vortex/pull/802))
- More explicit API for converting search sorted results into
  indices ([#777](https://github.com/spiraldb/vortex/pull/777))
- overload the name 'Footer' a bit less ([#773](https://github.com/spiraldb/vortex/pull/773))

##

`vortex-expr` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.8.0...vortex-expr-v0.9.0) - 2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Other

- Reorder row filters ([#825](https://github.com/spiraldb/vortex/pull/825))

##

`vortex-scalar` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.8.0...vortex-scalar-v0.9.0) -
2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Other

- Define consistent float ordering ([#808](https://github.com/spiraldb/vortex/pull/808))
- Actually fuzz Struct and Chunked Arrays ([#805](https://github.com/spiraldb/vortex/pull/805))
- Fuzz Chunked and Struct arrays ([#801](https://github.com/spiraldb/vortex/pull/801))
- Fuzzer performs multiple operations on the underlying array instead of just
  one ([#766](https://github.com/spiraldb/vortex/pull/766))

##

`vortex-flatbuffers` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.8.0...vortex-flatbuffers-v0.9.0) -
2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

##

`vortex-error` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.8.0...vortex-error-v0.9.0) -
2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

##

`vortex-dtype` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.8.0...vortex-dtype-v0.9.0) -
2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Other

- Add description to new `vortex-schema` crate ([#829](https://github.com/spiraldb/vortex/pull/829))
- Define consistent float ordering ([#808](https://github.com/spiraldb/vortex/pull/808))
- Actually fuzz Struct and Chunked Arrays ([#805](https://github.com/spiraldb/vortex/pull/805))

##

`vortex-datetime-dtype` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.8.0...vortex-datetime-dtype-v0.9.0) -
2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Other

- release to Test PyPI on each push to version tags ([#760](https://github.com/spiraldb/vortex/pull/760))

## `vortex-array` - [0.9.0](https://github.com/spiraldb/vortex/compare/0.8.0...0.9.0) - 2024-09-17

### Added

- implement search_sorted_many ([#840](https://github.com/spiraldb/vortex/pull/840))
- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Other

- Update to rust 1.81 binary_search algorithm ([#851](https://github.com/spiraldb/vortex/pull/851))
- Fix chunked filter handling of set slices spanning multiple
  chunks ([#842](https://github.com/spiraldb/vortex/pull/842))
- Handle empty filters when filtering empty structs ([#834](https://github.com/spiraldb/vortex/pull/834))
- Handle filtering empty struct arrays ([#827](https://github.com/spiraldb/vortex/pull/827))
- Simplify/idiomize the way arrays return `&Array` ([#826](https://github.com/spiraldb/vortex/pull/826))
- Define consistent float ordering ([#808](https://github.com/spiraldb/vortex/pull/808))
- Actually fuzz Struct and Chunked Arrays ([#805](https://github.com/spiraldb/vortex/pull/805))
- Add is_encoding to array and fix cases of redundant encoding id
  checks ([#796](https://github.com/spiraldb/vortex/pull/796))
- implement FilterFn for ChunkedArray ([#794](https://github.com/spiraldb/vortex/pull/794))
- Fix ChunkedArray find_chunk_idx for empty chunks ([#802](https://github.com/spiraldb/vortex/pull/802))
- Fuzz Chunked and Struct arrays ([#801](https://github.com/spiraldb/vortex/pull/801))
- implement FilterFn for SparseArray ([#799](https://github.com/spiraldb/vortex/pull/799))
- Better scalar compare using collect_bool ([#792](https://github.com/spiraldb/vortex/pull/792))
- greedily combine chunks before compressing ([#783](https://github.com/spiraldb/vortex/pull/783))
- Introduce MaybeCompareFn trait to allow for partial compare
  specializations ([#768](https://github.com/spiraldb/vortex/pull/768))
- More explicit API for converting search sorted results into
  indices ([#777](https://github.com/spiraldb/vortex/pull/777))
- Fix slicing already sliced SparseArray ([#780](https://github.com/spiraldb/vortex/pull/780))
- Fix SearchSorted for SparseArray when searching from Right ([#770](https://github.com/spiraldb/vortex/pull/770))
- Fix StructArray::filter length calculation ([#769](https://github.com/spiraldb/vortex/pull/769))
- Fuzzer performs multiple operations on the underlying array instead of just
  one ([#766](https://github.com/spiraldb/vortex/pull/766))
- Filter struct arrays ([#767](https://github.com/spiraldb/vortex/pull/767))
- Fix unary/binary fn on `PrimitiveArray` ([#764](https://github.com/spiraldb/vortex/pull/764))
- Fix benchmarks ([#762](https://github.com/spiraldb/vortex/pull/762))
- Better implementation for `Validity::and` ([#758](https://github.com/spiraldb/vortex/pull/758))

## `vortex-alp` - [0.9.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.8.0...vortex-alp-v0.9.0) - 2024-09-17

### Added

- more Results, fewer panics, always have backtraces ([#761](https://github.com/spiraldb/vortex/pull/761))

### Other

- Simplify/idiomize the way arrays return `&Array` ([#826](https://github.com/spiraldb/vortex/pull/826))
- Introduce MaybeCompareFn trait to allow for partial compare
  specializations ([#768](https://github.com/spiraldb/vortex/pull/768))

##

`vortex-runend-bool` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.7.0...vortex-runend-bool-v0.8.0) -
2024-09-05

### Other

- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Fix RunEnd take and scalar_at compute functions ([#588](https://github.com/spiraldb/vortex/pull/588))

##

`vortex-bytebool` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.7.0...vortex-bytebool-v0.8.0) -
2024-09-05

### Other

- Fix issues discovered by fuzzer ([#707](https://github.com/spiraldb/vortex/pull/707))
- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-zigzag` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.7.0...vortex-zigzag-v0.8.0) -
2024-09-05

### Other

- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-runend` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.7.0...vortex-runend-v0.8.0) -
2024-09-05

### Other

- Teach RunEnd take to respect its own validity ([#691](https://github.com/spiraldb/vortex/pull/691))
- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Assert expected row count in tpch_benchmark binary ([#620](https://github.com/spiraldb/vortex/pull/620))
- RunEnd array scalar_at respects validity ([#608](https://github.com/spiraldb/vortex/pull/608))
- Fix RunEnd take and scalar_at compute functions ([#588](https://github.com/spiraldb/vortex/pull/588))

##

`vortex-roaring` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.7.0...vortex-roaring-v0.8.0) -
2024-09-05

### Other

- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-fsst` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-fsst-v0.7.0...vortex-fsst-v0.8.0) - 2024-09-05

### Other

- FSSTCompressor ([#664](https://github.com/spiraldb/vortex/pull/664))
- Fix issues discovered by fuzzer ([#707](https://github.com/spiraldb/vortex/pull/707))
- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))

##

`vortex-dict` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.7.0...vortex-dict-v0.8.0) - 2024-09-05

### Other

- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-datetime-parts` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.7.0...vortex-datetime-parts-v0.8.0) -
2024-09-05

### Other

- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-sampling-compressor` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.7.0...vortex-sampling-compressor-v0.8.0) -
2024-09-05

### Other

- Add fuzzing for Take and SearchSorted functions ([#724](https://github.com/spiraldb/vortex/pull/724))
- FSSTCompressor ([#664](https://github.com/spiraldb/vortex/pull/664))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- FoR compressor handles nullable arrays ([#617](https://github.com/spiraldb/vortex/pull/617))
- Use then vs then_some for values that have to be lazy ([#599](https://github.com/spiraldb/vortex/pull/599))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-fastlanes` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.7.0...vortex-fastlanes-v0.8.0) -
2024-09-05

### Other

- Fix search_sorted for FoRArray, BitPacked and PrimitiveArray ([#732](https://github.com/spiraldb/vortex/pull/732))
- Fix issues discovered by fuzzer ([#707](https://github.com/spiraldb/vortex/pull/707))
- FoR decompression happens in place if possible ([#699](https://github.com/spiraldb/vortex/pull/699))
- Remove length of patches from ALP and BitPacked array ([#688](https://github.com/spiraldb/vortex/pull/688))
- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Bitpacking validity is checked first when getting a scalar ([#630](https://github.com/spiraldb/vortex/pull/630))
- Fix FoRArray decompression with non 0 shift ([#625](https://github.com/spiraldb/vortex/pull/625))
- FoR compressor handles nullable arrays ([#617](https://github.com/spiraldb/vortex/pull/617))
- Basic fuzzing for compression and slicing functions ([#600](https://github.com/spiraldb/vortex/pull/600))
- Use then vs then_some for values that have to be lazy ([#599](https://github.com/spiraldb/vortex/pull/599))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-serde` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.7.0...vortex-serde-v0.8.0) -
2024-09-05

### Other

- Upgrade rust nightly toolchain & MSRV ([#745](https://github.com/spiraldb/vortex/pull/745))
- directly implement VortexReadAt on File ([#738](https://github.com/spiraldb/vortex/pull/738))
- Teach schema dtype() and into_dtype() ([#714](https://github.com/spiraldb/vortex/pull/714))
- Add method for converting VortexExpr into equivalent pruning
  expression ([#701](https://github.com/spiraldb/vortex/pull/701))
- Primitive and Bool array roundtrip serialization ([#704](https://github.com/spiraldb/vortex/pull/704))
- Move flatbuffer schema project functions around ([#680](https://github.com/spiraldb/vortex/pull/680))
- Deduplicate filter projection with result projection ([#668](https://github.com/spiraldb/vortex/pull/668))
- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- Bring back ability to convert ArrayView to ArrayData ([#626](https://github.com/spiraldb/vortex/pull/626))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- Assert expected row count in tpch_benchmark binary ([#620](https://github.com/spiraldb/vortex/pull/620))
- ByteBufferReader assumes flatbuffers are validated ([#610](https://github.com/spiraldb/vortex/pull/610))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-expr` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.7.0...vortex-expr-v0.8.0) - 2024-09-05

### Other

- Add method for converting VortexExpr into equivalent pruning
  expression ([#701](https://github.com/spiraldb/vortex/pull/701))
- Fix Operator::swap ([#672](https://github.com/spiraldb/vortex/pull/672))
- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- cargo-sort related maintenance  ([#650](https://github.com/spiraldb/vortex/pull/650))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- Generate more structured inputs for fuzzing ([#635](https://github.com/spiraldb/vortex/pull/635))
- Fix bug where operations were negated instead of swapped when lhs/rhs were
  flipped ([#619](https://github.com/spiraldb/vortex/pull/619))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-datafusion` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.7.0...vortex-datafusion-v0.8.0) -
2024-09-05

### Other

- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- Remove dead code after disk and in memory table provider
  unification ([#633](https://github.com/spiraldb/vortex/pull/633))
- Unify expression evaluation for both Table Providers ([#632](https://github.com/spiraldb/vortex/pull/632))
- `Exact` support for more expressions  ([#628](https://github.com/spiraldb/vortex/pull/628))
- Assert expected row count in tpch_benchmark binary ([#620](https://github.com/spiraldb/vortex/pull/620))
- Fix a bug in vortex in-memory predicate pushdown ([#618](https://github.com/spiraldb/vortex/pull/618))
- Nulls as false respects original array nullability ([#606](https://github.com/spiraldb/vortex/pull/606))
- Fix a bug in the handling the conversion physical expression ([#601](https://github.com/spiraldb/vortex/pull/601))
- Vortex physical expressions support for on-disk data ([#581](https://github.com/spiraldb/vortex/pull/581))
- *(deps)* update datafusion to v41 (major) ([#595](https://github.com/spiraldb/vortex/pull/595))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-scalar` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.7.0...vortex-scalar-v0.8.0) -
2024-09-05

### Other

- Fix search_sorted for FoRArray, BitPacked and PrimitiveArray ([#732](https://github.com/spiraldb/vortex/pull/732))
- Add fuzzing for Take and SearchSorted functions ([#724](https://github.com/spiraldb/vortex/pull/724))
- impl Display for Time, Date, and Timestamp ([#683](https://github.com/spiraldb/vortex/pull/683))
- impl Display for StructValue ([#682](https://github.com/spiraldb/vortex/pull/682))
- impl Display for Utf8Scalar and BinaryScalar ([#678](https://github.com/spiraldb/vortex/pull/678))
- Add doc to pvalue typed accessor methods ([#658](https://github.com/spiraldb/vortex/pull/658))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-proto` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-proto-v0.7.0...vortex-proto-v0.8.0) -
2024-09-05

### Other

- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- cargo-sort related maintenance  ([#650](https://github.com/spiraldb/vortex/pull/650))

##

`vortex-flatbuffers` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.7.0...vortex-flatbuffers-v0.8.0) -
2024-09-05

### Other

- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-error` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.7.0...vortex-error-v0.8.0) -
2024-09-05

### Other

- impl Display for Time, Date, and Timestamp ([#683](https://github.com/spiraldb/vortex/pull/683))

##

`vortex-dtype` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.7.0...vortex-dtype-v0.8.0) -
2024-09-05

### Other

- Move flatbuffer schema project functions around ([#680](https://github.com/spiraldb/vortex/pull/680))
- DType serde project requires flatbuffers feature ([#679](https://github.com/spiraldb/vortex/pull/679))
- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- Get beyond the immediate fuzzing failures ([#611](https://github.com/spiraldb/vortex/pull/611))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-datetime-dtype` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-dtype-v0.7.0...vortex-datetime-dtype-v0.8.0) -
2024-09-05

### Other

- impl Display for Time, Date, and Timestamp ([#683](https://github.com/spiraldb/vortex/pull/683))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- cargo-sort related maintenance  ([#650](https://github.com/spiraldb/vortex/pull/650))

##

`vortex-buffer` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.7.0...vortex-buffer-v0.8.0) -
2024-09-05

### Other

- Primitive Iterator API ([#689](https://github.com/spiraldb/vortex/pull/689))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-array` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.7.0...vortex-array-v0.8.0) -
2024-09-05

### Other

- PyVortex ([#729](https://github.com/spiraldb/vortex/pull/729))
- Upgrade rust nightly toolchain & MSRV ([#745](https://github.com/spiraldb/vortex/pull/745))
- Fix search_sorted for FoRArray, BitPacked and PrimitiveArray ([#732](https://github.com/spiraldb/vortex/pull/732))
- Unary and Binary functions trait ([#726](https://github.com/spiraldb/vortex/pull/726))
- Add fuzzing for Take and SearchSorted functions ([#724](https://github.com/spiraldb/vortex/pull/724))
- Fix issues discovered by fuzzer ([#707](https://github.com/spiraldb/vortex/pull/707))
- Primitive Iterator API ([#689](https://github.com/spiraldb/vortex/pull/689))
- StructArray roundtrips arrow conversion ([#705](https://github.com/spiraldb/vortex/pull/705))
- Primitive and Bool array roundtrip serialization ([#704](https://github.com/spiraldb/vortex/pull/704))
- Fix pack_varbin ([#674](https://github.com/spiraldb/vortex/pull/674))
- Slightly faster iter of `LogicalValidity` to `Validity` ([#673](https://github.com/spiraldb/vortex/pull/673))
- Fix Operator::swap ([#672](https://github.com/spiraldb/vortex/pull/672))
- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Push filter schema manipulation into layout reader and reuse ipc message writer in file
  writer ([#651](https://github.com/spiraldb/vortex/pull/651))
- Faster canonicalization ([#663](https://github.com/spiraldb/vortex/pull/663))
- Fix slicing of ChunkedArray if end index == array length ([#660](https://github.com/spiraldb/vortex/pull/660))
- Implement LogicalValidity for ChunkedArray ([#661](https://github.com/spiraldb/vortex/pull/661))
- Support Temporal scalar conversion between datafusion and arrow ([#657](https://github.com/spiraldb/vortex/pull/657))
- Bring back ability to convert ArrayView to ArrayData ([#626](https://github.com/spiraldb/vortex/pull/626))
- Improve Primitive array comparison ([#644](https://github.com/spiraldb/vortex/pull/644))
- Let chunked arrays use specialized `compare` implementations ([#640](https://github.com/spiraldb/vortex/pull/640))
- Expand fuzzing space ([#639](https://github.com/spiraldb/vortex/pull/639))
- Move expression filters out of datafusion ([#638](https://github.com/spiraldb/vortex/pull/638))
- `Exact` support for more expressions  ([#628](https://github.com/spiraldb/vortex/pull/628))
- Fix bug where operations were negated instead of swapped when lhs/rhs were
  flipped ([#619](https://github.com/spiraldb/vortex/pull/619))
- Get beyond the immediate fuzzing failures ([#611](https://github.com/spiraldb/vortex/pull/611))
- Basic fuzzing for compression and slicing functions ([#600](https://github.com/spiraldb/vortex/pull/600))
- Vortex physical expressions support for on-disk data ([#581](https://github.com/spiraldb/vortex/pull/581))
- Use then vs then_some for values that have to be lazy ([#599](https://github.com/spiraldb/vortex/pull/599))
- Child assert includes index and encoding id ([#598](https://github.com/spiraldb/vortex/pull/598))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))
- Add tests to sparse array slicing + extra length validation ([#590](https://github.com/spiraldb/vortex/pull/590))

## `vortex-alp` - [0.8.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.7.0...vortex-alp-v0.8.0) - 2024-09-05

### Other

- ALP compressor is better at roundtripping values ([#736](https://github.com/spiraldb/vortex/pull/736))
- Fix issues discovered by fuzzer ([#707](https://github.com/spiraldb/vortex/pull/707))
- Primitive Iterator API ([#689](https://github.com/spiraldb/vortex/pull/689))
- ALP decompress in place ([#700](https://github.com/spiraldb/vortex/pull/700))
- Remove length of patches from ALP and BitPacked array ([#688](https://github.com/spiraldb/vortex/pull/688))
- Add `scalar_at_unchecked` ([#666](https://github.com/spiraldb/vortex/pull/666))
- Fix alp null handling ([#623](https://github.com/spiraldb/vortex/pull/623))
- Clippy deny `unwrap` & `panic` in functions that return `Result` ([#578](https://github.com/spiraldb/vortex/pull/578))

##

`vortex-array` - [0.7.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.6.0...vortex-array-v0.7.0) -
2024-08-09

### Other

- Fix REE slicing with end being equal to array len ([#586](https://github.com/spiraldb/vortex/pull/586))
- Fix vortex compressed benchmarks ([#577](https://github.com/spiraldb/vortex/pull/577))

##

`vortex-serde` - [0.6.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.5.0...vortex-serde-v0.6.0) -
2024-08-09

### Other

- Only deserialize the required dtypes by projection from the
  footer ([#569](https://github.com/spiraldb/vortex/pull/569))

##

`vortex-buffer` - [0.6.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.5.0...vortex-buffer-v0.6.0) -
2024-08-09

### Other

- enforce docstrings in vortex-buffer ([#575](https://github.com/spiraldb/vortex/pull/575))

##

`vortex-array` - [0.6.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.5.0...vortex-array-v0.6.0) -
2024-08-09

### Other

- Remove to_present_null_buffer from LogicalValidity ([#579](https://github.com/spiraldb/vortex/pull/579))

##

`vortex-runend-bool` - [0.5.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.4.12...vortex-runend-bool-v0.5.0) -
2024-08-08

### Other

- Re-import array types ([#559](https://github.com/spiraldb/vortex/pull/559))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Fix out ouf bounds when taking from run end arrays ([#501](https://github.com/spiraldb/vortex/pull/501))
- Change codes for runendbool so it doesn't conflict with
  datetimeparts ([#498](https://github.com/spiraldb/vortex/pull/498))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))

##

`vortex-bytebool` - [0.5.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.4.12...vortex-bytebool-v0.5.0) -
2024-08-08

### Other

- Refactor specialized conversion traits into `From` and `Into` ([#560](https://github.com/spiraldb/vortex/pull/560))
- Re-import array types ([#559](https://github.com/spiraldb/vortex/pull/559))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Simpler ByteBool slice ([#527](https://github.com/spiraldb/vortex/pull/527))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))

##

`vortex-serde` - [0.5.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.4.12...vortex-serde-v0.5.0) -
2024-08-08

### Other

- Push column projections down to the file IO layer ([#568](https://github.com/spiraldb/vortex/pull/568))
- Lots of things to try and get publishing working ([#557](https://github.com/spiraldb/vortex/pull/557))
- Support dynamic layouts with io batching ([#533](https://github.com/spiraldb/vortex/pull/533))
- Re-import array types ([#559](https://github.com/spiraldb/vortex/pull/559))
- File-based table provider for Datafusion ([#546](https://github.com/spiraldb/vortex/pull/546))
- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Add identity projection to the file reader ([#532](https://github.com/spiraldb/vortex/pull/532))
- Support reading unaligned chunks across columns ([#531](https://github.com/spiraldb/vortex/pull/531))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))

##

`vortex-datafusion` - [0.5.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.4.12...vortex-datafusion-v0.5.0) -
2024-08-08

### Other

- Hook on-disk vortex files into benchmarking ([#565](https://github.com/spiraldb/vortex/pull/565))

##

`vortex-error` - [0.5.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.4.12...vortex-error-v0.5.0) -
2024-08-08

### Other

- Lots of things to try and get publishing working ([#557](https://github.com/spiraldb/vortex/pull/557))

##

`vortex-array` - [0.5.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.4.12...vortex-array-v0.5.0) -
2024-08-08

### Other

- Lots of things to try and get publishing working ([#557](https://github.com/spiraldb/vortex/pull/557))
- Hook on-disk vortex files into benchmarking ([#565](https://github.com/spiraldb/vortex/pull/565))

##

`vortex-runend-bool` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-runend-bool-v0.2.0) - 2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Fix out ouf bounds when taking from run end arrays ([#501](https://github.com/spiraldb/vortex/pull/501))
- Change codes for runendbool so it doesn't conflict with
  datetimeparts ([#498](https://github.com/spiraldb/vortex/pull/498))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))

## `vortex-bytebool` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-bytebool-v0.2.0) - 2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Simpler ByteBool slice ([#527](https://github.com/spiraldb/vortex/pull/527))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))

## `vortex-serde` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-serde-v0.2.0) - 2024-08-05

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Add identity projection to the file reader ([#532](https://github.com/spiraldb/vortex/pull/532))
- Support reading unaligned chunks across columns ([#531](https://github.com/spiraldb/vortex/pull/531))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))

##

`vortex-sampling-compressor` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-sampling-compressor-v0.2.0) -
2024-08-05

### Fixed

- fix UB and run tests with miri ([#517](https://github.com/spiraldb/vortex/pull/517))

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- FoR will compress signed array when min == 0 now ([#511](https://github.com/spiraldb/vortex/pull/511))
- Smoketest for SamplingCompressor, fix bug in varbin stats ([#510](https://github.com/spiraldb/vortex/pull/510))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- Remove LocalDateTimeArray, introduce TemporalArray ([#480](https://github.com/spiraldb/vortex/pull/480))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))

## `vortex-runend` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-runend-v0.2.0) - 2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Fix out ouf bounds when taking from run end arrays ([#501](https://github.com/spiraldb/vortex/pull/501))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Use shorthand canonicalize methods ([#460](https://github.com/spiraldb/vortex/pull/460))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

## `vortex-datafusion` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-datafusion-v0.2.0) - 2024-08-05

### Fixed

- fix UB and run tests with miri ([#517](https://github.com/spiraldb/vortex/pull/517))

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Follow up for 537 ([#538](https://github.com/spiraldb/vortex/pull/538))
- Rename the pushdown config into a positive boolean value ([#537](https://github.com/spiraldb/vortex/pull/537))
- Ignore tests that miri can't run ([#514](https://github.com/spiraldb/vortex/pull/514))
- Add and/or compute functions ([#481](https://github.com/spiraldb/vortex/pull/481))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Remove LocalDateTimeArray, introduce TemporalArray ([#480](https://github.com/spiraldb/vortex/pull/480))
- Expand pushdown support with more comparison and logical
  operations ([#478](https://github.com/spiraldb/vortex/pull/478))
- Debug compilation caching ([#475](https://github.com/spiraldb/vortex/pull/475))
- Basic predicate pushdown support for Datafusion ([#472](https://github.com/spiraldb/vortex/pull/472))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Run tpch_benchmark queries single-threaded in rayon pool ([#463](https://github.com/spiraldb/vortex/pull/463))
- Update datafusion to v40 (major) ([#455](https://github.com/spiraldb/vortex/pull/455))
- Make into_arrow truly zero-copy, rewrite DataFusion operators ([#451](https://github.com/spiraldb/vortex/pull/451))
- Setup TPC-H benchmark infra ([#444](https://github.com/spiraldb/vortex/pull/444))
- v0 Datafusion with late materialization ([#414](https://github.com/spiraldb/vortex/pull/414))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- DataFusion TableProvider for memory arrays ([#384](https://github.com/spiraldb/vortex/pull/384))

## `vortex-scalar` - [0.2.0](https://github.com/spiraldb/vortex/releases/tag/vortex-scalar-v0.2.0) - 2024-08-05

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))
- More specialized compare functions ([#488](https://github.com/spiraldb/vortex/pull/488))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Basic predicate pushdown support for Datafusion ([#472](https://github.com/spiraldb/vortex/pull/472))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- FoR encoding doesn't panic if array min > i64::MAX ([#406](https://github.com/spiraldb/vortex/pull/406))
- Add `ByteBoolArray` type and fixe a bug in `BoolArray` ([#383](https://github.com/spiraldb/vortex/pull/383))
- FoR array holds encoded values as unsinged ([#401](https://github.com/spiraldb/vortex/pull/401))
- DataFusion expr conversion ([#349](https://github.com/spiraldb/vortex/pull/349))
- Fix FOR bug, also fix bench to compile ([#341](https://github.com/spiraldb/vortex/pull/341))
- Implement StructValue proto serde without google.protobuf.Value ([#343](https://github.com/spiraldb/vortex/pull/343))
- Random access benchmarks are runnable again ([#330](https://github.com/spiraldb/vortex/pull/330))
- define ScalarValue in VortexScalar protobuf ([#323](https://github.com/spiraldb/vortex/pull/323))
- Proto Refactor ([#325](https://github.com/spiraldb/vortex/pull/325))
- IPC Bench ([#319](https://github.com/spiraldb/vortex/pull/319))
- Static ArrayView ([#310](https://github.com/spiraldb/vortex/pull/310))
- StatsView2 ([#305](https://github.com/spiraldb/vortex/pull/305))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- DType Serialization ([#298](https://github.com/spiraldb/vortex/pull/298))
- OwnedBuffer ([#300](https://github.com/spiraldb/vortex/pull/300))
- Add validity to Struct arrays ([#289](https://github.com/spiraldb/vortex/pull/289))
- Extension Array ([#287](https://github.com/spiraldb/vortex/pull/287))
- Remove composite and decimal ([#285](https://github.com/spiraldb/vortex/pull/285))
- Add convenience stats retrieval functions and avoid needless copy when unwrapping stat
  value ([#279](https://github.com/spiraldb/vortex/pull/279))
- Scalar subtraction ([#270](https://github.com/spiraldb/vortex/pull/270))
- Add ExtDType ([#281](https://github.com/spiraldb/vortex/pull/281))
- Refactor for DType::Primitive ([#276](https://github.com/spiraldb/vortex/pull/276))
- Extract a vortex-scalar crate ([#275](https://github.com/spiraldb/vortex/pull/275))

##

`vortex-runend-bool` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-runend-bool-v0.1.0...vortex-runend-bool-v0.2.0) -
2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Fix out ouf bounds when taking from run end arrays ([#501](https://github.com/spiraldb/vortex/pull/501))
- Change codes for runendbool so it doesn't conflict with
  datetimeparts ([#498](https://github.com/spiraldb/vortex/pull/498))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))

##

`vortex-bytebool` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-bytebool-v0.1.0...vortex-bytebool-v0.2.0) -
2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Simpler ByteBool slice ([#527](https://github.com/spiraldb/vortex/pull/527))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))

##

`vortex-serde` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-serde-v0.1.0...vortex-serde-v0.2.0) -
2024-08-05

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Add identity projection to the file reader ([#532](https://github.com/spiraldb/vortex/pull/532))
- Support reading unaligned chunks across columns ([#531](https://github.com/spiraldb/vortex/pull/531))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))

##

`vortex-zigzag` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-zigzag-v0.1.0...vortex-zigzag-v0.2.0) -
2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Fastlanez -> Fastlanes ([#381](https://github.com/spiraldb/vortex/pull/381))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

##

`vortex-sampling-compressor` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-sampling-compressor-v0.1.0...vortex-sampling-compressor-v0.2.0) -
2024-08-05

### Fixed

- fix UB and run tests with miri ([#517](https://github.com/spiraldb/vortex/pull/517))

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- FoR will compress signed array when min == 0 now ([#511](https://github.com/spiraldb/vortex/pull/511))
- Smoketest for SamplingCompressor, fix bug in varbin stats ([#510](https://github.com/spiraldb/vortex/pull/510))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- Remove LocalDateTimeArray, introduce TemporalArray ([#480](https://github.com/spiraldb/vortex/pull/480))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))

##

`vortex-runend` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-runend-v0.1.0...vortex-runend-v0.2.0) -
2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Fix out ouf bounds when taking from run end arrays ([#501](https://github.com/spiraldb/vortex/pull/501))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Use shorthand canonicalize methods ([#460](https://github.com/spiraldb/vortex/pull/460))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

##

`vortex-roaring` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-roaring-v0.1.0...vortex-roaring-v0.2.0) -
2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Ignore tests that miri can't run ([#514](https://github.com/spiraldb/vortex/pull/514))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

##

`vortex-fastlanes` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-fastlanes-v0.1.0...vortex-fastlanes-v0.2.0) -
2024-08-05

### Fixed

- fix UB and run tests with miri ([#517](https://github.com/spiraldb/vortex/pull/517))

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Use shorthand canonicalize methods ([#460](https://github.com/spiraldb/vortex/pull/460))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Fix semantic conflict between searching and slicing sparse and bitpacked
  arrays ([#412](https://github.com/spiraldb/vortex/pull/412))
- Fix Slice and SearchSorted for BitPackedArray ([#410](https://github.com/spiraldb/vortex/pull/410))
- FoR encoding doesn't panic if array min > i64::MAX ([#406](https://github.com/spiraldb/vortex/pull/406))
- Add search_sorted for FOR, Bitpacked and Sparse arrays ([#398](https://github.com/spiraldb/vortex/pull/398))
- FoR array holds encoded values as unsinged ([#401](https://github.com/spiraldb/vortex/pull/401))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Fastlanez -> Fastlanes ([#381](https://github.com/spiraldb/vortex/pull/381))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

##

`vortex-dict` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-dict-v0.1.0...vortex-dict-v0.2.0) - 2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Basic predicate pushdown support for Datafusion ([#472](https://github.com/spiraldb/vortex/pull/472))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Run `cargo doc` at CI time ([#469](https://github.com/spiraldb/vortex/pull/469))
- Use shorthand canonicalize methods ([#460](https://github.com/spiraldb/vortex/pull/460))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

##

`vortex-datetime-parts` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-datetime-parts-v0.1.0...vortex-datetime-parts-v0.2.0) -
2024-08-05

### Fixed

- canonicalization of chunked ExtensionArray ([#499](https://github.com/spiraldb/vortex/pull/499))

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Add license check to CI ([#518](https://github.com/spiraldb/vortex/pull/518))
- Fix SparseArray validity logic and give DateTimeParts unique
  code ([#495](https://github.com/spiraldb/vortex/pull/495))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Remove LocalDateTimeArray, introduce TemporalArray ([#480](https://github.com/spiraldb/vortex/pull/480))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Use shorthand canonicalize methods ([#460](https://github.com/spiraldb/vortex/pull/460))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))

##

`vortex-datafusion` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-datafusion-v0.1.0...vortex-datafusion-v0.2.0) -
2024-08-05

### Fixed

- fix UB and run tests with miri ([#517](https://github.com/spiraldb/vortex/pull/517))

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Follow up for 537 ([#538](https://github.com/spiraldb/vortex/pull/538))
- Rename the pushdown config into a positive boolean value ([#537](https://github.com/spiraldb/vortex/pull/537))
- Ignore tests that miri can't run ([#514](https://github.com/spiraldb/vortex/pull/514))
- Add and/or compute functions ([#481](https://github.com/spiraldb/vortex/pull/481))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Remove LocalDateTimeArray, introduce TemporalArray ([#480](https://github.com/spiraldb/vortex/pull/480))
- Expand pushdown support with more comparison and logical
  operations ([#478](https://github.com/spiraldb/vortex/pull/478))
- Debug compilation caching ([#475](https://github.com/spiraldb/vortex/pull/475))
- Basic predicate pushdown support for Datafusion ([#472](https://github.com/spiraldb/vortex/pull/472))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Run tpch_benchmark queries single-threaded in rayon pool ([#463](https://github.com/spiraldb/vortex/pull/463))
- Update datafusion to v40 (major) ([#455](https://github.com/spiraldb/vortex/pull/455))
- Make into_arrow truly zero-copy, rewrite DataFusion operators ([#451](https://github.com/spiraldb/vortex/pull/451))
- Setup TPC-H benchmark infra ([#444](https://github.com/spiraldb/vortex/pull/444))
- v0 Datafusion with late materialization ([#414](https://github.com/spiraldb/vortex/pull/414))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- DataFusion TableProvider for memory arrays ([#384](https://github.com/spiraldb/vortex/pull/384))

##

`vortex-scalar` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-scalar-v0.1.0...vortex-scalar-v0.2.0) -
2024-08-05

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))
- More specialized compare functions ([#488](https://github.com/spiraldb/vortex/pull/488))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Basic predicate pushdown support for Datafusion ([#472](https://github.com/spiraldb/vortex/pull/472))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- FoR encoding doesn't panic if array min > i64::MAX ([#406](https://github.com/spiraldb/vortex/pull/406))
- Add `ByteBoolArray` type and fixe a bug in `BoolArray` ([#383](https://github.com/spiraldb/vortex/pull/383))
- FoR array holds encoded values as unsinged ([#401](https://github.com/spiraldb/vortex/pull/401))
- DataFusion expr conversion ([#349](https://github.com/spiraldb/vortex/pull/349))
- Fix FOR bug, also fix bench to compile ([#341](https://github.com/spiraldb/vortex/pull/341))
- Implement StructValue proto serde without google.protobuf.Value ([#343](https://github.com/spiraldb/vortex/pull/343))
- Random access benchmarks are runnable again ([#330](https://github.com/spiraldb/vortex/pull/330))
- define ScalarValue in VortexScalar protobuf ([#323](https://github.com/spiraldb/vortex/pull/323))
- Proto Refactor ([#325](https://github.com/spiraldb/vortex/pull/325))
- IPC Bench ([#319](https://github.com/spiraldb/vortex/pull/319))
- Static ArrayView ([#310](https://github.com/spiraldb/vortex/pull/310))
- StatsView2 ([#305](https://github.com/spiraldb/vortex/pull/305))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- DType Serialization ([#298](https://github.com/spiraldb/vortex/pull/298))
- OwnedBuffer ([#300](https://github.com/spiraldb/vortex/pull/300))
- Add validity to Struct arrays ([#289](https://github.com/spiraldb/vortex/pull/289))
- Extension Array ([#287](https://github.com/spiraldb/vortex/pull/287))
- Remove composite and decimal ([#285](https://github.com/spiraldb/vortex/pull/285))
- Add convenience stats retrieval functions and avoid needless copy when unwrapping stat
  value ([#279](https://github.com/spiraldb/vortex/pull/279))
- Scalar subtraction ([#270](https://github.com/spiraldb/vortex/pull/270))
- Add ExtDType ([#281](https://github.com/spiraldb/vortex/pull/281))
- Refactor for DType::Primitive ([#276](https://github.com/spiraldb/vortex/pull/276))
- Extract a vortex-scalar crate ([#275](https://github.com/spiraldb/vortex/pull/275))

##

`vortex-expr` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-expr-v0.1.0...vortex-expr-v0.2.0) - 2024-08-05

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Clean up fields / field paths ([#353](https://github.com/spiraldb/vortex/pull/353))
- Expression proto serde ([#351](https://github.com/spiraldb/vortex/pull/351))
- DataFusion expr conversion ([#349](https://github.com/spiraldb/vortex/pull/349))
- FilterIndices compute function ([#326](https://github.com/spiraldb/vortex/pull/326))
- Introduce FieldPath abstraction, restrict predicates to Field, Op, (
  Field|Scalar) ([#324](https://github.com/spiraldb/vortex/pull/324))
- Minimal expressions API for vortex ([#318](https://github.com/spiraldb/vortex/pull/318))

##

`vortex-flatbuffers` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-flatbuffers-v0.1.0...vortex-flatbuffers-v0.2.0) -
2024-08-05

### Other

- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- Remove unused dependencies (and bump lance) ([#286](https://github.com/spiraldb/vortex/pull/286))
- Add ExtDType ([#281](https://github.com/spiraldb/vortex/pull/281))
- IPC Terminator ([#267](https://github.com/spiraldb/vortex/pull/267))
- Refactor ([#237](https://github.com/spiraldb/vortex/pull/237))
- Constant ([#230](https://github.com/spiraldb/vortex/pull/230))
- Format imports ([#184](https://github.com/spiraldb/vortex/pull/184))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))

##

`vortex-error` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-error-v0.1.0...vortex-error-v0.2.0) -
2024-08-05

### Other

- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Minimal support for reading vortex with object_store ([#427](https://github.com/spiraldb/vortex/pull/427))
- IPC Bench ([#319](https://github.com/spiraldb/vortex/pull/319))
- More Async IPC ([#313](https://github.com/spiraldb/vortex/pull/313))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- Extension Array ([#283](https://github.com/spiraldb/vortex/pull/283))
- Struct Array ([#217](https://github.com/spiraldb/vortex/pull/217))
- Array Data + View ([#210](https://github.com/spiraldb/vortex/pull/210))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Reduce number of distinct error types and capture tracebacks ([#175](https://github.com/spiraldb/vortex/pull/175))
- Random Access Benchmark ([#149](https://github.com/spiraldb/vortex/pull/149))
- Vortex Error ([#133](https://github.com/spiraldb/vortex/pull/133))

##

`vortex-dtype` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-dtype-v0.1.0...vortex-dtype-v0.2.0) -
2024-08-05

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))
- Add and/or compute functions ([#481](https://github.com/spiraldb/vortex/pull/481))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Avoid dtype comparison failure in `take` -- upcast indices in
  `take_strict_sorted`  ([#464](https://github.com/spiraldb/vortex/pull/464))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- DataFusion TableProvider for memory arrays ([#384](https://github.com/spiraldb/vortex/pull/384))
- Clean up fields / field paths ([#353](https://github.com/spiraldb/vortex/pull/353))
- Expression proto serde ([#351](https://github.com/spiraldb/vortex/pull/351))
- DataFusion expr conversion ([#349](https://github.com/spiraldb/vortex/pull/349))
- FilterIndices compute function ([#326](https://github.com/spiraldb/vortex/pull/326))
- Proto Refactor ([#325](https://github.com/spiraldb/vortex/pull/325))
- Introduce FieldPath abstraction, restrict predicates to Field, Op, (
  Field|Scalar) ([#324](https://github.com/spiraldb/vortex/pull/324))
- Minimal expressions API for vortex ([#318](https://github.com/spiraldb/vortex/pull/318))
- IPC Bench ([#319](https://github.com/spiraldb/vortex/pull/319))
- Remove buffer -> dtype dependency ([#309](https://github.com/spiraldb/vortex/pull/309))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- DType Serialization ([#298](https://github.com/spiraldb/vortex/pull/298))
- Add validity to Struct arrays ([#289](https://github.com/spiraldb/vortex/pull/289))
- Remove unused dependencies (and bump lance) ([#286](https://github.com/spiraldb/vortex/pull/286))
- Remove composite and decimal ([#285](https://github.com/spiraldb/vortex/pull/285))
- Extension Array ([#283](https://github.com/spiraldb/vortex/pull/283))
- Add convenience stats retrieval functions and avoid needless copy when unwrapping stat
  value ([#279](https://github.com/spiraldb/vortex/pull/279))
- Scalar subtraction ([#270](https://github.com/spiraldb/vortex/pull/270))
- Add ExtDType ([#281](https://github.com/spiraldb/vortex/pull/281))
- Refactor for DType::Primitive ([#276](https://github.com/spiraldb/vortex/pull/276))
- Move PType into vortex-dtype ([#274](https://github.com/spiraldb/vortex/pull/274))
- Vortex Schema -> Vortex DType ([#273](https://github.com/spiraldb/vortex/pull/273))

##

`vortex-buffer` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-buffer-v0.1.0...vortex-buffer-v0.2.0) -
2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- setup automated releases with release-plz ([#547](https://github.com/spiraldb/vortex/pull/547))
- Make into_arrow truly zero-copy, rewrite DataFusion operators ([#451](https://github.com/spiraldb/vortex/pull/451))
- DataFusion TableProvider for memory arrays ([#384](https://github.com/spiraldb/vortex/pull/384))
- Buffer into_vec respects value alignment ([#387](https://github.com/spiraldb/vortex/pull/387))
- More IPC Refactorings ([#329](https://github.com/spiraldb/vortex/pull/329))
- IPC Bench ([#319](https://github.com/spiraldb/vortex/pull/319))
- More Async IPC ([#313](https://github.com/spiraldb/vortex/pull/313))
- Async IPC ([#307](https://github.com/spiraldb/vortex/pull/307))
- Remove buffer -> dtype dependency ([#309](https://github.com/spiraldb/vortex/pull/309))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- OwnedBuffer ([#300](https://github.com/spiraldb/vortex/pull/300))
- Vortex Buffer Crate ([#299](https://github.com/spiraldb/vortex/pull/299))

##

`vortex-array` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-array-v0.1.0...vortex-array-v0.2.0) -
2024-08-05

### Fixed

- fix UB and run tests with miri ([#517](https://github.com/spiraldb/vortex/pull/517))
- canonicalization of chunked ExtensionArray ([#499](https://github.com/spiraldb/vortex/pull/499))
- fix comment on TemporalArray::new_time ([#482](https://github.com/spiraldb/vortex/pull/482))

### Other

- build-vortex -> vortex-build ([#552](https://github.com/spiraldb/vortex/pull/552))
- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- setup automated releases with release-plz ([#547](https://github.com/spiraldb/vortex/pull/547))
- Initial version of simple FileReader/Writer ([#516](https://github.com/spiraldb/vortex/pull/516))
- Use Arrow's varbin builder ([#519](https://github.com/spiraldb/vortex/pull/519))
- Smoketest for SamplingCompressor, fix bug in varbin stats ([#510](https://github.com/spiraldb/vortex/pull/510))
- Added bool iterators index and slice and filtering across some array
  types ([#505](https://github.com/spiraldb/vortex/pull/505))
- Fix remaining copies ([#504](https://github.com/spiraldb/vortex/pull/504))
- Remove some vortex mem allocations in "zero-copy" memory
  transformations ([#503](https://github.com/spiraldb/vortex/pull/503))
- Lazy deserialize metadata from ArrayData and ArrayView ([#502](https://github.com/spiraldb/vortex/pull/502))
- Fix out ouf bounds when taking from run end arrays ([#501](https://github.com/spiraldb/vortex/pull/501))
- More specialized compare functions ([#488](https://github.com/spiraldb/vortex/pull/488))
- Fix SparseArray validity logic and give DateTimeParts unique
  code ([#495](https://github.com/spiraldb/vortex/pull/495))
- Add and/or compute functions ([#481](https://github.com/spiraldb/vortex/pull/481))
- Implement CastFn for chunkedarray ([#497](https://github.com/spiraldb/vortex/pull/497))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- Remove LocalDateTimeArray, introduce TemporalArray ([#480](https://github.com/spiraldb/vortex/pull/480))
- Debug compilation caching ([#475](https://github.com/spiraldb/vortex/pull/475))
- Basic predicate pushdown support for Datafusion ([#472](https://github.com/spiraldb/vortex/pull/472))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Avoid dtype comparison failure in `take` -- upcast indices in
  `take_strict_sorted`  ([#464](https://github.com/spiraldb/vortex/pull/464))
- Use shorthand canonicalize methods ([#460](https://github.com/spiraldb/vortex/pull/460))
- Add FilterFn trait + default implementation ([#458](https://github.com/spiraldb/vortex/pull/458))
- Make into_arrow truly zero-copy, rewrite DataFusion operators ([#451](https://github.com/spiraldb/vortex/pull/451))
- Completely remove scalar_buffer method ([#448](https://github.com/spiraldb/vortex/pull/448))
- Chunked take ([#447](https://github.com/spiraldb/vortex/pull/447))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Buffer chunks to read when taking rows ([#419](https://github.com/spiraldb/vortex/pull/419))
- v0 Datafusion with late materialization ([#414](https://github.com/spiraldb/vortex/pull/414))
- Add SearchSortedFn for ConstantArray ([#417](https://github.com/spiraldb/vortex/pull/417))
- Add SliceFn implementation for ConstantArray ([#416](https://github.com/spiraldb/vortex/pull/416))
- Fix Slice and SearchSorted for BitPackedArray ([#410](https://github.com/spiraldb/vortex/pull/410))
- Fix SearchSorted on sliced sparse array ([#411](https://github.com/spiraldb/vortex/pull/411))
- Add `ByteBoolArray` type and fixe a bug in `BoolArray` ([#383](https://github.com/spiraldb/vortex/pull/383))
- Add search_sorted for FOR, Bitpacked and Sparse arrays ([#398](https://github.com/spiraldb/vortex/pull/398))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- DataFusion TableProvider for memory arrays ([#384](https://github.com/spiraldb/vortex/pull/384))
- Use ChunkedArrayReader in random access benchmark ([#393](https://github.com/spiraldb/vortex/pull/393))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Buffer into_vec respects value alignment ([#387](https://github.com/spiraldb/vortex/pull/387))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Fastlanez -> Fastlanes ([#381](https://github.com/spiraldb/vortex/pull/381))
- Use IntoArrayData when we have owned arrays ([#376](https://github.com/spiraldb/vortex/pull/376))
- Clean up fields / field paths ([#353](https://github.com/spiraldb/vortex/pull/353))
- Use new search-sorted for finding chunk index ([#342](https://github.com/spiraldb/vortex/pull/342))
- NullArray + statsset cleanup ([#350](https://github.com/spiraldb/vortex/pull/350))
- Expression proto serde ([#351](https://github.com/spiraldb/vortex/pull/351))
- DataFusion expr conversion ([#349](https://github.com/spiraldb/vortex/pull/349))
- Fix FOR bug, also fix bench to compile ([#341](https://github.com/spiraldb/vortex/pull/341))
- Array comparison compute function ([#336](https://github.com/spiraldb/vortex/pull/336))
- FilterIndices compute function ([#326](https://github.com/spiraldb/vortex/pull/326))
- Take Rows Chunked Array ([#331](https://github.com/spiraldb/vortex/pull/331))
- Random access benchmarks are runnable again ([#330](https://github.com/spiraldb/vortex/pull/330))
- ChunkedArray is not a flat encoding ([#332](https://github.com/spiraldb/vortex/pull/332))
- More IPC Refactorings ([#329](https://github.com/spiraldb/vortex/pull/329))
- Add ArrayIterator and ArrayStream ([#327](https://github.com/spiraldb/vortex/pull/327))
- Stats don't allocate errors on missing stats ([#320](https://github.com/spiraldb/vortex/pull/320))
- IPC Bench ([#319](https://github.com/spiraldb/vortex/pull/319))
- Remove flatbuffers build.rs ([#316](https://github.com/spiraldb/vortex/pull/316))
- BoolArray stats respect nulls ([#314](https://github.com/spiraldb/vortex/pull/314))
- Remove array lifetimes ([#312](https://github.com/spiraldb/vortex/pull/312))
- Static ArrayView ([#310](https://github.com/spiraldb/vortex/pull/310))
- Async IPC ([#307](https://github.com/spiraldb/vortex/pull/307))
- Remove buffer -> dtype dependency ([#309](https://github.com/spiraldb/vortex/pull/309))
- Fix chunked array stat merging ([#303](https://github.com/spiraldb/vortex/pull/303))
- Include stats in IPC messages ([#302](https://github.com/spiraldb/vortex/pull/302))
- StatsView2 ([#305](https://github.com/spiraldb/vortex/pull/305))
- Add ScalarView ([#301](https://github.com/spiraldb/vortex/pull/301))
- DType Serialization ([#298](https://github.com/spiraldb/vortex/pull/298))
- OwnedBuffer ([#300](https://github.com/spiraldb/vortex/pull/300))
- Vortex Buffer Crate ([#299](https://github.com/spiraldb/vortex/pull/299))
- Support WASM ([#297](https://github.com/spiraldb/vortex/pull/297))
- Add Context and remove linkme ([#295](https://github.com/spiraldb/vortex/pull/295))
- Add validity to Struct arrays ([#289](https://github.com/spiraldb/vortex/pull/289))
- IPC take returns an iterator instead of ChunkedArray ([#271](https://github.com/spiraldb/vortex/pull/271))
- Extension Array ([#287](https://github.com/spiraldb/vortex/pull/287))
- Remove unused dependencies (and bump lance) ([#286](https://github.com/spiraldb/vortex/pull/286))
- Remove composite and decimal ([#285](https://github.com/spiraldb/vortex/pull/285))
- DateTimeParts ([#284](https://github.com/spiraldb/vortex/pull/284))
- Extension Array ([#283](https://github.com/spiraldb/vortex/pull/283))
- Add convenience stats retrieval functions and avoid needless copy when unwrapping stat
  value ([#279](https://github.com/spiraldb/vortex/pull/279))
- Scalar subtraction ([#270](https://github.com/spiraldb/vortex/pull/270))
- Add ExtDType ([#281](https://github.com/spiraldb/vortex/pull/281))
- Bring back slice for ChunkedArray ([#280](https://github.com/spiraldb/vortex/pull/280))
- Refactor for DType::Primitive ([#276](https://github.com/spiraldb/vortex/pull/276))
- Extract a vortex-scalar crate ([#275](https://github.com/spiraldb/vortex/pull/275))
- Move PType into vortex-dtype ([#274](https://github.com/spiraldb/vortex/pull/274))
- Vortex Schema -> Vortex DType ([#273](https://github.com/spiraldb/vortex/pull/273))
- Implement take for StreamArrayReader ([#266](https://github.com/spiraldb/vortex/pull/266))
- Don't skip first element in stats calculation ([#268](https://github.com/spiraldb/vortex/pull/268))
- Enable sparse compression ([#262](https://github.com/spiraldb/vortex/pull/262))
- Logical validity from stats ([#264](https://github.com/spiraldb/vortex/pull/264))
- Refactor ([#237](https://github.com/spiraldb/vortex/pull/237))
- Comparison artifacts & analysis ([#247](https://github.com/spiraldb/vortex/pull/247))
- Fix binary stats for arrays containing null bytes and match stats behaviour between varbin and primitive
  arrays ([#233](https://github.com/spiraldb/vortex/pull/233))
- Address comments from varbin enhancement pr ([#231](https://github.com/spiraldb/vortex/pull/231))
- SearchSorted can return whether search resulted in exact match ([#226](https://github.com/spiraldb/vortex/pull/226))
- Convert slice to compute function ([#227](https://github.com/spiraldb/vortex/pull/227))
- Constant ([#230](https://github.com/spiraldb/vortex/pull/230))
- Array2 compute ([#224](https://github.com/spiraldb/vortex/pull/224))
- Better iterators for VarBin/VarBinView that don't always copy ([#221](https://github.com/spiraldb/vortex/pull/221))
- Try to inline WithCompute calls ([#223](https://github.com/spiraldb/vortex/pull/223))
- Struct Array ([#217](https://github.com/spiraldb/vortex/pull/217))
- Optimize bitpacked `take` ([#192](https://github.com/spiraldb/vortex/pull/192))
- SparseArray TakeFn returns results in the requested order ([#212](https://github.com/spiraldb/vortex/pull/212))
- Add TakeFn for SparseArray ([#206](https://github.com/spiraldb/vortex/pull/206))
- Slightly simplified SparseArray FlattenFn ([#205](https://github.com/spiraldb/vortex/pull/205))
- Don't zero memory when reading a buffer ([#208](https://github.com/spiraldb/vortex/pull/208))
- Move validity into a trait ([#198](https://github.com/spiraldb/vortex/pull/198))
- Patching Bitpacked and ALP arrays doesn't require multiple
  copies ([#189](https://github.com/spiraldb/vortex/pull/189))
- Implement Validity for SparseArray, make scalar_at for bitpacked array respect
  patches ([#194](https://github.com/spiraldb/vortex/pull/194))
- Simplify chunk end searching in ChunkedArray ([#199](https://github.com/spiraldb/vortex/pull/199))
- Compute with a primitive trait ([#191](https://github.com/spiraldb/vortex/pull/191))
- Skip codecs where can_compress on sample is null ([#188](https://github.com/spiraldb/vortex/pull/188))
- Accessor lifetime ([#186](https://github.com/spiraldb/vortex/pull/186))
- Validity array ([#185](https://github.com/spiraldb/vortex/pull/185))
- Format imports ([#184](https://github.com/spiraldb/vortex/pull/184))
- IPC Prototype ([#181](https://github.com/spiraldb/vortex/pull/181))
- Use wrapping arithmetic for Frame of Reference ([#178](https://github.com/spiraldb/vortex/pull/178))
- Reduce number of distinct error types and capture tracebacks ([#175](https://github.com/spiraldb/vortex/pull/175))
- Implement generic search sorted using scalar_at ([#167](https://github.com/spiraldb/vortex/pull/167))
- Add Take for Bitpacked array ([#161](https://github.com/spiraldb/vortex/pull/161))
- Scalar_at for FORArray ([#159](https://github.com/spiraldb/vortex/pull/159))
- Random Access Benchmark ([#149](https://github.com/spiraldb/vortex/pull/149))
- Remove unknown ([#156](https://github.com/spiraldb/vortex/pull/156))
- Nullable scalars ([#152](https://github.com/spiraldb/vortex/pull/152))
- Implement Flatten for DictArray ([#143](https://github.com/spiraldb/vortex/pull/143))
- Implement take for BoolArray ([#146](https://github.com/spiraldb/vortex/pull/146))
- Chunked Take ([#141](https://github.com/spiraldb/vortex/pull/141))
- Fix dict encoding validity ([#138](https://github.com/spiraldb/vortex/pull/138))
- Add Validity enum ([#136](https://github.com/spiraldb/vortex/pull/136))
- Vortex Error ([#133](https://github.com/spiraldb/vortex/pull/133))
- Fastlanes delta ([#57](https://github.com/spiraldb/vortex/pull/57))
- Fix encoding discovery ([#132](https://github.com/spiraldb/vortex/pull/132))
- Upgrade arrow-rs to 51.0.0 and extract common dependencies to top
  level ([#127](https://github.com/spiraldb/vortex/pull/127))
- Make EncodingID Copy ([#131](https://github.com/spiraldb/vortex/pull/131))
- Noah's Arc ([#130](https://github.com/spiraldb/vortex/pull/130))
- Use flatbuffers to serialize dtypes  ([#126](https://github.com/spiraldb/vortex/pull/126))
- DateTime encoding ([#90](https://github.com/spiraldb/vortex/pull/90))
- Split vortex-schema from main crate ([#124](https://github.com/spiraldb/vortex/pull/124))
- flatten ALP arrays ([#123](https://github.com/spiraldb/vortex/pull/123))
- Composite Arrays ([#122](https://github.com/spiraldb/vortex/pull/122))
- Rename Typed to Composite ([#120](https://github.com/spiraldb/vortex/pull/120))
- Replace iter arrow with flatten ([#109](https://github.com/spiraldb/vortex/pull/109))
- Decompress to Arrow ([#106](https://github.com/spiraldb/vortex/pull/106))
- Add ability to define composite dtypes, i.e. dtypes redefining
  meaning ([#103](https://github.com/spiraldb/vortex/pull/103))
- Serde errors ([#105](https://github.com/spiraldb/vortex/pull/105))
- Trim down arrow dependency ([#98](https://github.com/spiraldb/vortex/pull/98))
- Add bit shifting to FOR ([#89](https://github.com/spiraldb/vortex/pull/89))
- remove dead polars code ([#95](https://github.com/spiraldb/vortex/pull/95))
- Add sizeof tests ([#94](https://github.com/spiraldb/vortex/pull/94))
- Scalars are an enum ([#93](https://github.com/spiraldb/vortex/pull/93))
- Search sorted ([#91](https://github.com/spiraldb/vortex/pull/91))
- More Compression ([#87](https://github.com/spiraldb/vortex/pull/87))
- Cleanup Dict encoding ([#82](https://github.com/spiraldb/vortex/pull/82))
- Compression Updates ([#84](https://github.com/spiraldb/vortex/pull/84))
- Array display ([#83](https://github.com/spiraldb/vortex/pull/83))
- Compressor recursion ([#73](https://github.com/spiraldb/vortex/pull/73))
- Rust ALP ([#72](https://github.com/spiraldb/vortex/pull/72))
- Remove PrimitiveArray::from_vec in favour of PrimitiveArray::from ([#70](https://github.com/spiraldb/vortex/pull/70))
- Fill forward compute function ([#69](https://github.com/spiraldb/vortex/pull/69))
- Root project is vortex-array ([#67](https://github.com/spiraldb/vortex/pull/67))

## `vortex-alp` - [0.2.0](https://github.com/spiraldb/vortex/compare/vortex-alp-v0.1.0...vortex-alp-v0.2.0) - 2024-08-05

### Other

- Use versioned workspace deps ([#551](https://github.com/spiraldb/vortex/pull/551))
- Run cargo-sort on the whole workspace ([#550](https://github.com/spiraldb/vortex/pull/550))
- setup automated releases with release-plz ([#547](https://github.com/spiraldb/vortex/pull/547))
- Make unary functions nicer to `use` ([#493](https://github.com/spiraldb/vortex/pull/493))
- use FQDNs in impl_encoding macro ([#490](https://github.com/spiraldb/vortex/pull/490))
- demo module level imports granularity ([#485](https://github.com/spiraldb/vortex/pull/485))
- DType variant traits ([#473](https://github.com/spiraldb/vortex/pull/473))
- Slightly nicer use statements for compute functions ([#466](https://github.com/spiraldb/vortex/pull/466))
- Array Length ([#445](https://github.com/spiraldb/vortex/pull/445))
- Remove ViewContext and assign stable ids to encodings ([#433](https://github.com/spiraldb/vortex/pull/433))
- Split compression from encodings ([#422](https://github.com/spiraldb/vortex/pull/422))
- Add search_sorted for FOR, Bitpacked and Sparse arrays ([#398](https://github.com/spiraldb/vortex/pull/398))
- Rename flatten -> canonicalize + bugfix + a secret third thing ([#402](https://github.com/spiraldb/vortex/pull/402))
- ArrayData can contain child Arrays instead of just ArrayData ([#391](https://github.com/spiraldb/vortex/pull/391))
- Rename typed_data to as_slice ([#386](https://github.com/spiraldb/vortex/pull/386))
- Fastlanez -> Fastlanes ([#381](https://github.com/spiraldb/vortex/pull/381))
- Move encodings into directory ([#379](https://github.com/spiraldb/vortex/pull/379))
