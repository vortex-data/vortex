# vortex-spatial

This crate is where Vortex's spatial support lives. It defines GeoArrow-compatible geometry types backed by regular Vortex arrays. It also contains the spatial compute functions and chunk-pruning rules used by the DuckDB integration.

The format is still experimental. Spatial fields written today use the draft
`spatial2026.08.0` edition, which may change before it is frozen. Any session that reads or writes these fields must be initialized with this crate.

## How geometry is stored

A native geometry column is an extension array. The extension dtype says what the values mean, and the storage is made from normal Vortex structs and lists:

| Vortex dtype | GeoArrow type | Storage |
| --- | --- | --- |
| `vortex.st.point` | `geoarrow.point` | `Struct<x, y[, z][, m]>` |
| `vortex.st.linestring` | `geoarrow.linestring` | `List<coordinate>` |
| `vortex.st.multipoint` | `geoarrow.multipoint` | `List<coordinate>` |
| `vortex.st.polygon` | `geoarrow.polygon` | `List<List<coordinate>>` |
| `vortex.st.multilinestring` | `geoarrow.multilinestring` | `List<List<coordinate>>` |
| `vortex.st.multipolygon` | `geoarrow.multipolygon` | `List<List<List<coordinate>>>` |
| `vortex.st.box` | `geoarrow.box` | `Struct<xmin, ymin[, ...], xmax, ymax[, ...]>` |
| `vortex.st.wkb` | `geoarrow.wkb` | `Binary` |

Coordinates are stored as separate, non-nullable `f64` fields named `x`, `y`, and optionally `z` or `m`. Nullability belongs to the geometry row. `SpatialMetadata` stores the coordinate reference system (CRS) string.

## Setting up a session

Call `initialize` before using spatial data with a session:

```rust
let session = vortex_array::array_session();
vortex_spatial::initialize(&session);
```

This registers the geometry dtypes, Arrow converters, scalar functions, AABB statistic, pruning rules, and spatial edition. It is safe to call more than once. The DuckDB, JNI, and benchmark sessions already call it during setup.

## Compute functions

Spatial functions are lazy: applying one creates a `ScalarFnArray`, and the work happens when that array is executed. Binary functions accept constants as well as columns, and the two operands do not need to have the same geometry type.

| Function | Input | Output | Notes |
| --- | --- | --- | --- |
| `vortex.st.area` | any native geometry | `f64` | Unsigned planar area |
| `vortex.st.collect` | `List<Point>`, `List<LineString>`, or `List<Polygon>` | matching multi-geometry | Collects within each row; it is not an aggregate |
| `vortex.st.contains` | two native geometries | `bool` | OGC containment; argument order matters |
| `vortex.st.convex_hull` | `MultiPoint` | `Polygon` | Two-dimensional convex hull |
| `vortex.st.distance` | two native geometries | `f64` | Euclidean distance |
| `vortex.st.envelope` | any native geometry | `Box` | Two-dimensional bounding box |
| `vortex.st.intersects` | two native geometries | `bool` | Boundary contact counts as intersection |
| `vortex.st.length` | `LineString` or `MultiLineString` | `f64` | Euclidean length |
| `vortex.st.make_line` | two `Point` values | `LineString` | Keeps the points in argument order |

Null geometry arguments produce null results. `collect` also ignores null elements inside a valid list row, matching DuckDB.

Most functions that need a general geometry algorithm use the row-function adapters in `src/scalar_fn/row.rs`. They decode valid rows to `geo_types` and call the `geo` implementation. `length`, `envelope`, `make_line`, and `collect` work directly on the nested Vortex arrays. Use the structural path when it avoids unnecessary decoding or needs to preserve extra ordinates; otherwise, the row-function path is simpler.

Avoid per-row `scalar_at` or validity lookups in either path. Materialize coordinates, offsets, and validity once for the batch.

## Chunk pruning

`GeometryAabb` computes the two-dimensional axis-aligned bounding box (AABB) of a geometry column. The writer stores it as the default zone statistic for native geometry columns.

The current rules can prune:

- positive `ST_Intersects(geometry_column, constant_geometry)` filters when the boxes are disjoint;
- `<`, `<=`, `>`, and `>=` comparisons on
  `ST_Distance(geometry_column, constant_geometry)` using box-distance bounds.

These proofs are conservative. If the expression has a different shape, the constant is null or empty, or the file has no AABB statistic, the chunk is kept. Box overlap alone never proves that two geometries intersect.

## Interoperability and limits

Arrow import and export use the standard GeoArrow extension names and preserve CRS metadata. Native import only supports separated coordinates; interleaved GeoArrow arrays are rejected.

DuckDB uses WKB for general geometry values. Its Vortex expression lowering converts supported WKB constants to native scalars and pushes `ST_Contains`, `ST_Distance`, and `ST_Intersects` into a Vortex scan.

Some important limits:

- Geometry calculations are planar. The crate does not transform coordinates or apply geodesic corrections based on the CRS.
- Geometry calculations use `x` and `y`. Structural operations can preserve `z` and `m`: `ST_Collect` reuses the input storage, and `ST_MakeLine` preserves and promotes point dimensions.
- There is no native `GeometryCollection` or mixed-geometry union type so far.
- WKB columns cannot be passed to the native compute functions or AABB pruning.
- AABB statistics prune chunks; they are not a spatial index or spatial-join implementation.

When adding a serialized dtype or statistic, register it in `initialize` and add it to `src/editions.rs`.

## Tests and benchmarks

```bash
cargo nextest run -p vortex-spatial
cargo test --doc -p vortex-spatial
```

Each compute-function family has a focused benchmark under `benches/`. For example:

```bash
cargo bench -p vortex-spatial --bench distance
```

The extension types follow the type and metadata model from the
[GeoArrow project](https://github.com/geoarrow/geoarrow-rs).
