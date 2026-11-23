# Spatial coordinate systems

:::{note}
This page is a design proposal. Only [Current state](#current-state) describes what
`vortex-spatial` does today. Everything else is planned and not implemented yet.
:::

## Summary

- **Problem.** Vortex stores a coordinate reference system (CRS) for every geometry column but no
  function reads it. Distances on longitude/latitude data come back in degrees, and functions over
  two columns with different CRSs return wrong results without an error.
- **Key idea.** Supporting many CRSs means answering two separate questions:
  1. *How do we compute inside one CRS?* The answer depends only on whether the space is flat or
     curved. Three kernels cover every CRS.
  2. *How do we combine two different CRSs?* The answer depends on the transformation between
     them. This is query planning.
- **Split.** Vortex answers the first question. The query engine (SpiralDB, DuckDB, DataFusion)
  answers the second. Vortex refuses to combine inputs whose CRSs differ.
- **First step.** Support two CRSs, EPSG:4326 and EPSG:32633, which is enough to run the example
  below end to end.

## Running example

The rest of this page refers back to one query.

A satellite image tile is delivered in a flat (planar) CRS. A Sentinel-2 tile, for example, is
about 110 km across and georeferenced in a WGS 84 UTM zone. We want the distance in metres, along
the Earth's surface, between two points in that tile. Answering takes three steps:

```text
 (1) pixel (row, column)
        |  affine map: the tile's geotransform
        v
 (2) UTM easting, northing (metres, flat)            EPSG:32633
        |  inverse map projection
        v
 (3) longitude, latitude (degrees, on the ellipsoid)  EPSG:4326
        |  geodesic distance
        v
     metres along the Earth's surface
```

Steps 1 and 2 change *where the coordinates live*. Step 3 *computes* in the final space. That
difference is the basis of this design.

Inside one UTM tile, flat UTM distance and geodesic distance differ by under 0.04 %, so the result
itself is not the point of the example. Its value is that it exercises every layer: an affine map,
a projection, a curved-surface kernel, and the boundary between engine and Vortex.

## Background

### What a CRS contains

A CRS bundles three things:

| Part | What it fixes | Examples |
| --- | --- | --- |
| Coordinate system | How many axes, their order, their units | Cartesian 2D (x, y), ellipsoidal 2D (lat, lon) |
| Reference frame (datum) | What the coordinates are measured from, including the ellipsoid | WGS 84, NAD83, ETRS89 |
| Map projection (projected CRSs only) | How the curved Earth is flattened into x, y | UTM zone 33N, Pseudo-Mercator |

There are thousands of CRSs but only a handful of coordinate system types.
[ISO 19111](https://docs.ogc.org/as/18-005r5/18-005r5.html) defines eleven, and common EPSG CRSs
use four of them: Cartesian 2D (projected), Cartesian 3D (Earth-centred), ellipsoidal 2D or 3D
(geographic), and vertical 1D.

Sharing a coordinate system type does not make two CRSs interchangeable. EPSG:4326 (WGS 84) and
EPSG:4269 (NAD83) are both latitude/longitude, but the same numbers denote positions up to about
1 m apart, and much more for older datums such as NAD27.

### Edges

Two lists of vertices can be identical and still describe different shapes, because the segment
between two vertices can be drawn in different ways. On longitude/latitude it can be:

- a straight line in degree space (`planar`);
- the shortest path on a sphere, a great-circle arc (`spherical`);
- the shortest path on the ellipsoid, a geodesic (`karney`, `vincenty`, `thomas`, `andoyer`, named
  after the algorithm that computes it).

[GeoArrow](https://geoarrow.org/extension-types.html) and GeoParquet store this as the `edges`
metadata key, separate from `crs`. The default is `planar`. Points do not depend on `edges`; lines
and polygons do.

## Current state

- Every geometry type stores coordinates as `Struct<x, y[, z][, m]>` of non-nullable `f64`
  (`vortex-spatial/src/extension/coordinate.rs`). `x` is longitude or easting and `y` is latitude
  or northing.
- The CRS is an optional string in the extension metadata (`SpatialMetadata` in
  `vortex-spatial/src/extension/mod.rs`) and is exported as the GeoArrow CRS. There is no `edges`
  field.
- No function reads the CRS. Every scalar function runs flat-plane math.

In the running example this causes two failures:

- Step 3 returns degrees, not metres: `vortex.st.distance` measures a straight line in degree
  space.
- If a UTM column is compared with a longitude/latitude literal, the function compares metres with
  degrees and returns a result without an error.

## Two separate questions

When two datasets differ, the kind of difference decides what the system must do:

| How two datasets differ | Example | Question | New kernel? |
| --- | --- | --- | --- |
| Unit | EPSG:2263 (US feet) and EPSG:32118 (metres), same projection | Combine | No: multiply the result by a constant |
| Origin shifted | EPSG:32633 and EPSG:32733, false northing differs by 10,000,000 m | Combine | No: most results are unchanged |
| Projection | EPSG:32633 (UTM) and EPSG:4326 (lon/lat) | Combine | No: transform one side first |
| Datum | EPSG:4326 (WGS 84) and EPSG:4269 (NAD83) | Combine | No: transform one side first |
| Flat or curved space | UTM, and lon/lat with geodesic edges | Compute | **Yes** |
| Ellipsoid shape | WGS 84 and Clarke 1866 | Compute | No: same kernel, different parameter |

Only the shape of the space needs new code. Every other difference is about bringing both inputs
into the same CRS, or proving that this is unnecessary.

The next two sections answer each question in turn.

## Question 1: computing inside one CRS

### Three kernels

A kernel is the implementation of a function (distance, area, intersects) for one kind of space.
Three are enough:

| Kernel | Space | Used for | Parameters |
| --- | --- | --- | --- |
| K1 flat | A flat plane, or flat 3D space | Projected CRSs, Earth-centred XYZ, pixel space, lon/lat with planar edges | None |
| K2 sphere | Surface of a sphere | Lon/lat with spherical edges | Radius R |
| K3 ellipsoid | Surface of an ellipsoid | Lon/lat with geodesic edges | Semi-major axis a, flattening f |

Parameters fall into two groups:

- **Scale-only parameters do not change the computation.** The unit of K1, the radius R of K2, and
  the semi-major axis a of K3 only multiply the result: compute once, then scale lengths by the
  parameter and areas by its square.
- **Flattening f is the only parameter that changes the computation.** WGS 84 and GRS80 agree on f
  to about 1e-11, so most modern data shares one value. Older ellipsoids (Clarke 1866, Bessel 1841,
  International 1924) differ by 1e-5 to 1e-4, and using the wrong f gives relative errors of that
  order.

So one flat kernel serves every projected CRS: UTM, State Plane, Web Mercator, and national grids
all reuse the existing flat code.

### Choosing a kernel

The kernel follows from two values: the kind of coordinate space, which comes from the CRS, and
`edges`.

| Coordinate space | `edges` | Kernel |
| --- | --- | --- |
| Projected (flat, x/y) | any | K1 |
| Geographic (lon/lat) | `planar` | K1 in degree space, today's behavior |
| Geographic (lon/lat) | `spherical` | K2 |
| Geographic (lon/lat) | `karney`, `vincenty`, `thomas`, `andoyer` | K3 for distance, length, area; K2 for predicates |
| Earth-centred (XYZ) | any | K1 in 3D |

A function with no implementation for a combination returns an error rather than falling back to
flat math.

### Behavior worth knowing

- **Predicates use the sphere.** Exact intersection of ellipsoidal geodesics is expensive, so
  systems such as PostGIS `geography` and BigQuery evaluate intersects and contains on the sphere,
  and use the ellipsoid for distance, length, and area.
- **Flat distance in a projected CRS is not ground distance.** Every projection stretches the map.
  UTM stretches by under 0.04 % inside a zone; Web Mercator by 1 / cos(latitude), about 2x at
  60 degrees. A function on a projected column returns flat distance, as PostGIS `geometry` does.
  Ground distance requires an explicit transform to longitude/latitude. A kernel never switches to
  a different space on its own.
- **Planar edges on longitude/latitude are valid but not a true distance.** Predicates are correct
  for that interpretation. Distance and area come back in degrees, and away from the equator a
  degree of longitude is shorter than a degree of latitude.

### What curved spaces need

K2 and K3 need math that the flat code does not provide:

- **Sphere predicates.** Convert each longitude/latitude to a 3D unit vector. The flat orientation
  test (does point C lie left of line AB?) becomes the sign of a triple product, and segment
  intersection becomes a test on great-circle normals. [S2](https://s2geometry.io/) and BigQuery
  work this way.
- **Ellipsoid distance, length, and area.** Karney's algorithms
  ([GeographicLib](https://geographiclib.sourceforge.io/); in Rust `geographiclib-rs`, which the
  `geo` crate already uses) work per segment in closed form and vectorize well.
- **New edge cases.** Geometries that cross the 180-degree meridian, geometries around a pole, and
  polygon interiors: on a sphere every ring divides the surface into two parts, so the ring's
  direction decides which part is inside.

## Question 2: combining two CRSs

### How two CRSs can relate

The transformation between two CRSs falls into one of six classes. Each class includes the ones
above it:

| Level | Class | Example CRS pair |
| --- | --- | --- |
| 0 | Axis swap | EPSG:4326 (lat, lon) and OGC:CRS84 (lon, lat) |
| 1 | Unit change (uniform scale) | EPSG:2263 (US feet) and EPSG:32118 (metres) |
| 2 | Shift (translation) | EPSG:32633 and EPSG:32733 |
| 3 | Rotation and scale (Helmert) | Datum shift between Earth-centred CRSs, such as EPSG:7789 and EPSG:4936 |
| 4 | General affine (also shear, unequal scales) | Pixel space and a projected CRS (step 1 of the example) |
| 5 | Non-linear | Map projections (step 2 of the example); NAD27 to NAD83 through a correction grid |

Levels 0 to 4 are linear algebra on the coordinates. Level 5 bends straight lines.

The class depends on which coordinates you look at, not only on the physical operation:

- A datum shift is a level-3 rotation and scale in Earth-centred XYZ, but level 5 in
  longitude/latitude or in projected coordinates.
- A prime meridian change, such as EPSG:4807 (Paris meridian, in grads), looks like a longitude
  shift. On the sphere it is a rotation about the polar axis, so K2 and K3 results are unchanged,
  but longitudes must wrap at 360 degrees.

In practice the class can be read from the steps of the transformation pipeline that
[PROJ](https://proj.org/) returns for the pair. A pipeline made only of `axisswap`, `unitconvert`,
`affine`, and `helmert` steps is level 0 to 4. Any projection or grid step makes it level 5.

### Which results survive a transformation

If a function's result does not change under a transformation, or changes by a known factor, the
column never needs to be transformed:

| Function | Axis swap | Unit ×k | Shift | Rotation | Affine A | Non-linear |
| --- | --- | --- | --- | --- | --- | --- |
| intersects, contains, equals | same | same | same | same | same | changes |
| distance, length | same | ×k | same | same | changes | changes |
| dwithin(d) | same | threshold ×k | same | same | changes | changes |
| area | same | ×k² | same | same | ×\|det A\| | changes |
| convex hull, centroid | commutes | commutes | commutes | commutes | commutes | changes |
| bounding box (AABB) statistics | swap axes | commutes | commutes | recompute | commutes for diagonal A | changes |

"Same" means the result is identical. "Commutes" means transforming then computing gives the same
answer as computing then transforming.

Two caveats:

- Predicates fail under non-linear maps because the edges themselves change: a straight UTM
  segment becomes a curve in longitude/latitude.
- An affine map with a negative determinant, such as the y-axis flip common in pixel space, reverses
  ring direction. Predicates are unaffected, but signed area and anything that infers the inside of
  a polygon from ring direction are.

### What the planner does

For a function over two inputs, `f(a, b)`:

1. **Same CRS:** pick the kernel as in [Choosing a kernel](#choosing-a-kernel). Done.
2. **Different CRS, level 0 to 4, and `f` survives that level** (see the table): transform the
   cheap side, usually a literal, into the column's CRS, and scale the result or threshold if the
   table says so. The column is not rewritten, and its stored bounding-box statistics still prune.
3. **Otherwise:** require an explicit `ST_Transform`, or insert one. Points transform one by one.
   Lines and polygons need extra vertices added along each segment before transforming
   (densification), because a straight segment becomes a curve.

Pruning follows the same logic. Transform the query window into the column's CRS. At level 0 to 4
its bounding box transforms directly. At level 5, densify the window's boundary first and take the
bounds of the result: transforming only the four corners can miss where an edge bulges.

## Who answers which question

Vortex answers question 1. The query engine answers question 2.

Reasons for putting question 2 in the engine:

- **It is query planning.** Deciding whether to reject, transform a literal, or insert a transform
  needs the whole query: which side is constant, which is a column, how joins line up. A Vortex
  scalar function or pruning rule sees a single expression.
- **It needs PROJ.** Non-linear transforms need PROJ, its EPSG database, and correction grids that
  may be downloaded at run time. These do not belong in Vortex's core dependencies.
- **Engines already have it.** DuckDB spatial implements `ST_Transform` on top of PROJ, and
  DuckDB's `GEOMETRY` type carries a CRS that `vortex-duckdb` already reads and writes
  (`duckdb_geometry_type_get_crs`, `duckdb_vx_create_geometry`). A second implementation inside
  Vortex would compete with the engine's.
- **Pruning does not need it.** Once the engine has rewritten the literal into the column's CRS,
  Vortex pruning only sees inputs with the same CRS.

| Responsibility | Owner |
| --- | --- |
| Store `crs` and `edges` in the file | Vortex |
| Pass `crs` and `edges` through to engines (GeoArrow metadata, DuckDB `GEOMETRY` CRS) | Vortex |
| Map a CRS to its coordinate space, without an EPSG database | Vortex |
| Run K1, K2, K3 | Vortex |
| Refuse functions over inputs with different CRSs | Vortex |
| Normalize CRS strings to PROJJSON when data is loaded | Engine |
| Classify CRS pairs, transform literals, insert `ST_Transform` | Engine |
| Run `ST_Transform` through PROJ | Engine |

For the running example:

| Step | Owner | How |
| --- | --- | --- |
| 1. Pixel to UTM | Engine | The tile's geotransform, folded into the same PROJ pipeline as step 2 |
| 2. UTM to EPSG:4326 | Engine | `ST_Transform` through PROJ; two points need no densification |
| 3. Geodesic distance | Vortex | `vortex.st.distance` picks K3 from the column's `crs` and `edges` |

Written for DuckDB spatial today, which has separate functions per kernel instead of reading
`edges`:

```sql
SELECT ST_Distance_Spheroid(
    ST_Transform(a, 'EPSG:32633', 'EPSG:4326'),
    ST_Transform(b, 'EPSG:32633', 'EPSG:4326')
)
FROM tiles;
```

One part of question 2 could move into Vortex later. Levels 0 to 4 are pure linear algebra with no
database, so a `vortex.st.affine` scalar function could run the affine part of a PROJ pipeline,
vectorized over the coordinate struct. The engine would still decide when to use it. This is a
speed-up, not a requirement.

## Vortex-side design

### Metadata

Add `edges` to `SpatialMetadata` next to `crs`, with GeoArrow's values, and keep it through Arrow
export and import. A missing value means `planar`, as in GeoArrow. This is the only storage change:
coordinates keep their current layout.

### Mapping a CRS to its coordinate space

```rust
pub enum CoordKind {
    /// Projected, engineering, or pixel space: flat, in the stored units.
    Planar,
    /// Longitude and latitude on an ellipsoid, always in (lon, lat) order.
    Geographic { semi_major: f64, flattening: f64 },
    /// Earth-centred Cartesian (X, Y, Z).
    Geocentric,
}

pub struct CoordSpace {
    pub kind: CoordKind,
    /// The CRS identity, compared for equality between inputs.
    pub crs: Option<Arc<str>>,
}
```

- **PROJJSON** states the coordinate system type (`coordinate_system.subtype`) and the ellipsoid,
  so it can be mapped by parsing JSON alone.
- **Bare codes** such as `EPSG:4326` are mapped from a small built-in table.
- **Unknown or missing CRS** keeps today's behavior: flat math. Changing that default would change
  results on existing files and is a separate decision.
- **Axis order** is normalized to (lon, lat) when data is loaded, as GeoArrow requires, so kernels
  never handle it.

### CRS check

Functions over two geometries (`vortex.st.distance`, `vortex.st.intersects`, `vortex.st.contains`)
check that both inputs have the same `crs` and `edges`, and return an error otherwise. This is the
contract with the engine: Vortex never guesses a transformation.

### Pruning

The bounding-box statistic (`GeometryAabb`) and the pruning rules (`SpatialDistancePrune`,
`SpatialIntersectsPrune`, `SpatialContainsPrune`) are only correct for planar edges. Curved edges
break them in three ways:

- A great-circle arc bulges toward the pole, beyond the latitudes of both endpoints, so the box
  around the vertices does not contain the whole geometry.
- A geometry that crosses the 180-degree meridian has a longitude range that wraps around, which a
  single min/max pair cannot express.
- A `dwithin` radius in metres covers more degrees of longitude the closer it is to a pole, in
  proportion to 1 / cos(latitude).

Until these are handled, the pruning rules skip columns whose edges are not planar.

## Implementation plan

### Phase 1: two CRSs

The first phase supports the two CRSs of the running example:

| CRS | Coordinate space | Kernel |
| --- | --- | --- |
| EPSG:4326 (also accepted as OGC:CRS84) | Geographic, WGS 84 ellipsoid | K3 with geodesic edges, K1 with planar edges |
| EPSG:32633 (WGS 84 / UTM zone 33N) | Planar, metres | K1 |

Two CRSs are enough to exercise every part of the design: choosing a kernel by coordinate space, a
curved-surface kernel, the CRS check, and an engine-side transform between them. Every other CRS
keeps today's flat behavior.

Vortex:

1. Add `edges` to `SpatialMetadata` with Arrow round-trip.
2. Check that `crs` and `edges` match in functions over two geometries.
3. Map CRSs to coordinate spaces with a two-entry table: EPSG:4326 and OGC:CRS84 are geographic,
   EPSG:32633 is planar, anything else is unknown and stays flat.
4. Add K3 distance and length, chosen by `vortex.st.distance` and `vortex.st.length` for
   EPSG:4326 with geodesic edges.
5. Skip pruning for columns whose edges are not planar.

Engine, starting with DataFusion:

1. Read `crs` and `edges` from field metadata during planning.
2. Implement `ST_Transform` from EPSG:32633 to EPSG:4326 through PROJ, setting the output's `crs`.
3. Refuse functions over different CRSs, with an error that points at `ST_Transform`.
4. Run the example end to end, and compare K1 in UTM with K3 in EPSG:4326.

Vortex steps 1 and 2 do not depend on the engine and can land first.

### Phase 2: generalize

1. Map any CRS given as PROJJSON, so no table entry is needed.
2. Add K2 sphere predicates, then bounding-box statistics and pruning that are safe for curved
   edges.
3. Transform literals in the engine for levels 0 to 4, instead of refusing.
4. Consider `vortex.st.affine` for the affine part of a pipeline.

## Open questions

- **Asking for geodesic edges.** After `ST_Transform` to EPSG:4326 the output has planar edges by
  default. How does a query ask for geodesic ones: a cast, as PostGIS does with `geography`, a
  function that sets `edges`, or separate functions, as DuckDB does with `ST_Distance_Spheroid`?
- **Distance with planar edges on longitude/latitude.** Return degrees, as PostGIS `geometry`
  does, or return an error?
- **Unknown CRS.** Keep flat math by default, or require a known CRS for functions that return
  distances and areas?
- **Sphere radius.** Mean Earth radius (6,371,008.8 m) or the ellipsoid's semi-major axis.
- **DataFusion hook.** Where CRS-aware planning goes, likely a `ScalarUDF` that reads field
  metadata at planning time.

## References

- [ISO 19111 / OGC Abstract Specification Topic 2](https://docs.ogc.org/as/18-005r5/18-005r5.html):
  what a CRS and a coordinate system are.
- [GeoArrow extension types](https://geoarrow.org/extension-types.html): the `crs` and `edges`
  metadata.
- [PROJ](https://proj.org/): coordinate transformations and pipelines.
- [GeographicLib](https://geographiclib.sourceforge.io/): Karney's ellipsoid algorithms.
- [S2 Geometry](https://s2geometry.io/): sphere predicates on unit vectors.
- [PostGIS geography](https://postgis.net/docs/using_postgis_dbmanagement.html#PostGIS_Geography):
  the split between flat `geometry` and curved `geography`.
