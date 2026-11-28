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
- **Storage.** The coordinate layout does not change. A CRS is already stored per column; phase 2
  adds one optional metadata field, `edges`.
- **First steps.** Phase 1a supports flat (projected) CRSs: Vortex needs no CRS list, only the
  check that two inputs share a CRS, and the engine decides which CRSs it can transform between.
  Phase 1b adds curved-surface distance for the WGS 84 family, choosing the kernel by function
  name. Together they run the example below end to end without a datum transform.

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

Inside one UTM tile, flat UTM distance and geodesic distance differ by at most about 0.1 %, so the result
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
metadata key, separate from `crs`. The default is `planar`.

`edges` also fixes how distance is measured: the distance between two points is the length of the
edge that would connect them. Planar edges give a straight-line distance in the coordinates,
spherical edges a great-circle distance, geodesic edges an ellipsoidal one. So `edges` matters for
points too, even though points have no edges. GeoArrow defines `edges` only as the interpolation
rule; using it to choose the distance metric follows from that definition.

### CRS and edges answer different questions

The CRS says where the vertices are: which surface (a flat projected plane or the ellipsoid), its
shape, the units, the origin, the axis order. `edges` says what connects the vertices. A
geographic CRS provides a curved surface; `edges` decides whether computation follows it.

| How two datasets differ | Differs in `crs` | Differs in `edges` |
| --- | --- | --- |
| US feet and metres, same projection (EPSG:2263, EPSG:32118) | yes: unit | no |
| UTM north and south (EPSG:32633, EPSG:32733) | yes: origin | no |
| EPSG:4326 with planar edges and EPSG:4326 with geodesic edges | no | yes |
| WGS 84 and NAD27, both with geodesic edges (EPSG:4326, EPSG:4267) | yes: datum and ellipsoid | no |
| UTM and EPSG:4326 with geodesic edges | yes | yes |

Only a geographic CRS gives `edges` a real choice. In a projected CRS the coordinates already lie
on a plane, where the shortest path is a straight line, so `edges` is always `planar`. Folding
`edges` into the CRS string would mix positions with shapes and break interoperability with
GeoArrow, GeoParquet, and DuckDB.

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

The target design chooses the kernel from two values: the kind of coordinate space, which comes
from the CRS, and `edges`. Phase 1 has no `edges` yet and chooses by function name instead; see
[Implementation plan](#implementation-plan).

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
- **Flat results in a projected CRS are not ground results.** A function on a projected column
  returns the distance, length, or area in that CRS's plane and units, as PostGIS `geometry` does.
  Ground values require an explicit transform to longitude/latitude; a kernel never switches to a
  different space on its own. How far the plane is from the ground depends on the projection:

  | Projection | Examples | Distance | Area |
  | --- | --- | --- | --- |
  | Conformal, small extent | UTM, British National Grid (EPSG:27700) | Scale 0.9996 on a zone's central meridian to about 1.001 at its edge: -0.04 % to +0.1 % | About twice the distance error |
  | Conformal, for surveying | State Plane (EPSG:2263, EPSG:32118) | Designed to stay within 1 part in 10,000 | Within about 2 parts in 10,000 |
  | Conformal, large extent | Web Mercator (EPSG:3857), World Mercator (EPSG:3395) | Scale 1 / cos(latitude): 2x at 60 degrees | Scale 1 / cos²(latitude): 4x at 60 degrees |
  | Equal-area | EASE-Grid 2.0 (EPSG:6933), US Albers (EPSG:5070) | Distorted | Equal to ground area |
  | Polar stereographic | UPS, EPSG:3413, EPSG:3031 | Good near the pole, worse away from it | Same |
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

Rule 2 is an implicit transform, so it applies only when the transform is exact and the result
does not depend on which side is transformed. A literal point under a predicate qualifies. Distance
and area over two CRSs do not: computed in UTM they come back in metres, computed in EPSG:4326 in
degrees, so the query must say which space it means. Two columns in different CRSs never qualify,
because there is no single direction to choose. PostGIS rejects mixed SRIDs for the same reasons.

The first version applies only rule 1 and otherwise requires an explicit `ST_Transform`.
Relaxing later, starting with literal points under predicates, breaks no existing query;
tightening would.

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
| 3. Geodesic distance | Vortex | `vortex.st.distance_spheroid` runs K3 for the column's geographic `crs`; from phase 2, `vortex.st.distance` can also pick K3 from `crs` and `edges` |

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

### Storage

The coordinate layout does not change. Every CRS uses the same `Struct<x, y[, z][, m]>`; only the
meaning of the fields depends on the CRS (easting and northing, longitude and latitude, or
Earth-centred X, Y, Z). The geometry types and the encodings of their `f64` columns are unaffected.

Phase 1 changes no metadata either. Phase 2 adds `edges` to `SpatialMetadata` next to `crs`, with
GeoArrow's values, as optional field 2 of the protobuf that holds the CRS today:

- **New readers, old files:** the field is missing and reads as `planar`, which is today's
  behavior.
- **Old readers, new files:** protobuf skips the unknown field, so an old reader computes on
  geodesic data as planar without an error. Tying the field to an edition would turn that into an
  error.
- **Dtype equality:** two columns that differ only in `edges` have different dtypes, which is the
  intended effect.

Three conventions need no storage change but must be stated. Longitude is always `x`, even for
EPSG:4326, whose official axis order is latitude first. A bounding box (`Rect` and the AABB
statistics) that crosses the 180-degree meridian can follow GeoParquet's convention of
`xmin > xmax`. And for an Earth-centred CRS `z` is a coordinate, not the elevation that
`coordinate.rs` documents.

### What Vortex reads from a CRS

Vortex stores the CRS as given and does not define its own CRS fields. What a CRS contains is
already standardized: [PROJJSON](https://proj.org/specifications/projjson.html) is ISO 19111 in
JSON, with a `type` (`GeographicCRS`, `ProjectedCRS`, `GeodeticCRS`, `VerticalCRS`,
`CompoundCRS`, `EngineeringCRS`), a `datum` with its ellipsoid, a `coordinate_system` with axes and
units, a `base_crs` and `conversion` for projected CRSs, and an `id`. Defining a Vortex-specific
version would reimplement that model and break interoperability with GeoArrow, GeoParquet, and
DuckDB.

What Vortex does define is the small view it derives from a CRS to compute with. The view is
computed when needed and never written to the file:

```rust
pub struct CoordSpace {
    /// Selects the kernel family.
    pub kind: CoordKind, // Planar | Geographic | Geocentric
    /// K3's parameters; present only for a geographic CRS.
    pub ellipsoid: Option<Ellipsoid>, // semi-major axis a, flattening f
    /// Labels results and gives unit-change factors: metre, US survey foot, degree.
    pub linear_unit: Option<Unit>,
    /// Canonical identity for the CRS check, so EPSG:4326 and OGC:CRS84 compare equal.
    pub id: CrsId,
}
```

| Field | Source in PROJJSON | Phase 1a | Phase 1b | Phase 2 |
| --- | --- | --- | --- | --- |
| `id` | `id`, or the canonicalized definition | Raw CRS string instead | WGS 84 family codes and aliases | Any CRS |
| `kind` | `type` | Not needed | Needed: only geographic CRSs run the spheroid functions | Needed |
| `ellipsoid` | `datum.ellipsoid`, or `base_crs.datum.ellipsoid` for a projected CRS | Not needed | Fixed to WGS 84 | Any ellipsoid |
| `linear_unit` | `coordinate_system.axis[*].unit` | Not needed | Optional | Optional |

Two things are deliberately left out. Axis order is normalized when data is loaded, so kernels
never handle it. Projection parameters belong to `ST_Transform`, which the engine owns. The cost
of supporting another CRS in Vortex is therefore only whether these four fields can be read from
it.

GeoArrow metadata can also carry an optional `crs_type` (`projjson`, `wkt2:2019`,
`authority_code`, `srid`) naming how `crs` is written. Vortex does not store it and exports every
CRS as an unknown type. The form can be read from the value itself (`{` starts PROJJSON, `EPSG:`
starts an authority code), so storing `crs_type` is optional and only improves round-trip
fidelity.

How the view is derived:

- **PROJJSON** states the type and the ellipsoid, so it can be mapped by parsing JSON alone. Phase
  1b reads only its `id` member (`{"authority": "EPSG", "code": 32633}`).
- **Bare codes** such as `EPSG:4326` are mapped by a few rules. The UTM zones are one rule with two
  parameters, the zone and the hemisphere, not 120 table entries.
- **Unknown or missing CRS** keeps today's behavior: flat math. Changing that default would change
  results on existing files and is a separate decision.
- **Axis order** is normalized to (lon, lat) when data is loaded, as GeoArrow requires, so kernels
  never handle it.

### CRS check

Most Vortex scalar functions require their operands to have equal dtypes, and an extension
dtype's metadata, here the CRS, is part of the dtype. Geometry functions cannot require that:
each geometry kind is its own extension type, and `vortex.st.distance(Point, Polygon)` is valid.
So functions over two geometries (`vortex.st.distance`, `vortex.st.intersects`,
`vortex.st.contains`, `vortex.st.make_line`) require equal CRSs instead, checked when the
expression is typed. Vortex never guesses a transformation; the engine inserts `ST_Transform`.

- **Same CRS:** accepted.
- **Both without a CRS:** accepted. Unreferenced data, such as SpatialBench's, keeps working.
- **One side with a CRS and one without:** rejected. A missing CRS is a CRS of its own,
  unreferenced coordinates, not a wildcard.
- **Different CRSs:** rejected.

The check is one function, `ensure_eq_crs`, which reads each operand's metadata through an
`AnyNativeGeometry` matcher over the native geometry types. Phase 1 compares CRS strings exactly,
so `EPSG:4326` and `OGC:CRS84` are rejected together. Comparing the derived coordinate spaces
instead, and adding `edges` to the comparison in phase 2, changes only that function.

Exact equality only works if no step drops a CRS on the way:

- **Literals.** A geometry literal pushed from an engine arrives as WKB. The WKB bytes carry no
  CRS, but the literal's `WellKnownBinary` dtype does, and `native_geometry_scalar_from_wkb` takes
  that scalar and keeps its CRS. DuckDB already casts a CRS-less literal to the column's CRS
  before pushdown. Literals that truly have no CRS, such as a parsed WKT string or a JNI literal,
  stay unreferenced; giving them a CRS is the engine's decision.
- **Outputs.** A function that returns a geometry keeps its operands' CRS: `make_line`, `collect`,
  `convex_hull`, and `envelope`.

A union `Geometry` type would remove the special rule: every geometry operand would have the same
extension type, so the ordinary equal-dtype check would cover the CRS. It can be an extension type
whose storage emulates a sparse union (a `type_id` plus one nullable child per geometry kind),
which does not need a core union dtype.

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

Phase 1 is unaffected: the existing rules only match the flat functions, and the new spheroid
functions have no pruning rules. From phase 2, until these are handled, the pruning rules skip
columns whose edges are not planar.

## Implementation plan

### Phase 1a: flat CRSs

Every projected CRS already computes correctly with K1, in its own plane and units. So in phase 1a
Vortex does not interpret any CRS. It stores the CRS and checks that the two inputs of a binary
function share one (see [CRS check](#crs-check)). Which CRSs users can work with is decided by the
engine: it is the set its `ST_Transform` can convert between.

| CRS | Phase 1a behavior |
| --- | --- |
| Projected (UTM, State Plane, Web Mercator, national grids, polar, equal-area) | Supported: flat results in the CRS's units; see the projection table in [Behavior worth knowing](#behavior-worth-knowing) |
| Engineering, local, or pixel space | Supported: flat by definition |
| Geographic (EPSG:4326, OGC:CRS84) | Unchanged: planar edges, distances in degrees. Rejecting them would break existing data and tests that tag points with EPSG:4326 |
| Geocentric (EPSG:4978) | Not supported: needs 3D distance, and the kernels read only `x` and `y` |

Vortex:

1. Reject functions over two geometries whose CRSs differ.
2. Test that results follow the CRS's units: the same two points give distances in EPSG:2263
   (US survey feet) and EPSG:32118 (metres) whose ratio is the foot-to-metre factor, about 3.2808.

Engine, starting with DataFusion:

1. Read `crs` from field metadata during planning.
2. Implement `ST_Transform` between projected CRSs, such as two UTM zones or EPSG:2263 and
   EPSG:32118, setting the output's `crs`.
3. Refuse functions over different CRSs, with an error that points at `ST_Transform`.

### Phase 1b: curved distance for the WGS 84 family

Phase 1b supports CRSs that share the WGS 84 datum:

| CRS | Coordinate space | Why |
| --- | --- | --- |
| EPSG:4326, OGC:CRS84 | Geographic, WGS 84 ellipsoid | The most common CRS; GeoParquet's default |
| WGS 84 / UTM, EPSG:32601 to 32660 (north) and 32701 to 32760 (south) | Planar, metres | Satellite tiles; the running example |
| EPSG:3857 (Web Mercator), optional | Planar, metres | Web map tiles; a clear case of flat distance differing from ground distance |

A single datum keeps phase 1b small:

- **One ellipsoid:** K3 has one fixed pair of parameters.
- **No datum transforms:** every transform between these CRSs is projection math (transverse
  Mercator, Mercator), with no Helmert parameters and no correction grids. The engine may manage
  with the pure-Rust `proj4rs` instead of PROJ; this is not yet verified.
- **Enough levels to test:** EPSG:4326 and OGC:CRS84 differ by an axis swap (level 0), UTM north
  and south by a shift (level 2), and UTM or Web Mercator and EPSG:4326 by a projection (level 5).

Phase 1b adds no `edges`. The curved-surface kernel is chosen by function name, as DuckDB spatial
does with `ST_Distance_Spheroid`: `vortex.st.distance` stays flat, and new spheroid functions run
K3. This changes no existing result and no storage, and it covers the running example, whose two
points have no edges to interpret.

Vortex:

1. Derive `CoordSpace` for the family above (see
   [What Vortex reads from a CRS](#what-vortex-reads-from-a-crs)). Any other CRS is unknown and
   keeps today's flat behavior.
2. Add `vortex.st.distance_spheroid` and `vortex.st.length_spheroid`, which run K3 and require a
   geographic CRS. They cover point-to-point distance and line length, the cases with closed-form
   geodesic algorithms.
3. Reject GeoArrow input whose `edges` is not `planar`. Import drops `edges` today
   (`spatial_metadata_from_arrow`), so spherical data is computed on as planar without an error.
   Rejecting it is a behavior change.

Engine:

1. Extend `ST_Transform` to and from EPSG:4326.
2. Run the example end to end, and compare K1 in UTM with K3 in EPSG:4326.

Vortex steps do not depend on the engine and can land first.

### Phase 2: edges and curved predicates

1. Add `edges` to `SpatialMetadata` (see [Storage](#storage)), keep it through Arrow export and
   import, and include it in the CRS check.
2. Choose kernels from `(kind, edges)`, so `vortex.st.distance` picks K3 for geodesic edges. The
   spheroid functions remain as an explicit spelling.
3. Add K2 sphere predicates, then bounding-box statistics and pruning that are safe for curved
   edges.
4. Map any CRS given as PROJJSON, and add CRSs on other datums, which need PROJ.
5. Transform literals in the engine for the safe cases of [What the planner does](#what-the-planner-does).
6. Consider `vortex.st.affine` for the affine part of a pipeline.

## Open questions

- **Asking for geodesic edges in phase 2.** Phase 1 uses separate spheroid functions. Once `edges`
  exists, how does a query set it after `ST_Transform`: a cast, as PostGIS does with `geography`,
  or a function that sets `edges`?
- **Data without a CRS.** GeoParquet treats a missing CRS as OGC:CRS84; GeoArrow treats it as
  unknown. SpatialBench data has none. Treating it as unknown keeps today's results; the engine
  could write OGC:CRS84 explicitly when it reads GeoParquet.
- **Web Mercator in phase 1b.** The running example does not need it.
- **Where the engine work lives.** `vortex-datafusion` in this repository, or SpiralDB.
- **K3 implementation.** The [computation core epic](https://github.com/vortex-data/vortex/issues/9388)
  plans to move `geo` to a dev-dependency. If that stands, K3 depends on `geographiclib-rs`
  directly, which `geo::Geodesic` wraps, and `geo` serves as the test oracle.
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
