-- SpatialBench queries (Apache Sedona), WKB dialect. See sedona-spatialbench/docs/queries.md.
-- Numbered from Q0 (= SpatialBench Q1). Only Q0 is wired up today, the rest are not run yet.

-- Q0: Find trips starting within 50km of the Sedona city center, ranked by distance.
SELECT
  t_tripkey,
  ST_X(ST_GeomFromWKB(t_pickuploc)) AS pickup_lon,
  ST_Y(ST_GeomFromWKB(t_pickuploc)) AS pickup_lat,
  t_pickuptime,
  ST_Distance(ST_GeomFromWKB(t_pickuploc), ST_Point(-111.7610::double, 34.8697::double)) AS distance_to_center
FROM trip
WHERE ST_Distance(ST_GeomFromWKB(t_pickuploc), ST_Point(-111.7610::double, 34.8697::double)) <= 0.45::double
ORDER BY distance_to_center ASC, t_tripkey ASC;

-- Q1: Count trips starting within Coconino County (Arizona) zone.
SELECT COUNT(*) AS trip_count_in_coconino_county
FROM trip t
WHERE ST_Intersects(
    ST_GeomFromWKB(t.t_pickuploc),
    (
        SELECT ST_GeomFromWKB(z.z_boundary)
        FROM zone z
        WHERE z.z_name = 'Coconino County'
        LIMIT 1
    )
);

-- Q2: Monthly trip statistics within a 15km radius of the Sedona city center.
SELECT
    DATE_TRUNC('month', t.t_pickuptime) AS pickup_month,
    COUNT(t.t_tripkey) AS total_trips,
    AVG(t.t_distance) AS avg_distance,
    AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration,
    AVG(t.t_fare) AS avg_fare
FROM trip t
-- ST_DWithin(a, b, d) equals ST_Distance(a, b) <= d, written as the distance comparison so the
-- radius filter pushes into the scan (DuckDB stashes ST_DWithin's distance in bind_data, hiding it).
WHERE ST_Distance(
    ST_GeomFromWKB(t.t_pickuploc),
    ST_GeomFromText('POLYGON((
        -111.9060 34.7347, -111.6160 34.7347,
        -111.6160 35.0047, -111.9060 35.0047,
        -111.9060 34.7347
    ))') -- Bounding box around Sedona
) <= 0.045 -- 5km buffer in degrees
GROUP BY DATE_TRUNC('month', t.t_pickuptime)
ORDER BY pickup_month;

-- Q3: Zone distribution of top 1000 trips by tip amount.
SELECT
    z.z_zonekey,
    z.z_name,
    COUNT(*) AS trip_count
FROM
    zone z
    JOIN (
        SELECT t.t_pickuploc
        FROM trip t
        ORDER BY t.t_tip DESC, t.t_tripkey ASC
        LIMIT 1000
    ) top_trips
    ON ST_Within(
        ST_GeomFromWKB(top_trips.t_pickuploc),
        ST_GeomFromWKB(z.z_boundary)
    )
GROUP BY z.z_zonekey, z.z_name
ORDER BY trip_count DESC, z.z_zonekey ASC;

-- Q4: Monthly travel patterns for repeat customers (convex hull of dropoff locations).
SELECT
    c.c_custkey,
    c.c_name AS customer_name,
    DATE_TRUNC('month', t.t_pickuptime) AS pickup_month,
    ST_Area(
        ST_ConvexHull(ST_Collect(ST_GeomFromWKB(t.t_dropoffloc)))
    ) AS monthly_travel_hull_area,
    COUNT(*) as dropoff_count
FROM trip t
JOIN customer c
    ON t.t_custkey = c.c_custkey
GROUP BY c.c_custkey, c.c_name, pickup_month
HAVING dropoff_count > 5 -- Only include repeat customers
ORDER BY monthly_travel_hull_area DESC, c.c_custkey ASC;

-- Q5: Zone statistics for trips within a 50km radius of the Sedona city center.
SELECT
    z.z_zonekey,
    z.z_name,
    COUNT(t.t_tripkey) AS total_pickups,
    AVG(t.t_distance) AS avg_distance,
    AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration
FROM trip t, zone z
WHERE ST_Intersects(
    ST_GeomFromText('POLYGON((
        -112.2110 34.4197, -111.3110 34.4197,
        -111.3110 35.3197, -112.2110 35.3197,
        -112.2110 34.4197
    ))'), -- Bounding box around Sedona
    ST_GeomFromWKB(z.z_boundary)
  )
  AND ST_Within(
    ST_GeomFromWKB(t.t_pickuploc),
    ST_GeomFromWKB(z.z_boundary)
  )
GROUP BY z.z_zonekey, z.z_name
ORDER BY total_pickups DESC, z.z_zonekey ASC;

-- Q6: Detect potential route detours by comparing reported vs. geometric distances.
WITH trip_lengths AS (
    SELECT
        t.t_tripkey,
        t.t_distance AS reported_distance_m,
        ST_Length(
            ST_MakeLine(
                ST_GeomFromWKB(t.t_pickuploc),
                ST_GeomFromWKB(t.t_dropoffloc)
            )
        ) * 111111 AS line_distance_m -- Approx. meters per degree
    FROM trip t
)
SELECT
    t.t_tripkey,
    t.reported_distance_m,
    t.line_distance_m,
    t.reported_distance_m / NULLIF(t.line_distance_m, 0) AS detour_ratio
FROM trip_lengths t
ORDER BY
    detour_ratio DESC NULLS LAST,
    reported_distance_m DESC,
    t_tripkey ASC;

-- Q7: Count nearby pickups for each building within a 500m radius.
SELECT b.b_buildingkey, b.b_name, COUNT(*) AS nearby_pickup_count
FROM trip t
JOIN building b
ON ST_DWithin(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(b.b_boundary), 0.0045) -- ~500m
GROUP BY b.b_buildingkey, b.b_name
ORDER BY nearby_pickup_count DESC, b.b_buildingkey ASC;

-- Q8: Building conflation (duplicate/overlap detection via IoU).
WITH b1 AS (
   SELECT b_buildingkey AS id, ST_GeomFromWKB(b_boundary) AS geom
   FROM building
),
b2 AS (
    SELECT b_buildingkey AS id, ST_GeomFromWKB(b_boundary) AS geom
    FROM building
),
pairs AS (
    SELECT
        b1.id AS building_1,
        b2.id AS building_2,
        ST_Area(b1.geom) AS area1,
        ST_Area(b2.geom) AS area2,
        ST_Area(ST_Intersection(b1.geom, b2.geom)) AS overlap_area
    FROM b1
    JOIN b2 ON b1.id < b2.id AND ST_Intersects(b1.geom, b2.geom)
)
SELECT
   building_1,
   building_2,
   area1,
   area2,
   overlap_area,
   CASE
       WHEN (area1 + area2 - overlap_area) = 0 THEN 1.0
       ELSE overlap_area / (area1 + area2 - overlap_area)
   END AS iou
FROM pairs
ORDER BY iou DESC, building_1 ASC, building_2 ASC;

-- Q9: Zone statistics for trips starting within each zone.
SELECT
    z.z_zonekey,
    z.z_name AS pickup_zone,
    AVG(t.t_dropofftime - t.t_pickuptime) AS avg_duration,
    AVG(t.t_distance) AS avg_distance,
    COUNT(t.t_tripkey) AS num_trips
FROM
    zone z
    LEFT JOIN trip t
    ON ST_Within(
        ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(z.z_boundary)
    )
GROUP BY z.z_zonekey, z.z_name
ORDER BY avg_duration DESC NULLS LAST, z.z_zonekey ASC;

-- Q10: Count trips that cross between different zones.
SELECT COUNT(*) AS cross_zone_trip_count
FROM
    trip t
    JOIN zone pickup_zone
        ON ST_Within(
            ST_GeomFromWKB(t.t_pickuploc),
            ST_GeomFromWKB(pickup_zone.z_boundary)
        )
    JOIN zone dropoff_zone
        ON ST_Within(
            ST_GeomFromWKB(t.t_dropoffloc),
            ST_GeomFromWKB(dropoff_zone.z_boundary)
        )
WHERE pickup_zone.z_zonekey != dropoff_zone.z_zonekey;

-- Q11: Find five nearest buildings to each trip pickup location using KNN join.
WITH trip_with_geom AS (
    SELECT
        t_tripkey,
        t_pickuploc,
        ST_GeomFromWKB(t_pickuploc) as pickup_geom
    FROM trip
),
building_with_geom AS (
    SELECT
        b_buildingkey,
        b_name,
        b_boundary,
        ST_GeomFromWKB(b_boundary) as boundary_geom
    FROM building
)
SELECT
    t.t_tripkey,
    t.t_pickuploc,
    b.b_buildingkey,
    b.b_name AS building_name,
    ST_Distance(t.pickup_geom, b.boundary_geom) AS distance_to_building
FROM trip_with_geom t
JOIN building_with_geom b
    ON ST_KNN(t.pickup_geom, b.boundary_geom, 5, FALSE)
ORDER BY t.t_tripkey ASC, distance_to_building ASC, b.b_buildingkey ASC;
