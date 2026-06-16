-- Q1: Find trips starting within 50km of the Sedona city center, ranked by distance.
SELECT
  t_tripkey,
  ST_X(ST_GeomFromWKB(t_pickuploc)) AS pickup_lon,
  ST_Y(ST_GeomFromWKB(t_pickuploc)) AS pickup_lat,
  t_pickuptime,
  ST_Distance(ST_GeomFromWKB(t_pickuploc), ST_Point(-111.7610::double, 34.8697::double)) AS distance_to_center
FROM trip
WHERE ST_Distance(ST_GeomFromWKB(t_pickuploc), ST_Point(-111.7610::double, 34.8697::double)) <= 0.45::double
ORDER BY distance_to_center ASC, t_tripkey ASC;
