WITH RECURSIVE MovieHierarchy AS (
    SELECT t.id AS movie_id,
           t.title,
           t.production_year,
           t.kind_id,
           0 AS hierarchy_level,
           t.id AS root_movie_id
    FROM aka_title t
    WHERE t.production_year IS NOT NULL
    UNION ALL
    SELECT t.id AS movie_id,
           t.title,
           t.production_year,
           t.kind_id,
           mh.hierarchy_level + 1,
           mh.root_movie_id
    FROM aka_title t
    JOIN movie_link ml ON t.id = ml.linked_movie_id
    JOIN MovieHierarchy mh ON ml.movie_id = mh.movie_id
    WHERE mh.hierarchy_level < 5
)
, CastDetails AS (
    SELECT ci.movie_id,
           COUNT(DISTINCT ci.person_id) AS actor_count,
           COUNT(DISTINCT ci.person_role_id) AS role_count,
           SUM(CASE WHEN ci.note IS NOT NULL THEN 1 ELSE 0 END) AS note_count
    FROM cast_info ci
    GROUP BY ci.movie_id
),
DetailedMovies AS (
    SELECT mh.movie_id,
           mh.title,
           mh.production_year,
           kt.kind AS genre,
           cd.actor_count,
           cd.role_count,
           cd.note_count,
           ROW_NUMBER() OVER (PARTITION BY mh.production_year ORDER BY cd.actor_count DESC) AS rank_per_year
    FROM MovieHierarchy mh
    LEFT JOIN kind_type kt ON mh.kind_id = kt.id
    LEFT JOIN CastDetails cd ON mh.movie_id = cd.movie_id
)
SELECT d.title,
       d.production_year,
       d.genre,
       d.actor_count,
       d.role_count,
       d.note_count,
       CASE 
           WHEN d.actor_count IS NULL THEN 'No actors'
           WHEN d.actor_count > 10 THEN 'Blockbuster'
           WHEN d.actor_count BETWEEN 1 AND 10 THEN 'Indie Film'
           ELSE 'Uncredited'
       END AS film_type,
       COALESCE(d.note_count, 0) AS actual_notes,
       FIRST_VALUE(d.title) OVER (PARTITION BY d.production_year ORDER BY d.actor_count DESC) AS top_film_current_year
FROM DetailedMovies d
WHERE (d.production_year > 2000 AND d.actor_count IS NOT NULL)
   OR (d.production_year <= 2000 AND d.role_count IS NULL)
ORDER BY d.production_year, d.actor_count DESC
LIMIT 100;