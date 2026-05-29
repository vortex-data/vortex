WITH RankedMovies AS (
    SELECT
        t.id AS movie_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS rank
    FROM
        aka_title t
    WHERE
        t.production_year IS NOT NULL
),
ActorCount AS (
    SELECT
        c.movie_id,
        COUNT(DISTINCT c.person_id) AS actor_count
    FROM
        cast_info c
    GROUP BY
        c.movie_id
),
MoviesWithActors AS (
    SELECT
        rm.movie_id,
        rm.title,
        rm.production_year,
        COALESCE(ac.actor_count, 0) AS actor_count
    FROM
        RankedMovies rm
    LEFT JOIN
        ActorCount ac ON rm.movie_id = ac.movie_id
)
SELECT
    mwa.movie_id,
    mwa.title,
    mwa.production_year,
    mwa.actor_count,
    CASE
        WHEN mwa.actor_count > 10 THEN 'Ensemble Cast'
        WHEN mwa.actor_count BETWEEN 5 AND 10 THEN 'Moderate Cast'
        ELSE 'Minimal Cast'
    END AS cast_size_description
FROM
    MoviesWithActors mwa
WHERE
    mwa.actor_count > (SELECT AVG(actor_count) FROM ActorCount)
ORDER BY
    mwa.production_year DESC, mwa.actor_count DESC;
