
WITH RankedMovies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        t.kind_id,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS rank_per_year
    FROM 
        aka_title t
    WHERE 
        t.production_year IS NOT NULL
),
MovieCast AS (
    SELECT 
        m.movie_id,
        COUNT(c.person_id) AS cast_count,
        STRING_AGG(CONCAT(a.name, ' (', r.role, ')'), ', ') AS full_cast
    FROM 
        RankedMovies m
    JOIN 
        cast_info c ON m.movie_id = c.movie_id
    JOIN 
        aka_name a ON c.person_id = a.person_id
    JOIN 
        role_type r ON c.role_id = r.id
    GROUP BY 
        m.movie_id
),
TopMovies AS (
    SELECT 
        rm.movie_id,
        rm.title,
        rm.production_year,
        mc.cast_count,
        mc.full_cast,
        rm.kind_id
    FROM 
        RankedMovies rm
    JOIN 
        MovieCast mc ON rm.movie_id = mc.movie_id
    WHERE 
        rm.rank_per_year <= 3
)
SELECT 
    tm.title,
    tm.production_year,
    tm.cast_count,
    tm.full_cast,
    ki.kind 
FROM 
    TopMovies tm
JOIN 
    kind_type ki ON tm.kind_id = ki.id
ORDER BY 
    tm.production_year DESC, 
    tm.title;
