WITH MovieDetails AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        COUNT(DISTINCT c.person_id) AS cast_count,
        STRING_AGG(DISTINCT ak.name, ', ') AS actor_names
    FROM 
        aka_title t
    LEFT JOIN 
        cast_info c ON t.id = c.movie_id
    LEFT JOIN 
        aka_name ak ON ak.person_id = c.person_id
    WHERE 
        t.production_year >= 2000
    GROUP BY 
        t.id, t.title, t.production_year
),
TopMovies AS (
    SELECT 
        movie_id,
        title,
        production_year,
        cast_count,
        actor_names,
        ROW_NUMBER() OVER (ORDER BY cast_count DESC) AS rn
    FROM 
        MovieDetails
)
SELECT 
    tm.title,
    tm.production_year,
    tm.cast_count,
    tm.actor_names,
    CASE 
        WHEN tm.cast_count > 5 THEN 'Blockbuster'
        WHEN tm.cast_count BETWEEN 3 AND 5 THEN 'Moderate'
        ELSE 'Low'
    END AS popularity_category
FROM 
    TopMovies tm
WHERE 
    tm.rn <= 10
UNION ALL
SELECT 
    'Total Movies' AS title,
    NULL AS production_year,
    COUNT(*) AS cast_count,
    NULL AS actor_names,
    'Aggregate' AS popularity_category
FROM 
    TopMovies
WHERE 
    cast_count IS NOT NULL;
