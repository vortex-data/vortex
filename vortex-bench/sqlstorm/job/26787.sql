WITH RankedMovies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        COUNT(DISTINCT ci.person_id) AS cast_count,
        STRING_AGG(DISTINCT ak.name, ', ') AS actor_names
    FROM 
        aka_title t
    JOIN 
        cast_info ci ON ci.movie_id = t.id
    JOIN 
        aka_name ak ON ak.person_id = ci.person_id
    WHERE 
        t.production_year >= 2000
    GROUP BY 
        t.id, t.title, t.production_year
),
HighCastMovies AS (
    SELECT 
        rm.movie_id,
        rm.title,
        rm.production_year,
        rm.cast_count,
        rm.actor_names,
        ROW_NUMBER() OVER (ORDER BY rm.cast_count DESC) AS rank
    FROM 
        RankedMovies rm
    WHERE 
        rm.cast_count > 5
)
SELECT 
    hcm.title,
    hcm.production_year,
    hcm.cast_count,
    hcm.actor_names,
    k.keyword AS genre
FROM 
    HighCastMovies hcm
LEFT JOIN 
    movie_keyword mk ON mk.movie_id = hcm.movie_id
LEFT JOIN 
    keyword k ON k.id = mk.keyword_id
ORDER BY 
    hcm.rank, hcm.title;
