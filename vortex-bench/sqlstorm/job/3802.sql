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
CastDetails AS (
    SELECT 
        c.movie_id,
        COUNT(*) AS cast_count,
        MIN(a.name) AS first_actor_name,
        MAX(a.name) AS last_actor_name
    FROM 
        cast_info c
    JOIN 
        aka_name a ON c.person_id = a.person_id
    GROUP BY 
        c.movie_id
),
MovieKeywords AS (
    SELECT 
        mk.movie_id,
        STRING_AGG(k.keyword, ', ') AS all_keywords
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
),
MoviesWithDetails AS (
    SELECT 
        rm.movie_id,
        rm.title,
        rm.production_year,
        cd.cast_count,
        cd.first_actor_name,
        cd.last_actor_name,
        COALESCE(mk.all_keywords, 'No Keywords') AS keywords
    FROM 
        RankedMovies rm
    LEFT JOIN 
        CastDetails cd ON rm.movie_id = cd.movie_id
    LEFT JOIN 
        MovieKeywords mk ON rm.movie_id = mk.movie_id
)
SELECT 
    m.title,
    m.production_year,
    m.cast_count,
    m.first_actor_name,
    m.last_actor_name,
    m.keywords
FROM 
    MoviesWithDetails m
WHERE 
    m.production_year = (SELECT MAX(production_year) FROM RankedMovies)
ORDER BY 
    m.cast_count DESC, m.title;
