WITH RankedMovies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY COUNT(ci.person_id) DESC) AS rn
    FROM 
        aka_title t
    JOIN 
        complete_cast cc ON t.id = cc.movie_id
    JOIN 
        cast_info ci ON cc.subject_id = ci.person_id
    GROUP BY 
        t.id, t.title, t.production_year
),
TopActors AS (
    SELECT 
        ak.name,
        ci.movie_id,
        COUNT(*) AS role_count
    FROM 
        aka_name ak
    JOIN 
        cast_info ci ON ak.person_id = ci.person_id
    GROUP BY 
        ak.name, ci.movie_id
    HAVING 
        COUNT(*) > 1
),
MoviesWithKeywords AS (
    SELECT 
        mt.movie_id,
        STRING_AGG(k.keyword, ', ') AS keywords
    FROM 
        movie_keyword mt
    JOIN 
        keyword k ON mt.keyword_id = k.id
    GROUP BY 
        mt.movie_id
)
SELECT 
    rm.title, 
    rm.production_year,
    CASE 
        WHEN t.role_count IS NULL THEN 'No prominent actor'
        ELSE t.name
    END AS prominent_actor,
    COALESCE(mkw.keywords, 'No keywords') AS keywords
FROM 
    RankedMovies rm
LEFT JOIN 
    TopActors t ON rm.movie_id = t.movie_id AND rm.rn = 1
LEFT JOIN 
    MoviesWithKeywords mkw ON rm.movie_id = mkw.movie_id
WHERE 
    rm.production_year >= 2000
ORDER BY 
    rm.production_year DESC, 
    rm.title;
