WITH RankedMovies AS (
    SELECT 
        a.id AS movie_id,
        a.title AS movie_title,
        a.production_year,
        k.keyword AS movie_keyword,
        COUNT(ci.id) AS cast_count
    FROM 
        aka_title a
    LEFT JOIN 
        movie_keyword mk ON a.id = mk.movie_id
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id
    LEFT JOIN 
        cast_info ci ON a.id = ci.movie_id
    GROUP BY 
        a.id, a.title, a.production_year, k.keyword
), FilteredMovies AS (
    SELECT 
        movie_id,
        movie_title,
        production_year,
        movie_keyword,
        cast_count,
        ROW_NUMBER() OVER (PARTITION BY production_year ORDER BY cast_count DESC) AS rnk
    FROM 
        RankedMovies
    WHERE 
        production_year IS NOT NULL
)

SELECT 
    f.movie_title,
    f.production_year,
    f.movie_keyword,
    f.cast_count
FROM 
    FilteredMovies f
WHERE 
    f.rnk <= 5
ORDER BY 
    f.production_year DESC, 
    f.cast_count DESC;
