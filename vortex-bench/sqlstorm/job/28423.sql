WITH RankedMovies AS (
    SELECT 
        m.id AS movie_id,
        m.title AS movie_title,
        m.production_year,
        COUNT(ci.person_id) AS cast_count,
        STRING_AGG(DISTINCT a.name, ', ') AS actor_names
    FROM 
        aka_title AS m
    JOIN 
        cast_info AS ci ON m.id = ci.movie_id
    JOIN 
        aka_name AS a ON ci.person_id = a.person_id
    WHERE 
        m.production_year BETWEEN 2000 AND 2023
    GROUP BY 
        m.id, m.title, m.production_year
),
MovieInfo AS (
    SELECT 
        ri.movie_id,
        ri.movie_title,
        ri.production_year,
        ri.cast_count,
        ri.actor_names,
        COUNT(DISTINCT mi.info_type_id) AS info_count,
        STRING_AGG(DISTINCT k.keyword, ', ') AS keywords
    FROM 
        RankedMovies AS ri
    LEFT JOIN 
        movie_info AS mi ON ri.movie_id = mi.movie_id
    LEFT JOIN 
        movie_keyword AS mk ON ri.movie_id = mk.movie_id
    LEFT JOIN 
        keyword AS k ON mk.keyword_id = k.id
    GROUP BY 
        ri.movie_id, ri.movie_title, ri.production_year, ri.cast_count, ri.actor_names
),
TopMovies AS (
    SELECT 
        movie_id,
        movie_title,
        production_year,
        cast_count,
        actor_names,
        info_count,
        keywords,
        RANK() OVER (ORDER BY cast_count DESC) AS rank_by_cast_count
    FROM 
        MovieInfo
)
SELECT 
    tm.movie_id,
    tm.movie_title,
    tm.production_year,
    tm.cast_count,
    tm.actor_names,
    tm.info_count,
    tm.keywords
FROM 
    TopMovies AS tm
WHERE 
    tm.rank_by_cast_count <= 10
ORDER BY 
    tm.production_year DESC, tm.cast_count DESC;
