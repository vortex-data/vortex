WITH MovieTitleInfo AS (
    SELECT 
        t.title AS movie_title,
        t.production_year,
        k.keyword AS movie_keyword,
        r.role AS cast_role,
        a.name AS actor_name,
        c.note AS cast_note
    FROM 
        aka_title t
    JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
    JOIN 
        cast_info c ON t.id = c.movie_id
    JOIN 
        aka_name a ON c.person_id = a.person_id
    JOIN 
        role_type r ON c.role_id = r.id
    WHERE 
        t.production_year BETWEEN 2000 AND 2023
),
AggregateKeywordCount AS (
    SELECT 
        movie_title,
        production_year,
        STRING_AGG(movie_keyword, ', ') AS keywords,
        COUNT(movie_keyword) AS keyword_count
    FROM 
        MovieTitleInfo
    GROUP BY 
        movie_title, production_year
),
DetailedMovieInfo AS (
    SELECT 
        m.movie_title,
        m.production_year,
        m.keywords,
        m.keyword_count,
        COUNT(DISTINCT c.id) AS cast_count,
        STRING_AGG(DISTINCT a.name, ', ') AS all_actors
    FROM 
        AggregateKeywordCount m
    JOIN 
        aka_title t ON m.movie_title = t.title AND m.production_year = t.production_year
    JOIN 
        cast_info c ON t.id = c.movie_id
    JOIN 
        aka_name a ON c.person_id = a.person_id
    GROUP BY 
        m.movie_title, m.production_year, m.keywords, m.keyword_count
)
SELECT 
    d.movie_title,
    d.production_year,
    d.keywords,
    d.keyword_count,
    d.cast_count,
    d.all_actors
FROM 
    DetailedMovieInfo d
ORDER BY 
    d.production_year DESC, d.keyword_count DESC;
