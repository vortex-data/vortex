WITH ranked_titles AS (
    SELECT 
        a.title AS movie_title,
        t.production_year,
        r.role,
        a.id AS title_id,
        ROW_NUMBER() OVER (PARTITION BY a.id ORDER BY t.production_year DESC) AS year_rank
    FROM 
        aka_title a
    JOIN 
        title t ON a.movie_id = t.id
    JOIN 
        cast_info ci ON a.movie_id = ci.movie_id
    JOIN 
        role_type r ON ci.role_id = r.id
    WHERE 
        t.production_year IS NOT NULL
), 
top_movies AS (
    SELECT 
        movie_title,
        production_year,
        role,
        title_id
    FROM 
        ranked_titles
    WHERE 
        year_rank = 1
), 
keyword_summary AS (
    SELECT 
        mk.movie_id,
        STRING_AGG(k.keyword, ', ') AS keywords
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
)
SELECT 
    t.movie_title,
    t.production_year,
    t.role,
    k.keywords
FROM 
    top_movies t
LEFT JOIN 
    keyword_summary k ON t.title_id = k.movie_id
ORDER BY 
    t.production_year DESC, 
    t.movie_title;
