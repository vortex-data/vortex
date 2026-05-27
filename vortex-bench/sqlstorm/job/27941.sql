WITH movie_details AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        a.name AS actor_name,
        r.role AS role_type,
        k.keyword AS movie_keyword
    FROM 
        aka_title t
    JOIN 
        cast_info ci ON t.id = ci.movie_id
    JOIN 
        aka_name a ON ci.person_id = a.person_id
    JOIN 
        role_type r ON ci.role_id = r.id
    LEFT JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id
    WHERE 
        t.production_year BETWEEN 2000 AND 2023
        AND LENGTH(a.name) > 5
),
aggregated_data AS (
    SELECT 
        production_year,
        COUNT(DISTINCT movie_id) AS total_movies,
        COUNT(DISTINCT actor_name) AS total_actors,
        STRING_AGG(DISTINCT movie_keyword, ', ') AS keywords
    FROM 
        movie_details
    GROUP BY 
        production_year
)
SELECT 
    ad.production_year,
    ad.total_movies,
    ad.total_actors,
    ad.keywords
FROM 
    aggregated_data ad
WHERE 
    ad.total_movies > 5
ORDER BY 
    ad.production_year DESC;
