WITH filtered_actors AS (
    SELECT 
        a.id AS actor_id,
        a.name AS actor_name,
        p.gender,
        COUNT(ci.movie_id) AS movies_count
    FROM 
        aka_name a
    JOIN 
        cast_info ci ON a.person_id = ci.person_id
    JOIN 
        name p ON a.person_id = p.imdb_id
    GROUP BY 
        a.id, a.name, p.gender
    HAVING 
        COUNT(ci.movie_id) > 3 
), 

top_movies AS (
    SELECT 
        m.id AS movie_id,
        m.title,
        m.production_year,
        COUNT(ci.person_id) AS cast_count
    FROM 
        aka_title m
    JOIN 
        cast_info ci ON m.id = ci.movie_id
    GROUP BY 
        m.id, m.title, m.production_year
    ORDER BY 
        cast_count DESC
    LIMIT 10 
)

SELECT 
    a.actor_name,
    a.gender,
    tm.title AS movie_title,
    tm.production_year,
    tm.cast_count
FROM 
    filtered_actors a
JOIN 
    cast_info ci ON a.actor_id = ci.person_id
JOIN 
    top_movies tm ON ci.movie_id = tm.movie_id
ORDER BY 
    a.actor_name, tm.production_year DESC;