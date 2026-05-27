WITH movie_actors AS (
    SELECT 
        a.name AS actor_name,
        t.title AS movie_title,
        t.production_year,
        c.nr_order,
        p.info AS actor_info
    FROM 
        aka_name a
    JOIN 
        cast_info c ON a.person_id = c.person_id
    JOIN 
        aka_title t ON c.movie_id = t.movie_id
    JOIN 
        person_info p ON c.person_id = p.person_id
    WHERE 
        p.info_type_id IN (SELECT id FROM info_type WHERE info LIKE '%Biography%')
), 
top_movies AS (
    SELECT 
        movie_title,
        production_year,
        COUNT(*) AS actor_count
    FROM 
        movie_actors
    GROUP BY 
        movie_title, production_year
    ORDER BY 
        actor_count DESC
    LIMIT 10
)
SELECT 
    t.movie_title,
    t.production_year,
    COUNT(ma.actor_name) AS total_actors,
    STRING_AGG(ma.actor_name, ', ') AS actor_list
FROM 
    top_movies t
JOIN 
    movie_actors ma ON t.movie_title = ma.movie_title AND t.production_year = ma.production_year
GROUP BY 
    t.movie_title, t.production_year
ORDER BY 
    total_actors DESC;
