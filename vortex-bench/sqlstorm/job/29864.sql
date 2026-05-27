WITH ranked_titles AS (
    SELECT 
        a.name AS actor_name,
        t.title AS movie_title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY a.person_id ORDER BY t.production_year DESC) AS rn
    FROM 
        aka_name a
    JOIN 
        cast_info ci ON a.person_id = ci.person_id
    JOIN 
        aka_title t ON ci.movie_id = t.movie_id
),
actor_movie_count AS (
    SELECT 
        actor_name,
        COUNT(*) AS movie_count
    FROM 
        ranked_titles
    WHERE 
        rn <= 5
    GROUP BY 
        actor_name
),
top_actors AS (
    SELECT 
        actor_name
    FROM 
        actor_movie_count
    WHERE 
        movie_count > 3
)
SELECT 
    ra.actor_name,
    STRING_AGG(rt.movie_title, ', ') AS movies,
    STRING_AGG(CONCAT(rt.movie_title, ' (', rt.production_year, ')'), ', ') AS movies_with_year
FROM 
    ranked_titles rt
JOIN 
    top_actors ra ON rt.actor_name = ra.actor_name
GROUP BY 
    ra.actor_name
ORDER BY 
    ra.actor_name;