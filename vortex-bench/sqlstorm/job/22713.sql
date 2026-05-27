WITH recursive movie_actors AS (
    SELECT 
        ca.movie_id,
        ka.name AS actor_name,
        COUNT(DISTINCT ca.role_id) AS role_count
    FROM 
        cast_info ca
    JOIN 
        aka_name ka ON ca.person_id = ka.person_id
    GROUP BY 
        ca.movie_id, ka.name
),
high_role_movies AS (
    SELECT 
        ma.movie_id,
        ma.actor_name,
        ma.role_count
    FROM 
        movie_actors ma
    WHERE 
        ma.role_count > (SELECT AVG(role_count) FROM movie_actors)
),
movie_details AS (
    SELECT 
        ht.title,
        ht.production_year,
        STRING_AGG(DISTINCT ha.actor_name, ', ') AS actors
    FROM 
        aka_title ht
    LEFT JOIN 
        high_role_movies ha ON ht.id = ha.movie_id
    GROUP BY 
        ht.title, ht.production_year
),
yearly_production AS (
    SELECT 
        md.production_year,
        COUNT(md.title) AS movie_count,
        MAX(md.title) AS latest_movie
    FROM 
        movie_details md
    GROUP BY 
        md.production_year
)
SELECT 
    yp.production_year,
    yp.movie_count,
    yp.latest_movie,
    CASE 
        WHEN yp.movie_count IS NULL THEN 'No movies produced'
        WHEN yp.movie_count > 10 THEN 'High Production Year'
        ELSE 'Low Production Year'
    END AS production_category,
    COALESCE(NULLIF(yp.latest_movie, ''), 'No title available') AS latest_movie_title
FROM 
    yearly_production yp
ORDER BY 
    yp.production_year DESC;
