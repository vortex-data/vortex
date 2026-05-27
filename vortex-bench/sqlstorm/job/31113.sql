WITH RECURSIVE movie_hierarchy AS (
    SELECT 
        m.id AS movie_id,
        m.title AS movie_title,
        1 AS depth
    FROM 
        aka_title m 
    WHERE 
        m.production_year >= 2000
    UNION ALL
    SELECT 
        m.id AS movie_id,
        CONCAT(h.movie_title, ' -> ', m.title) AS movie_title,
        h.depth + 1
    FROM 
        aka_title m
    INNER JOIN 
        movie_link ml ON m.id = ml.linked_movie_id
    INNER JOIN 
        movie_hierarchy h ON ml.movie_id = h.movie_id
    WHERE 
        h.depth < 3 
),
cast_roles AS (
    SELECT
        c.person_id,
        c.movie_id,
        r.role AS role_name,
        ROW_NUMBER() OVER (PARTITION BY c.person_id ORDER BY c.nr_order) AS role_order
    FROM 
        cast_info c
    INNER JOIN 
        role_type r ON c.role_id = r.id
),
top_cast AS (
    SELECT
        a.name AS actor_name,
        COUNT(DISTINCT cr.movie_id) AS num_movies,
        AVG(m.production_year) AS avg_production_year
    FROM 
        aka_name a
    JOIN 
        cast_roles cr ON a.person_id = cr.person_id
    JOIN 
        aka_title m ON cr.movie_id = m.id
    GROUP BY 
        a.name
    HAVING 
        COUNT(DISTINCT cr.movie_id) > 3 
),
cast_details AS (
    SELECT 
        a.name AS actor_name,
        mh.movie_title,
        mh.depth,
        CASE 
            WHEN cr.role_order = 1 THEN 'Lead'
            WHEN cr.role_order BETWEEN 2 AND 3 THEN 'Supporting'
            ELSE 'Cameo'
        END AS role_type
    FROM 
        movie_hierarchy mh
    JOIN 
        cast_roles cr ON mh.movie_id = cr.movie_id
    JOIN 
        aka_name a ON cr.person_id = a.person_id
    WHERE 
        mh.depth <= 2
),
combined_results AS (
    SELECT 
        t.actor_name, 
        t.movie_title,
        t.depth,
        t.role_type,
        COALESCE(tc.num_movies, 0) AS num_movies,
        COALESCE(tc.avg_production_year, 0) AS avg_production_year
    FROM 
        cast_details t
    LEFT JOIN 
        top_cast tc ON t.actor_name = tc.actor_name
)
SELECT
    actor_name,
    movie_title,
    depth,
    role_type,
    num_movies,
    avg_production_year
FROM 
    combined_results
WHERE 
    ((depth = 1 AND role_type = 'Lead') OR 
     (depth = 2 AND role_type = 'Supporting'))
ORDER BY 
    avg_production_year DESC, 
    actor_name;