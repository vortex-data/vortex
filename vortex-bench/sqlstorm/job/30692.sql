
WITH RECURSIVE movie_hierarchy AS (
    SELECT 
        m.id AS movie_id,
        m.title,
        m.production_year,
        1 AS level
    FROM 
        aka_title m
    WHERE 
        m.episode_of_id IS NULL  
    UNION ALL
    SELECT 
        ep.id AS movie_id,
        ep.title,
        ep.production_year,
        mh.level + 1
    FROM 
        aka_title ep
    INNER JOIN 
        movie_hierarchy mh ON ep.episode_of_id = mh.movie_id
),
ranked_cast AS (
    SELECT 
        ci.movie_id,
        a.name AS actor_name,
        ROW_NUMBER() OVER (PARTITION BY ci.movie_id ORDER BY ci.nr_order) AS actor_rank
    FROM 
        cast_info ci
    JOIN 
        aka_name a ON ci.person_id = a.person_id
),
company_info AS (
    SELECT 
        mc.movie_id,
        STRING_AGG(cn.name, ', ') AS company_names,
        STRING_AGG(ct.kind, ', ') AS company_types
    FROM 
        movie_companies mc
    JOIN 
        company_name cn ON mc.company_id = cn.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
    GROUP BY 
        mc.movie_id
)
SELECT 
    mh.movie_id,
    mh.title,
    mh.production_year,
    COALESCE(rc.actor_count, 0) AS total_actors,
    COALESCE(ci.company_names, 'No Companies') AS companies,
    COALESCE(ci.company_types, 'N/A') AS company_types,
    RANK() OVER (ORDER BY mh.production_year DESC) AS production_rank
FROM 
    movie_hierarchy mh
LEFT JOIN (
    SELECT 
        movie_id, COUNT(*) AS actor_count
    FROM 
        ranked_cast
    GROUP BY 
        movie_id
) rc ON mh.movie_id = rc.movie_id
LEFT JOIN 
    company_info ci ON mh.movie_id = ci.movie_id
WHERE 
    mh.production_year >= 2000
GROUP BY 
    mh.movie_id, mh.title, mh.production_year, rc.actor_count, ci.company_names, ci.company_types
ORDER BY 
    mh.title ASC, 
    production_rank DESC;
