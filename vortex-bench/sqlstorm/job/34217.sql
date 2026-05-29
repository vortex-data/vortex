
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
        m.id AS movie_id, 
        m.title, 
        m.production_year, 
        mh.level + 1
    FROM 
        aka_title m
    JOIN 
        movie_hierarchy mh ON m.episode_of_id = mh.movie_id
),
cast_details AS (
    SELECT 
        c.movie_id,
        a.name AS actor_name,
        a.surname_pcode
    FROM 
        cast_info c
    JOIN 
        aka_name a ON a.person_id = c.person_id
),
movie_info_summary AS (
    SELECT 
        m.id AS movie_id,
        COUNT(DISTINCT k.keyword) AS total_keywords,
        MAX(m.production_year) AS latest_info_year
    FROM 
        aka_title m
    LEFT JOIN 
        movie_keyword mk ON mk.movie_id = m.id
    LEFT JOIN 
        keyword k ON k.id = mk.keyword_id
    GROUP BY 
        m.id
)
SELECT 
    mh.movie_id,
    mh.title,
    mh.production_year,
    cd.actor_name,
    cd.surname_pcode,
    mi.total_keywords,
    CASE 
        WHEN mi.latest_info_year IS NULL THEN 'No info available'
        ELSE CAST(mi.latest_info_year AS VARCHAR)
    END AS latest_info_year,
    ROW_NUMBER() OVER (PARTITION BY mh.level ORDER BY mh.production_year DESC) AS rank_level
FROM 
    movie_hierarchy mh
LEFT JOIN 
    cast_details cd ON cd.movie_id = mh.movie_id
LEFT JOIN 
    movie_info_summary mi ON mi.movie_id = mh.movie_id
WHERE 
    mh.production_year >= 2000
  AND 
    cd.surname_pcode IS NOT NULL
ORDER BY 
    mh.level, 
    mh.production_year DESC;
