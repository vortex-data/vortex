WITH ranked_movies AS (
    SELECT 
        mt.id AS movie_id,
        mt.title AS movie_title,
        mt.production_year,
        ROW_NUMBER() OVER (PARTITION BY mt.production_year ORDER BY mt.title) AS title_rank,
        COUNT(*) OVER (PARTITION BY mt.production_year) AS total_movies
    FROM aka_title mt
    WHERE mt.production_year IS NOT NULL
),
cast_details AS (
    SELECT 
        c.movie_id,
        c.person_id,
        ak.name AS actor_name,
        rc.role AS role_type,
        COALESCE(CAST(COUNT(DISTINCT c.id) AS TEXT), '0') AS role_count
    FROM cast_info c
    INNER JOIN aka_name ak ON c.person_id = ak.person_id
    LEFT JOIN role_type rc ON c.role_id = rc.id
    GROUP BY c.movie_id, c.person_id, ak.name, rc.role
),
movie_keywords AS (
    SELECT 
        mk.movie_id,
        STRING_AGG(k.keyword, ', ') AS keywords
    FROM movie_keyword mk
    JOIN keyword k ON mk.keyword_id = k.id
    GROUP BY mk.movie_id
),
selected_movies AS (
    SELECT 
        rm.movie_id,
        rm.movie_title,
        rm.production_year,
        cd.actor_name,
        cd.role_type,
        mk.keywords,
        CASE 
            WHEN rm.title_rank <= 10 THEN 'Top Ten'
            WHEN rm.total_movies > 0 AND rm.title_rank > 10 THEN 'Non-Top Ten'
            ELSE 'Unknown Rank'
        END AS movie_rank_category
    FROM ranked_movies rm
    LEFT JOIN cast_details cd ON rm.movie_id = cd.movie_id
    LEFT JOIN movie_keywords mk ON rm.movie_id = mk.movie_id
)
SELECT 
    sm.movie_title,
    sm.production_year,
    sm.actor_name,
    sm.role_type,
    sm.keywords,
    sm.movie_rank_category
FROM selected_movies sm
WHERE (sm.production_year BETWEEN 2000 AND 2023)
    AND (sm.movie_rank_category = 'Top Ten' OR sm.keywords LIKE '%Action%')
ORDER BY sm.production_year DESC, sm.movie_title ASC
LIMIT 100;