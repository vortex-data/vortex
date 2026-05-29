
WITH RECURSIVE MovieHierarchy AS (
    SELECT 
        mt.id AS movie_id,
        mt.title,
        mt.production_year,
        CAST(NULL AS integer) AS parent_id,
        1 AS level
    FROM aka_title mt
    WHERE mt.production_year > 1990

    UNION ALL

    SELECT 
        m.id,
        m.title,
        m.production_year,
        mh.movie_id,
        mh.level + 1
    FROM aka_title m
    JOIN MovieHierarchy mh ON m.episode_of_id = mh.movie_id
),

RoleCounts AS (
    SELECT 
        c.movie_id,
        rc.role,
        COUNT(c.person_id) AS role_count
    FROM cast_info c
    JOIN role_type rc ON c.role_id = rc.id
    GROUP BY c.movie_id, rc.role
),

TopDirectors AS (
    SELECT 
        ci.movie_id,
        ak.name,
        COUNT(*) AS director_count
    FROM cast_info ci
    JOIN aka_name ak ON ci.person_id = ak.person_id
    WHERE ci.person_role_id = (SELECT id FROM role_type WHERE role = 'Director')
    GROUP BY ci.movie_id, ak.name
    HAVING COUNT(*) > 1
),

MovieDetails AS (
    SELECT 
        th.movie_id,
        th.title,
        th.production_year,
        COALESCE(SUM(rc.role_count), 0) AS total_roles,
        COALESCE(MAX(td.director_count), 0) AS max_directors
    FROM MovieHierarchy th
    LEFT JOIN RoleCounts rc ON th.movie_id = rc.movie_id
    LEFT JOIN TopDirectors td ON th.movie_id = td.movie_id
    GROUP BY th.movie_id, th.title, th.production_year
)

SELECT 
    md.title,
    md.production_year,
    md.total_roles,
    md.max_directors,
    CASE 
        WHEN md.max_directors > 0 THEN 'Multi-Directed' 
        ELSE 'Single-Directed or Not Available' 
    END AS director_status
FROM MovieDetails md
WHERE md.total_roles > 5
ORDER BY md.production_year DESC, md.total_roles DESC
LIMIT 10;
