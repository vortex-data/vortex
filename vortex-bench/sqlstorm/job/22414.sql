WITH RECURSIVE top_movie_titles AS (
    SELECT 
        t.id AS title_id, 
        t.title, 
        COALESCE(t.production_year, 0) AS production_year,
        'N/A' AS cast_member
    FROM title t
    WHERE t.production_year IS NOT NULL
    UNION ALL
    SELECT 
        t.id, 
        t.title, 
        t.production_year,
        ak.name AS cast_member
    FROM title t
    JOIN cast_info c ON c.movie_id = t.id
    JOIN aka_name ak ON ak.person_id = c.person_id
    WHERE ak.name IS NOT NULL
),
movie_keywords AS (
    SELECT 
        m.id AS movie_id,
        k.keyword,
        ROW_NUMBER() OVER (PARTITION BY m.id ORDER BY k.keyword) AS keyword_rank
    FROM movie_keyword mk
    JOIN keyword k ON k.id = mk.keyword_id
    JOIN aka_title m ON m.id = mk.movie_id
),
movie_company_data AS (
    SELECT 
        c.movie_id, 
        cn.name AS company_name,
        ct.kind AS company_type,
        ROW_NUMBER() OVER (PARTITION BY c.movie_id ORDER BY cn.name) AS company_rank
    FROM movie_companies c
    JOIN company_name cn ON cn.id = c.company_id
    JOIN company_type ct ON ct.id = c.company_type_id
),
filtered_titles AS (
    SELECT 
        title_id,
        title,
        production_year,
        STRING_AGG(cast_member, ', ') AS cast_list
    FROM top_movie_titles
    GROUP BY title_id, title, production_year
    HAVING COUNT(*) > 1
)
SELECT 
    ft.title_id,
    ft.title,
    ft.production_year,
    COALESCE(mk.keyword, 'No Keywords') AS keyword,
    COALESCE(mcd.company_name, 'Unknown Company') AS company_name,
    COALESCE(mcd.company_type, 'Unknown Type') AS company_type
FROM filtered_titles ft
LEFT JOIN movie_keywords mk ON mk.movie_id = ft.title_id AND mk.keyword_rank = 1
LEFT JOIN movie_company_data mcd ON mcd.movie_id = ft.title_id AND mcd.company_rank = 1
WHERE ft.production_year > 2000
ORDER BY ft.production_year DESC, ft.title;
