
WITH ranked_movies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        COUNT(DISTINCT ci.person_id) AS total_cast,
        RANK() OVER (PARTITION BY t.production_year ORDER BY COUNT(DISTINCT ci.person_id) DESC) AS rank_within_year
    FROM 
        aka_title t
    LEFT JOIN 
        cast_info ci ON t.id = ci.movie_id
    GROUP BY 
        t.id, t.title, t.production_year
),
high_cast_movies AS (
    SELECT 
        rm.movie_id, 
        rm.title,
        rm.production_year,
        rm.total_cast
    FROM 
        ranked_movies rm
    WHERE 
        rm.rank_within_year <= 5
),
movie_keywords AS (
    SELECT 
        mk.movie_id, 
        STRING_AGG(k.keyword, ', ') AS keywords
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
),
final_result AS (
    SELECT 
        hcm.movie_id,
        hcm.title,
        hcm.production_year,
        hcm.total_cast,
        COALESCE(mk.keywords, 'No Keywords') AS keywords
    FROM 
        high_cast_movies hcm
    LEFT JOIN 
        movie_keywords mk ON hcm.movie_id = mk.movie_id
)
SELECT 
    f.movie_id,
    f.title,
    f.production_year,
    f.total_cast,
    f.keywords,
    CASE 
        WHEN f.total_cast IS NULL THEN 'Unknown'
        ELSE CAST(f.total_cast AS VARCHAR) || ' Cast Members'
    END AS cast_info
FROM 
    final_result f
ORDER BY 
    f.production_year DESC, 
    f.total_cast DESC;
