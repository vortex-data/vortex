WITH ranked_movies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        RANK() OVER (PARTITION BY t.production_year ORDER BY COUNT(c.id) DESC) AS rank_by_cast_size
    FROM 
        aka_title t
    LEFT JOIN 
        cast_info c ON t.movie_id = c.movie_id
    GROUP BY 
        t.id, t.title, t.production_year
),
movie_keyword_info AS (
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
multi_word_titles AS (
    SELECT 
        m.id AS movie_id,
        m.title,
        LENGTH(m.title) - LENGTH(REPLACE(m.title, ' ', '')) + 1 AS word_count
    FROM 
        aka_title m
    WHERE 
        LENGTH(m.title) - LENGTH(REPLACE(m.title, ' ', '')) + 1 > 2
),
combination_results AS (
    SELECT 
        r.movie_id,
        r.title,
        COALESCE(k.keywords, 'No Keywords') AS keywords,
        r.production_year,
        rw.word_count,
        COUNT(c.person_id) AS cast_size,
        CASE 
            WHEN COUNT(c.person_id) > 10 THEN 'Large Cast'
            ELSE 'Small Cast'
        END AS cast_size_category
    FROM 
        ranked_movies r
    LEFT JOIN 
        movie_keyword_info k ON r.movie_id = k.movie_id
    LEFT JOIN 
        multi_word_titles rw ON r.movie_id = rw.movie_id
    LEFT JOIN 
        cast_info c ON r.movie_id = c.movie_id
    WHERE 
        r.rank_by_cast_size = 1 
    GROUP BY 
        r.movie_id, r.title, k.keywords, r.production_year, rw.word_count
)
SELECT 
    title,
    production_year,
    keywords,
    cast_size,
    CASE
        WHEN word_count IS NULL THEN 'Unknown Word Count'
        ELSE CAST(word_count AS TEXT)
    END AS word_count,
    cast_size_category
FROM 
    combination_results
ORDER BY 
    production_year DESC, title;