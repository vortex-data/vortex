
WITH RECURSIVE MoviesCTE AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        t.kind_id,
        COALESCE(SUM(CASE WHEN c.nr_order IS NOT NULL THEN 1 ELSE 0 END), 0) AS cast_count
    FROM 
        aka_title t
    LEFT JOIN 
        cast_info c ON t.id = c.movie_id
    WHERE 
        t.production_year IS NOT NULL
    GROUP BY 
        t.id, t.title, t.production_year, t.kind_id
    HAVING 
        t.production_year > 2000
), 
RankedMovies AS (
    SELECT 
        movie_id,
        title,
        production_year,
        cast_count,
        ROW_NUMBER() OVER (PARTITION BY production_year ORDER BY cast_count DESC) AS rank
    FROM 
        MoviesCTE
), 
TitleKeyword AS (
    SELECT 
        mt.movie_id,
        STRING_AGG(k.keyword, ', ') AS keywords
    FROM 
        movie_keyword mt
    JOIN 
        keyword k ON mt.keyword_id = k.id
    GROUP BY 
        mt.movie_id
), 
CompanyTitles AS (
    SELECT 
        m.title,
        c.name AS company_name,
        c.country_code
    FROM 
        aka_title m
    JOIN 
        movie_companies mc ON m.id = mc.movie_id
    JOIN 
        company_name c ON mc.company_id = c.id
    WHERE 
        c.country_code IS NOT NULL
)
SELECT 
    rm.title,
    rm.production_year,
    rm.cast_count,
    tk.keywords,
    ct.company_name,
    ct.country_code
FROM 
    RankedMovies rm
LEFT JOIN 
    TitleKeyword tk ON rm.movie_id = tk.movie_id
LEFT JOIN 
    CompanyTitles ct ON rm.title = ct.title
WHERE 
    (rm.rank <= 5 OR rm.cast_count >= 10) 
    AND COALESCE(ct.country_code, '') <> 'USA' 
    AND rm.production_year BETWEEN 2000 AND 2023 
ORDER BY 
    rm.production_year DESC, 
    rm.cast_count DESC
LIMIT 10;
