
WITH RankedTitles AS (
    SELECT 
        t.id AS title_id,
        t.title,
        t.production_year,
        k.keyword,
        ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY k.keyword) AS keyword_rank
    FROM 
        title t
    JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
),

TitleCounts AS (
    SELECT
        rt.title_id,
        COUNT(rt.keyword) AS keyword_count
    FROM
        RankedTitles rt
    GROUP BY
        rt.title_id
),

MoviesWithHighKeywords AS (
    SELECT
        tc.title_id,
        t.title,
        t.production_year,
        tc.keyword_count
    FROM
        TitleCounts tc
    JOIN
        title t ON tc.title_id = t.id
    WHERE
        tc.keyword_count > 5 
),

MovieDetails AS (
    SELECT 
        m.title_id,
        c.name AS company_name,
        m.production_year,
        STRING_AGG(p.name, ', ') AS cast_names,
        m.keyword_count
    FROM 
        MoviesWithHighKeywords m
    LEFT JOIN 
        movie_companies mc ON mc.movie_id = m.title_id
    LEFT JOIN 
        company_name c ON mc.company_id = c.id
    LEFT JOIN 
        cast_info ci ON ci.movie_id = m.title_id
    LEFT JOIN 
        aka_name p ON ci.person_id = p.person_id
    GROUP BY 
        m.title_id, c.name, m.production_year, m.keyword_count
)

SELECT 
    md.title_id,
    md.company_name,
    md.production_year,
    md.cast_names,
    md.keyword_count
FROM 
    MovieDetails md
ORDER BY 
    md.production_year DESC,
    md.keyword_count DESC;
