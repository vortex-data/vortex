WITH RankedMovies AS (
    SELECT 
        t.id AS title_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY SUM(COALESCE(mk.id, 0)) DESC) AS rank_by_keywords,
        COUNT(DISTINCT ci.person_id) AS total_cast_members,
        SUM(CASE WHEN ci.note IS NOT NULL THEN 1 ELSE 0 END) AS has_notes
    FROM 
        aka_title t
    LEFT JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    LEFT JOIN 
        complete_cast cc ON t.id = cc.movie_id
    LEFT JOIN 
        cast_info ci ON cc.subject_id = ci.id
    GROUP BY 
        t.id, t.title, t.production_year
), 
MoviesWithNotes AS (
    SELECT 
        rm.title_id,
        rm.title,
        rm.production_year,
        rm.rank_by_keywords,
        rm.total_cast_members,
        rm.has_notes,
        CASE 
            WHEN rm.total_cast_members > 5 AND rm.has_notes > 0 THEN 'Highly Casted with Notes'
            WHEN rm.total_cast_members > 5 THEN 'Highly Casted'
            WHEN rm.has_notes > 0 THEN 'Few Casted with Notes'
            ELSE 'Few Casted'
        END AS cast_category
    FROM 
        RankedMovies rm
), 
DistinctCompanies AS (
    SELECT 
        mc.movie_id,
        COUNT(DISTINCT c.name) AS unique_company_count
    FROM 
        movie_companies mc
    JOIN 
        company_name c ON mc.company_id = c.id
    GROUP BY 
        mc.movie_id
)
SELECT 
    mw.title,
    mw.production_year,
    mw.cast_category,
    COALESCE(dc.unique_company_count, 0) AS number_of_companies,
    CASE 
        WHEN mw.rank_by_keywords < 5 THEN 'Consider Watching'
        WHEN mw.rank_by_keywords BETWEEN 5 AND 10 THEN 'Mainstream Pick'
        ELSE 'Top Choice!'
    END AS recommendation
FROM 
    MoviesWithNotes mw
LEFT JOIN 
    DistinctCompanies dc ON mw.title_id = dc.movie_id
WHERE 
    mw.production_year IS NOT NULL
    AND mw.cast_category IN ('Highly Casted with Notes', 'Highly Casted')
ORDER BY 
    mw.production_year DESC,
    mw.rank_by_keywords;
