
WITH RankedMovies AS (
    SELECT 
        a.title, 
        a.production_year, 
        ROW_NUMBER() OVER (PARTITION BY a.production_year ORDER BY a.title) AS rank,
        COALESCE(m.title, 'Unknown') AS linked_movie_title,
        a.id
    FROM aka_title a
    LEFT JOIN movie_link ml ON a.id = ml.movie_id
    LEFT JOIN aka_title m ON ml.linked_movie_id = m.id
), 
CastStats AS (
    SELECT 
        c.movie_id,
        COUNT(c.person_id) AS total_cast,
        MAX(c.nr_order) AS highest_order,
        MIN(c.nr_order) AS lowest_order
    FROM cast_info c
    GROUP BY c.movie_id
), 
CompanyInfo AS (
    SELECT 
        mc.movie_id,
        COUNT(DISTINCT cp.country_code) AS unique_companies,
        STRING_AGG(DISTINCT cn.name, ', ') AS company_names
    FROM movie_companies mc
    JOIN company_name cn ON mc.company_id = cn.id
    JOIN company_type ct ON mc.company_type_id = ct.id
    LEFT JOIN company_name cp ON cp.id = mc.company_id
    GROUP BY mc.movie_id
)
SELECT 
    rm.title AS movie_title,
    rm.production_year,
    cs.total_cast,
    cs.highest_order,
    cs.lowest_order,
    ci.unique_companies,
    ci.company_names,
    CASE 
        WHEN cs.highest_order IS NULL THEN 'No cast members'
        ELSE 'Has cast members'
    END AS cast_status,
    COALESCE(rm.linked_movie_title, 'N/A') AS linked_movie
FROM RankedMovies rm
LEFT JOIN CastStats cs ON rm.id = cs.movie_id
LEFT JOIN CompanyInfo ci ON rm.id = ci.movie_id
WHERE rm.rank <= 5
ORDER BY rm.production_year DESC, rm.title;
