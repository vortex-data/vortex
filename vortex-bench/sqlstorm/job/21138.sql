
WITH RankedMovies AS (
    SELECT
        t.id AS movie_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS rank_within_year
    FROM
        aka_title t
    WHERE
        t.production_year >= 2000
        AND t.production_year <= 2023
),
CastDetails AS (
    SELECT
        ci.movie_id,
        c.name AS actor_name,
        COUNT(ci.id) AS role_count,
        SUM(CASE WHEN ci.note IS NULL THEN 1 ELSE 0 END) AS null_notes_count
    FROM
        cast_info ci
    JOIN
        aka_name c ON ci.person_id = c.person_id
    GROUP BY
        ci.movie_id, c.name
),
CompanyInfo AS (
    SELECT
        mc.movie_id,
        cc.name AS company_name,
        ct.kind AS company_type,
        COALESCE(NULLIF(mc.note, ''), 'No Note') AS company_note
    FROM
        movie_companies mc
    JOIN
        company_name cc ON mc.company_id = cc.id
    JOIN
        company_type ct ON mc.company_type_id = ct.id
)
SELECT
    rm.title,
    rm.production_year,
    cd.actor_name,
    cd.role_count,
    cd.null_notes_count,
    ci.company_name,
    ci.company_type,
    ci.company_note
FROM
    RankedMovies rm
LEFT JOIN
    CastDetails cd ON rm.movie_id = cd.movie_id
LEFT JOIN
    CompanyInfo ci ON rm.movie_id = ci.movie_id
WHERE
    rm.rank_within_year <= 5

UNION ALL

SELECT
    title.title,
    title.production_year,
    'Cameo Appearance' AS actor_name,
    1 AS role_count,
    0 AS null_notes_count,
    'Unknown Company' AS company_name,
    'Cameo' AS company_type,
    'Not Specified' AS company_note
FROM
    title
WHERE
    title.kind_id IN (SELECT id FROM kind_type WHERE kind LIKE '%Cameo%')
    AND title.production_year < 2000
    AND title.title IS NOT NULL

ORDER BY
    production_year, 
    role_count DESC;
