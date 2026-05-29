
WITH RankedMovies AS (
    SELECT
        t.id AS title_id,
        t.title,
        t.production_year,
        COUNT(DISTINCT c.person_id) AS actor_count,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY COUNT(DISTINCT c.person_id) DESC) AS rank
    FROM
        aka_title t
    JOIN
        cast_info c ON t.id = c.movie_id
    WHERE
        t.production_year IS NOT NULL
    GROUP BY
        t.id, t.title, t.production_year
),
TopMovies AS (
    SELECT
        title_id,
        title,
        production_year
    FROM
        RankedMovies
    WHERE
        rank <= 5
),
MovieDetails AS (
    SELECT
        tm.title_id,
        tm.title,
        tm.production_year,
        STRING_AGG(DISTINCT co.name, ', ') AS companies,
        STRING_AGG(DISTINCT k.keyword, ', ') AS keywords
    FROM
        TopMovies tm
    LEFT JOIN
        movie_companies mc ON tm.title_id = mc.movie_id
    LEFT JOIN
        company_name co ON mc.company_id = co.id
    LEFT JOIN
        movie_keyword mk ON tm.title_id = mk.movie_id
    LEFT JOIN
        keyword k ON mk.keyword_id = k.id
    GROUP BY
        tm.title_id, tm.title, tm.production_year
)
SELECT
    md.title,
    md.production_year,
    md.companies,
    md.keywords,
    COALESCE(NULLIF(md.companies, ''), 'No companies listed') AS company_info,
    CASE 
        WHEN md.production_year < 2000 THEN 'Classic'
        WHEN md.production_year BETWEEN 2000 AND 2010 THEN 'Modern'
        ELSE 'Recent'
    END AS era
FROM
    MovieDetails md
ORDER BY
    md.production_year DESC;
