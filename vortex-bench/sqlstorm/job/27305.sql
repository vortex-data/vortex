
WITH RankedMovies AS (
    SELECT 
        t.title,
        t.production_year,
        t.kind_id,
        STRING_AGG(DISTINCT ak.name, ', ') AS aka_names,
        STRING_AGG(DISTINCT k.keyword, ', ') AS keywords,
        ROW_NUMBER() OVER (PARTITION BY t.kind_id ORDER BY t.production_year DESC) AS rank
    FROM 
        aka_title AS t
    LEFT JOIN 
        movie_keyword AS mk ON t.id = mk.movie_id
    LEFT JOIN 
        keyword AS k ON mk.keyword_id = k.id
    LEFT JOIN 
        movie_companies AS mc ON t.id = mc.movie_id
    LEFT JOIN 
        company_name AS cn ON mc.company_id = cn.id
    LEFT JOIN 
        aka_name AS ak ON ak.person_id = mc.company_id
    WHERE 
        t.production_year >= 2000
    GROUP BY 
        t.title, t.production_year, t.kind_id
),
TopRankedMovies AS (
    SELECT 
        title,
        production_year,
        aka_names,
        keywords,
        kind_id
    FROM 
        RankedMovies
    WHERE 
        rank <= 5
)
SELECT 
    tr.title,
    tr.production_year,
    ct.kind AS company_type,
    tr.aka_names,
    tr.keywords
FROM 
    TopRankedMovies AS tr
JOIN 
    company_type AS ct ON tr.kind_id = ct.id
ORDER BY 
    tr.production_year DESC, 
    tr.title ASC;
