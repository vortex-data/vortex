
WITH MovieDetails AS (
    SELECT 
        t.id AS movie_id, 
        t.title, 
        t.production_year, 
        a.name AS actor_name, 
        ct.kind AS company_type, 
        k.keyword
    FROM 
        aka_title t
    JOIN 
        cast_info ci ON t.id = ci.movie_id
    JOIN 
        aka_name a ON ci.person_id = a.person_id
    LEFT JOIN 
        movie_companies mc ON t.id = mc.movie_id
    LEFT JOIN 
        company_name cn ON mc.company_id = cn.id
    LEFT JOIN 
        company_type ct ON mc.company_type_id = ct.id
    LEFT JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id
    WHERE 
        t.production_year >= 2000
),
AggregatedData AS (
    SELECT 
        movie_id, 
        title, 
        production_year, 
        STRING_AGG(DISTINCT actor_name, ', ') AS actors, 
        STRING_AGG(DISTINCT company_type, ', ') AS companies, 
        STRING_AGG(DISTINCT keyword, ', ') AS keywords
    FROM 
        MovieDetails
    GROUP BY 
        movie_id, title, production_year
)
SELECT 
    movie_id, 
    title, 
    production_year, 
    actors, 
    companies, 
    keywords
FROM 
    AggregatedData
ORDER BY 
    production_year DESC, title ASC
LIMIT 100;
