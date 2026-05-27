
WITH MovieDetails AS (
    SELECT 
        t.title AS movie_title,
        t.production_year,
        STRING_AGG(DISTINCT a.name, ', ') AS actors,
        STRING_AGG(DISTINCT k.keyword, ', ') AS keywords,
        ct.kind AS company_type,
        STRING_AGG(DISTINCT cn.name, ', ') AS companies
    FROM 
        aka_title t
    JOIN 
        complete_cast cc ON t.id = cc.movie_id
    JOIN 
        cast_info ci ON cc.subject_id = ci.person_id
    JOIN 
        aka_name a ON ci.person_id = a.person_id
    LEFT JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id
    JOIN 
        movie_companies mc ON t.id = mc.movie_id
    JOIN 
        company_name cn ON mc.company_id = cn.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
    GROUP BY 
        t.id, t.title, t.production_year, ct.kind
),
PopularMovies AS (
    SELECT 
        movie_title,
        production_year,
        actors,
        keywords,
        company_type,
        companies,
        ROW_NUMBER() OVER (PARTITION BY production_year ORDER BY COUNT(DISTINCT actors) DESC) AS rank
    FROM 
        MovieDetails
    GROUP BY 
        movie_title, production_year, actors, keywords, company_type, companies
)
SELECT 
    movie_title,
    production_year,
    actors,
    keywords,
    company_type,
    companies
FROM 
    PopularMovies
WHERE 
    rank <= 5
ORDER BY 
    production_year DESC, rank;
