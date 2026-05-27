WITH MovieDetails AS (
    SELECT 
        t.title AS movie_title,
        t.production_year,
        r.role AS cast_role,
        an.name AS actor_name,
        c.name AS company_name,
        k.keyword AS movie_keyword
    FROM 
        title t
    JOIN 
        complete_cast cc ON t.id = cc.movie_id
    JOIN 
        cast_info ci ON cc.subject_id = ci.id
    JOIN 
        aka_name an ON ci.person_id = an.person_id
    JOIN 
        movie_companies mc ON t.id = mc.movie_id
    JOIN 
        company_name c ON mc.company_id = c.id
    LEFT JOIN 
        movie_keyword mk ON t.id = mk.movie_id
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id
    JOIN 
        role_type r ON ci.role_id = r.id
    WHERE 
        t.production_year >= 2000
        AND c.country_code = 'USA'
),
AggregatedResults AS (
    SELECT 
        movie_title,
        production_year,
        ARRAY_AGG(DISTINCT actor_name) AS actors,
        ARRAY_AGG(DISTINCT company_name) AS production_companies,
        ARRAY_AGG(DISTINCT movie_keyword) AS keywords,
        COUNT(DISTINCT cast_role) AS role_count
    FROM 
        MovieDetails
    GROUP BY 
        movie_title, production_year
)
SELECT 
    movie_title,
    production_year,
    actors,
    production_companies,
    keywords,
    role_count
FROM 
    AggregatedResults
ORDER BY 
    production_year DESC, 
    movie_title;
