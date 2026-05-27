WITH movie_details AS (
    SELECT 
        mt.title AS movie_title,
        mt.production_year,
        ak.name AS actor_name,
        ak.id AS actor_id,
        ct.kind AS company_type,
        cn.name AS company_name,
        mi.info AS movie_info,
        ko.keyword AS movie_keyword
    FROM 
        aka_name ak
    JOIN 
        cast_info ci ON ak.person_id = ci.person_id
    JOIN 
        title mt ON ci.movie_id = mt.id
    JOIN 
        movie_companies mc ON mt.id = mc.movie_id
    JOIN 
        company_name cn ON mc.company_id = cn.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
    LEFT JOIN 
        movie_info mi ON mt.id = mi.movie_id
    LEFT JOIN 
        movie_keyword mk ON mt.id = mk.movie_id
    LEFT JOIN 
        keyword ko ON mk.keyword_id = ko.id
    WHERE 
        mt.production_year >= 2000
        AND ak.name IS NOT NULL
        AND cn.country_code = 'USA'
),
aggregated_details AS (
    SELECT 
        movie_title,
        production_year,
        actor_name,
        COUNT(DISTINCT actor_id) AS actor_count,
        STRING_AGG(DISTINCT company_name, ', ') AS production_companies,
        STRING_AGG(DISTINCT movie_info, ', ') AS additional_info,
        STRING_AGG(DISTINCT movie_keyword, ', ') AS keywords
    FROM 
        movie_details
    GROUP BY 
        movie_title, production_year, actor_name
)
SELECT 
    actor_name,
    movie_title,
    production_year,
    actor_count,
    production_companies,
    additional_info,
    keywords
FROM 
    aggregated_details
ORDER BY 
    production_year DESC, actor_count DESC;