WITH RankedMovies AS (
    SELECT 
        mt.id AS movie_id,
        mt.title,
        mt.production_year,
        COUNT(DISTINCT mc.company_id) AS company_count,
        COUNT(DISTINCT mk.keyword_id) AS keyword_count,
        ROW_NUMBER() OVER (PARTITION BY mt.production_year ORDER BY COUNT(DISTINCT mc.company_id) DESC) AS rank
    FROM 
        aka_title mt
    LEFT JOIN 
        movie_companies mc ON mt.id = mc.movie_id
    LEFT JOIN 
        movie_keyword mk ON mt.id = mk.movie_id
    GROUP BY 
        mt.id, mt.title, mt.production_year
),
TopRankedMovies AS (
    SELECT 
        movie_id,
        title,
        production_year,
        company_count,
        keyword_count
    FROM 
        RankedMovies
    WHERE 
        rank <= 5
)
SELECT 
    tr.title,
    tr.production_year,
    ak.name AS actor_name,
    COUNT(DISTINCT c.person_role_id) AS roles_played
FROM 
    TopRankedMovies tr
JOIN 
    complete_cast cc ON tr.movie_id = cc.movie_id
JOIN 
    cast_info c ON cc.subject_id = c.id
JOIN 
    aka_name ak ON c.person_id = ak.person_id
WHERE 
    ak.name IS NOT NULL
GROUP BY 
    tr.title, tr.production_year, ak.name
ORDER BY 
    tr.production_year DESC, roles_played DESC;
