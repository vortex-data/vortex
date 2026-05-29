WITH RankedMovies AS (
    SELECT 
        t.title,
        t.production_year,
        COUNT(DISTINCT mc.company_id) AS company_count,
        RANK() OVER (PARTITION BY t.production_year ORDER BY COUNT(DISTINCT mc.company_id) DESC) AS rank_in_year
    FROM 
        aka_title t
    LEFT JOIN 
        movie_companies mc ON t.id = mc.movie_id
    GROUP BY 
        t.id, t.title, t.production_year
),
TopMovies AS (
    SELECT 
        title, 
        production_year
    FROM 
        RankedMovies
    WHERE 
        rank_in_year <= 3
),
ActorInfo AS (
    SELECT 
        ak.name AS actor_name,
        t.title AS movie_title,
        t.production_year,
        ci.note AS role_note,
        ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY ci.nr_order) AS role_order
    FROM 
        cast_info ci
    JOIN 
        aka_name ak ON ci.person_id = ak.person_id
    JOIN 
        aka_title t ON ci.movie_id = t.id
    WHERE 
        ci.note IS NOT NULL
),
CombinedResults AS (
    SELECT 
        tm.production_year,
        tm.title,
        ai.actor_name,
        ai.role_note,
        CASE 
            WHEN ai.role_note IS NOT NULL THEN 'Role: ' || ai.role_note 
            ELSE 'Unknown Role' 
        END AS role_description
    FROM 
        TopMovies tm
    LEFT JOIN 
        ActorInfo ai ON tm.title = ai.movie_title AND tm.production_year = ai.production_year
)
SELECT 
    production_year,
    title,
    STRING_AGG(actor_name, ', ') AS actors,
    MAX(role_description) AS sample_role_description
FROM 
    CombinedResults
GROUP BY 
    production_year, title
ORDER BY 
    production_year DESC, title;
