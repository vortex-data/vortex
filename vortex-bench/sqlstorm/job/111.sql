WITH RankedTitles AS (
    SELECT 
        t.id AS title_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS rank_per_year
    FROM 
        title t
    WHERE 
        t.production_year IS NOT NULL
),
ActorMovies AS (
    SELECT 
        ci.movie_id,
        ak.name AS actor_name,
        ROW_NUMBER() OVER (PARTITION BY ci.movie_id ORDER BY ak.name) AS actor_rank
    FROM 
        cast_info ci
    JOIN 
        aka_name ak ON ci.person_id = ak.person_id
    WHERE 
        ci.role_id IN (SELECT id FROM role_type WHERE role = 'actor')
),
MovieKeywords AS (
    SELECT 
        mk.movie_id,
        STRING_AGG(k.keyword, ', ') AS keyword_list
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
),
CompanyInfo AS (
    SELECT 
        mc.movie_id,
        COALESCE(cn.name, 'Unknown Company') AS company_name,
        ct.kind AS company_type
    FROM 
        movie_companies mc
    LEFT JOIN 
        company_name cn ON mc.company_id = cn.id
    LEFT JOIN 
        company_type ct ON mc.company_type_id = ct.id
)
SELECT 
    rt.title AS Movie_Title,
    rt.production_year AS Production_Year,
    am.actor_name AS Actor,
    mk.keyword_list AS Keywords,
    ci.company_name AS Production_Company,
    ci.company_type AS Company_Type,
    am.actor_rank,
    rt.rank_per_year
FROM 
    RankedTitles rt
LEFT JOIN 
    ActorMovies am ON rt.title_id = am.movie_id
LEFT JOIN 
    MovieKeywords mk ON rt.title_id = mk.movie_id
LEFT JOIN 
    CompanyInfo ci ON rt.title_id = ci.movie_id
WHERE 
    rt.rank_per_year <= 3
    AND rt.production_year > 2000
ORDER BY 
    rt.production_year DESC, 
    rt.title ASC, 
    am.actor_rank;
