
WITH RankedMovies AS (
    SELECT 
        a.title,
        a.production_year,
        k.keyword,
        ROW_NUMBER() OVER (PARTITION BY a.id ORDER BY a.production_year DESC) AS rn,
        a.id
    FROM 
        aka_title a
    JOIN 
        movie_keyword mk ON a.id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
    WHERE 
        a.production_year >= 2000
), CompanyInfo AS (
    SELECT 
        mc.movie_id,
        c.name AS company_name,
        ct.kind AS company_type
    FROM 
        movie_companies mc
    JOIN 
        company_name c ON mc.company_id = c.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
), CompleteCastWithRoles AS (
    SELECT 
        c.movie_id,
        an.name AS actor_name,
        r.role AS role_name
    FROM 
        cast_info c
    JOIN 
        aka_name an ON c.person_id = an.person_id
    JOIN 
        role_type r ON c.role_id = r.id
)
SELECT 
    rm.title,
    rm.production_year,
    rm.keyword,
    ci.company_name,
    ci.company_type,
    cc.actor_name,
    cc.role_name
FROM 
    RankedMovies rm
JOIN 
    CompanyInfo ci ON rm.id = ci.movie_id
JOIN 
    CompleteCastWithRoles cc ON rm.id = cc.movie_id
WHERE 
    rm.rn = 1
ORDER BY 
    rm.production_year DESC, ci.company_name, cc.actor_name;
