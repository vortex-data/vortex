
WITH RankedTitles AS (
    SELECT 
        t.id AS title_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS title_rank
    FROM 
        title t
    WHERE 
        t.production_year IS NOT NULL
),
ActorRoles AS (
    SELECT 
        a.id AS aka_id,
        c.movie_id,
        c.role_id,
        r.role,
        COUNT(*) AS role_count
    FROM 
        cast_info c
    JOIN 
        aka_name a ON c.person_id = a.person_id
    JOIN 
        role_type r ON c.role_id = r.id
    GROUP BY 
        a.id, c.movie_id, c.role_id, r.role
),
MovieCompanyDetails AS (
    SELECT 
        mc.movie_id,
        STRING_AGG(CASE WHEN ct.kind = 'Producer' THEN cn.name ELSE NULL END, ', ') AS producers,
        STRING_AGG(CASE WHEN ct.kind = 'Distributor' THEN cn.name ELSE NULL END, ', ') AS distributors
    FROM 
        movie_companies mc
    JOIN 
        company_name cn ON mc.company_id = cn.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
    GROUP BY 
        mc.movie_id
)
SELECT 
    rt.title AS movie_title,
    rt.production_year,
    ar.role,
    ar.role_count,
    mcd.producers,
    mcd.distributors
FROM 
    RankedTitles rt
JOIN 
    ActorRoles ar ON rt.title_id = ar.movie_id
JOIN 
    MovieCompanyDetails mcd ON rt.title_id = mcd.movie_id
WHERE 
    rt.title_rank <= 10 
ORDER BY 
    rt.production_year DESC, 
    ar.role_count DESC;
