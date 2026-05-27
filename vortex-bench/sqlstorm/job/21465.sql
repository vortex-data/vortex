WITH RankedMovies AS (
    SELECT 
        t.id AS movie_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER(PARTITION BY t.production_year ORDER BY t.title) AS title_rank
    FROM 
        aka_title t 
    WHERE 
        t.production_year IS NOT NULL
),
ActorRoles AS (
    SELECT 
        c.movie_id,
        c.person_id,
        c.role_id,
        r.role AS role_name,
        COUNT(*) OVER(PARTITION BY c.person_id ORDER BY c.nr_order ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS total_roles
    FROM 
        cast_info c
    JOIN 
        role_type r ON c.role_id = r.id
),
MovieWithKeywords AS (
    SELECT 
        m.movie_id,
        STRING_AGG(k.keyword, ', ') AS keywords
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    JOIN 
        aka_title m ON mk.movie_id = m.id
    GROUP BY 
        m.movie_id
),
CompanyDetails AS (
    SELECT 
        mc.movie_id,
        co.name AS company_name,
        ct.kind AS company_type
    FROM 
        movie_companies mc
    JOIN 
        company_name co ON mc.company_id = co.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
),
FinalResults AS (
    SELECT 
        rm.title,
        rm.production_year,
        ar.person_id,
        ar.role_name,
        mk.keywords,
        cd.company_name,
        cd.company_type,
        CASE 
            WHEN ar.total_roles = 0 THEN 'No roles'
            WHEN ar.total_roles IS NULL THEN 'Role data not available'
            ELSE 'Acted in ' || ar.total_roles || ' roles'
        END AS role_info
    FROM 
        RankedMovies rm
    LEFT JOIN 
        ActorRoles ar ON rm.movie_id = ar.movie_id
    LEFT JOIN 
        MovieWithKeywords mk ON rm.movie_id = mk.movie_id
    LEFT JOIN 
        CompanyDetails cd ON rm.movie_id = cd.movie_id
    WHERE 
        rm.title_rank = 1
)
SELECT 
    title,
    production_year,
    person_id,
    role_name,
    keywords,
    company_name,
    company_type,
    role_info
FROM 
    FinalResults
ORDER BY 
    production_year DESC, title;
