WITH RankedTitles AS (
    SELECT 
        t.id AS title_id,
        t.title,
        t.production_year,
        a.name AS actor_name,
        ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY a.name) AS actor_rank
    FROM 
        aka_title t
    JOIN 
        complete_cast cc ON t.id = cc.movie_id
    JOIN 
        cast_info ci ON cc.subject_id = ci.person_id
    JOIN 
        aka_name a ON ci.person_id = a.person_id
    WHERE 
        t.production_year >= 2000
),

RecentCompanyDetails AS (
    SELECT 
        mc.movie_id,
        c.name AS company_name,
        ct.kind AS company_type,
        ROW_NUMBER() OVER (PARTITION BY mc.movie_id ORDER BY c.name) AS company_rank
    FROM 
        movie_companies mc
    JOIN 
        company_name c ON mc.company_id = c.id
    JOIN 
        company_type ct ON mc.company_type_id = ct.id
    WHERE 
        c.country_code = 'USA'
),

KeywordCount AS (
    SELECT 
        mk.movie_id,
        COUNT(k.keyword) AS keyword_count
    FROM 
        movie_keyword mk
    JOIN 
        keyword k ON mk.keyword_id = k.id
    GROUP BY 
        mk.movie_id
)

SELECT 
    rt.title,
    rt.production_year,
    rt.actor_name,
    rc.company_name,
    rc.company_type,
    kc.keyword_count
FROM 
    RankedTitles rt
LEFT JOIN 
    RecentCompanyDetails rc ON rt.title_id = rc.movie_id
LEFT JOIN 
    KeywordCount kc ON rt.title_id = kc.movie_id
WHERE 
    rt.actor_rank <= 3
ORDER BY 
    rt.production_year DESC, rt.title;
