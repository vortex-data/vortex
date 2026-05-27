
WITH RankedMovies AS (
    SELECT 
        a.id AS movie_id,
        a.title,
        a.production_year,
        a.kind_id,
        k.keyword AS movie_keyword,
        ROW_NUMBER() OVER (PARTITION BY a.production_year ORDER BY a.title) AS rn
    FROM 
        aka_title a
    JOIN 
        movie_keyword mk ON a.id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
    WHERE 
        a.production_year >= 2000
),
CastDetails AS (
    SELECT 
        ci.movie_id,
        COALESCE(aka.name, cn.name) AS actor_name,
        r.role,
        COUNT(ci.id) AS num_roles
    FROM 
        cast_info ci
    JOIN 
        role_type r ON ci.role_id = r.id
    LEFT JOIN 
        aka_name aka ON ci.person_id = aka.person_id
    LEFT JOIN 
        company_name cn ON ci.person_id = cn.imdb_id
    WHERE 
        ci.note IS NULL
    GROUP BY 
        ci.movie_id, actor_name, r.role
),
MovieInfo AS (
    SELECT 
        mv.movie_id,
        STRING_AGG(DISTINCT CONCAT(mv.info_type_id, ' - ', mv.info), '; ') AS movie_details
    FROM 
        movie_info mv
    GROUP BY 
        mv.movie_id
)

SELECT 
    rm.movie_id,
    rm.title,
    rm.production_year,
    rm.movie_keyword,
    cd.actor_name,
    cd.role,
    cd.num_roles,
    mi.movie_details
FROM 
    RankedMovies rm
LEFT JOIN 
    CastDetails cd ON rm.movie_id = cd.movie_id
LEFT JOIN 
    MovieInfo mi ON rm.movie_id = mi.movie_id
WHERE 
    rm.rn <= 5
ORDER BY 
    rm.production_year DESC, rm.title;
