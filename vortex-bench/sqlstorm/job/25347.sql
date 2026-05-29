WITH RankedMovies AS (
    SELECT 
        a.title AS movie_title,
        a.production_year,
        p.name AS person_name,
        rk.rnk,
        ROW_NUMBER() OVER (PARTITION BY a.id ORDER BY cc.nr_order) AS cast_order
    FROM 
        aka_title a
    JOIN 
        cast_info cc ON a.id = cc.movie_id
    JOIN 
        aka_name p ON cc.person_id = p.person_id
    JOIN 
        role_type rt ON cc.role_id = rt.id
    JOIN 
        movie_info mi ON a.id = mi.movie_id
    JOIN 
        info_type it ON mi.info_type_id = it.id
    LEFT JOIN 
        (SELECT 
            mi.movie_id,
            COUNT(*) AS rnk
        FROM 
            movie_info mi
        JOIN 
            info_type it ON mi.info_type_id = it.id 
        WHERE 
            it.info LIKE '%Award%'
        GROUP BY 
            mi.movie_id) rk ON a.id = rk.movie_id
    WHERE 
        a.production_year >= 2000
        AND a.kind_id IN (SELECT id FROM kind_type WHERE kind IN ('feature', 'short'))
),

FinalOutput AS (
    SELECT 
        rm.movie_title,
        rm.production_year,
        rm.person_name,
        rm.cast_order,
        COALESCE(rm.rnk, 0) AS rank_award_count
    FROM 
        RankedMovies rm
)

SELECT 
    movie_title,
    production_year,
    person_name,
    cast_order,
    rank_award_count
FROM 
    FinalOutput
ORDER BY 
    production_year DESC, 
    rank_award_count DESC, 
    cast_order ASC
LIMIT 100;
