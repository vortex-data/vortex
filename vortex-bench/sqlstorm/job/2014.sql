WITH RankedTitles AS (
    SELECT 
        a.title,
        a.production_year,
        ROW_NUMBER() OVER (PARTITION BY a.production_year ORDER BY a.production_year DESC) AS rank
    FROM 
        aka_title AS a
    WHERE 
        a.production_year IS NOT NULL
),
ActorMovieCounts AS (
    SELECT 
        ci.person_id,
        COUNT(DISTINCT ci.movie_id) AS movie_count
    FROM 
        cast_info AS ci
    GROUP BY 
        ci.person_id
),
RecentMovies AS (
    SELECT 
        m.id AS movie_id,
        m.title,
        m.production_year,
        COALESCE(k.keyword, 'No Keyword') AS keyword,
        CASE 
            WHEN m.production_year >= 2000 THEN 'Modern'
            ELSE 'Classic'
        END AS era
    FROM 
        aka_title AS m
    LEFT JOIN 
        movie_keyword AS mk ON m.id = mk.movie_id
    LEFT JOIN 
        keyword AS k ON mk.keyword_id = k.id
    WHERE 
        m.production_year IS NOT NULL AND
        m.production_year > (SELECT AVG(production_year) FROM aka_title)
),
SelectedActors AS (
    SELECT 
        p.id AS person_id,
        p.name,
        a.movie_count
    FROM 
        aka_name AS p
    JOIN 
        ActorMovieCounts AS a ON p.person_id = a.person_id
    WHERE 
        a.movie_count > 5
)
SELECT 
    rm.title,
    rm.production_year,
    rm.keyword,
    sa.name AS actor_name,
    sa.movie_count,
    rt.rank
FROM 
    RecentMovies AS rm
JOIN 
    complete_cast AS cc ON rm.movie_id = cc.movie_id
JOIN 
    SelectedActors AS sa ON cc.subject_id = sa.person_id
JOIN 
    RankedTitles AS rt ON rm.title = rt.title
WHERE 
    rm.era = 'Modern' AND 
    sa.name IS NOT NULL
ORDER BY 
    rm.production_year DESC, 
    sa.name;
