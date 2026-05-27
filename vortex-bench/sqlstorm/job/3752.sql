WITH RankedTitles AS (
    SELECT 
        t.id AS title_id,
        t.title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY t.production_year ORDER BY t.title) AS rank
    FROM 
        aka_title t
    WHERE 
        t.production_year IS NOT NULL
),
ActorMovies AS (
    SELECT 
        a.name AS actor_name,
        ti.title AS movie_title,
        ti.production_year,
        COUNT(ci.movie_id) AS total_roles
    FROM 
        aka_name a
    INNER JOIN 
        cast_info ci ON a.person_id = ci.person_id
    INNER JOIN 
        title ti ON ci.movie_id = ti.id
    GROUP BY 
        a.name, ti.title, ti.production_year
),
TotalMoviesPerActor AS (
    SELECT 
        actor_name,
        COUNT(DISTINCT movie_title) AS movie_count
    FROM 
        ActorMovies
    GROUP BY 
        actor_name
),
TopActors AS (
    SELECT 
        actor_name,
        movie_count,
        RANK() OVER (ORDER BY movie_count DESC) AS actor_rank
    FROM 
        TotalMoviesPerActor
)

SELECT 
    r.title_id,
    r.title,
    r.production_year,
    ta.actor_name,
    ta.movie_count,
    COALESCE(ta.actor_rank, 0) AS actor_rank,
    CASE 
        WHEN ta.movie_count > 5 THEN 'Veteran Actor'
        WHEN ta.movie_count BETWEEN 2 AND 5 THEN 'Emerging Actor'
        ELSE 'New Talent'
    END AS actor_category
FROM 
    RankedTitles r
LEFT JOIN 
    ActorMovies am ON r.title = am.movie_title AND r.production_year = am.production_year
LEFT JOIN 
    TopActors ta ON am.actor_name = ta.actor_name
WHERE 
    r.rank <= 10
ORDER BY 
    r.production_year DESC, 
    r.title;
