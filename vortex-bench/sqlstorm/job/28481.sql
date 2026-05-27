WITH ActorRoleInfo AS (
    SELECT 
        a.id AS actor_id,
        a.name AS actor_name,
        c.movie_id,
        t.title,
        t.production_year,
        r.role AS actor_role,
        k.keyword AS movie_keyword
    FROM 
        aka_name a
    JOIN 
        cast_info c ON a.person_id = c.person_id
    JOIN 
        title t ON c.movie_id = t.id
    JOIN 
        role_type r ON c.role_id = r.id
    JOIN 
        movie_keyword mk ON c.movie_id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
    WHERE 
        a.name ILIKE '%Smith%' 
),
ActorProductionCount AS (
    SELECT 
        actor_id,
        actor_name,
        COUNT(DISTINCT movie_id) AS total_movies
    FROM 
        ActorRoleInfo
    GROUP BY 
        actor_id, actor_name
),
TopActors AS (
    SELECT 
        actor_id,
        actor_name,
        total_movies
    FROM 
        ActorProductionCount
    ORDER BY 
        total_movies DESC
    LIMIT 10
)
SELECT 
    ta.actor_name,
    ta.total_movies,
    ARRAY_AGG(DISTINCT ari.title) AS movie_titles,
    ARRAY_AGG(DISTINCT ari.movie_keyword) AS keywords
FROM 
    TopActors ta
JOIN 
    ActorRoleInfo ari ON ta.actor_id = ari.actor_id
GROUP BY 
    ta.actor_id, ta.actor_name, ta.total_movies
ORDER BY 
    ta.total_movies DESC;