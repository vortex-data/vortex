WITH RankedTitles AS (
    SELECT 
        a.title, 
        a.production_year, 
        k.keyword, 
        ROW_NUMBER() OVER (PARTITION BY a.production_year ORDER BY LENGTH(a.title) DESC) AS rank_title_length
    FROM 
        aka_title a
    JOIN 
        movie_keyword mk ON a.id = mk.movie_id
    JOIN 
        keyword k ON mk.keyword_id = k.id
    WHERE 
        a.production_year IS NOT NULL
), FilteredActors AS (
    SELECT 
        an.name AS actor_name, 
        a.title AS movie_title, 
        a.production_year,
        COUNT(*) AS role_count
    FROM 
        cast_info c
    JOIN 
        aka_name an ON c.person_id = an.person_id
    JOIN 
        aka_title a ON c.movie_id = a.id
    GROUP BY 
        an.name, a.title, a.production_year
), Summary AS (
    SELECT 
        year.production_year,
        COUNT(DISTINCT year.title) AS unique_titles,
        SUM(role_count) AS total_roles,
        STRING_AGG(DISTINCT actor_name, ', ') AS all_actors
    FROM 
        RankedTitles year
    JOIN 
        FilteredActors actors ON year.title = actors.movie_title AND year.production_year = actors.production_year
    GROUP BY 
        year.production_year
)
SELECT 
    s.production_year, 
    s.unique_titles, 
    s.total_roles, 
    s.all_actors,
    (SELECT COUNT(*) FROM aka_title WHERE production_year = s.production_year) AS total_movies_for_year
FROM 
    Summary s
ORDER BY 
    s.production_year DESC;
