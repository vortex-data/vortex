
WITH RankedMovies AS (
    SELECT 
        m.id AS movie_id,
        m.title AS movie_title,
        m.production_year,
        COUNT(c.id) AS cast_count
    FROM 
        aka_title m
    JOIN 
        cast_info c ON m.id = c.movie_id
    WHERE 
        m.production_year >= 2000
    GROUP BY 
        m.id, m.title, m.production_year
),
TopMovies AS (
    SELECT 
        movie_id,
        movie_title,
        production_year,
        cast_count,
        ROW_NUMBER() OVER (ORDER BY cast_count DESC) AS rank
    FROM 
        RankedMovies
    WHERE 
        cast_count > 5
)
SELECT 
    tm.movie_title,
    tm.production_year,
    a.name AS actor_name,
    a.name_pcode_nf,
    a.name_pcode_cf,
    r.role,
    STRING_AGG(DISTINCT k.keyword, ',' ORDER BY k.keyword) AS keywords
FROM 
    TopMovies tm
JOIN 
    cast_info ci ON tm.movie_id = ci.movie_id
JOIN 
    aka_name a ON ci.person_id = a.person_id
JOIN 
    role_type r ON ci.role_id = r.id
JOIN 
    movie_keyword mk ON tm.movie_id = mk.movie_id
JOIN 
    keyword k ON mk.keyword_id = k.id
WHERE 
    tm.rank <= 10
GROUP BY 
    tm.movie_title, tm.production_year, a.name, a.name_pcode_nf, a.name_pcode_cf, r.role, tm.rank
ORDER BY 
    tm.rank, tm.movie_title;
