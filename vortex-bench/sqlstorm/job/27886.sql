WITH RankedMovies AS (
    SELECT 
        a.title AS movie_title,
        a.production_year,
        c.name AS company_name,
        COUNT(DISTINCT ci.person_id) AS cast_count,
        ROW_NUMBER() OVER (PARTITION BY a.production_year ORDER BY COUNT(DISTINCT ci.person_id) DESC) AS rnk
    FROM 
        aka_title a
    JOIN 
        movie_companies mc ON a.id = mc.movie_id
    JOIN 
        company_name c ON mc.company_id = c.id
    JOIN 
        complete_cast cc ON a.id = cc.movie_id
    JOIN 
        cast_info ci ON cc.subject_id = ci.id
    GROUP BY 
        a.title, a.production_year, c.name
),
TopMovies AS (
    SELECT 
        movie_title,
        production_year,
        company_name,
        cast_count
    FROM 
        RankedMovies
    WHERE 
        rnk <= 5
)

SELECT 
    tm.movie_title,
    tm.production_year,
    tm.company_name,
    tm.cast_count,
    mi.info AS movie_info
FROM 
    TopMovies tm
LEFT JOIN 
    movie_info mi ON tm.movie_title = mi.info
WHERE 
    tm.production_year >= 2000
ORDER BY 
    tm.production_year DESC, 
    tm.cast_count DESC;
