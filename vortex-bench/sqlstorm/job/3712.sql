WITH MovieDetails AS (
    SELECT 
        a.title, 
        a.production_year,
        COUNT(DISTINCT c.person_id) AS cast_count,
        STRING_AGG(DISTINCT k.keyword, ', ') AS keywords
    FROM 
        aka_title a 
    LEFT JOIN 
        complete_cast cc ON a.id = cc.movie_id 
    LEFT JOIN 
        cast_info c ON cc.subject_id = c.id 
    LEFT JOIN 
        movie_keyword mk ON a.id = mk.movie_id 
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id 
    GROUP BY 
        a.title, a.production_year
), 
TopMovies AS (
    SELECT 
        title, 
        production_year, 
        cast_count, 
        keywords,
        RANK() OVER (ORDER BY cast_count DESC) AS rank
    FROM 
        MovieDetails
)
SELECT 
    tm.title,
    tm.production_year,
    tm.cast_count,
    COALESCE(tm.keywords, 'No Keywords') AS keywords,
    (SELECT AVG(cast_count) FROM TopMovies) AS avg_cast_count,
    CASE 
        WHEN tm.cast_count > (SELECT AVG(cast_count) FROM TopMovies) 
        THEN 'Above Average' 
        ELSE 'Below Average' 
    END AS performance_category
FROM 
    TopMovies tm
WHERE 
    tm.rank <= 10
ORDER BY 
    tm.rank;
