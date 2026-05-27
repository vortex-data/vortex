WITH ranked_titles AS (
    SELECT 
        a.id AS aka_id,
        a.name AS aka_name,
        t.id AS title_id,
        t.title AS movie_title,
        t.production_year,
        ROW_NUMBER() OVER (PARTITION BY a.person_id ORDER BY t.production_year DESC) AS rn
    FROM 
        aka_name a
    JOIN 
        cast_info ci ON a.person_id = ci.person_id
    JOIN 
        aka_title t ON ci.movie_id = t.movie_id
    WHERE 
        a.name ILIKE '%John%'
        AND t.production_year >= 2000
),
top_titles AS (
    SELECT 
        aka_id,
        aka_name,
        title_id,
        movie_title,
        production_year
    FROM 
        ranked_titles
    WHERE 
        rn <= 5
),
movie_details AS (
    SELECT 
        tt.aka_id,
        tt.aka_name,
        tt.movie_title,
        tt.production_year,
        array_agg(DISTINCT k.keyword) AS keywords,
        array_agg(DISTINCT c.kind) AS company_types
    FROM 
        top_titles tt
    LEFT JOIN 
        movie_keyword mk ON tt.title_id = mk.movie_id
    LEFT JOIN 
        keyword k ON mk.keyword_id = k.id
    LEFT JOIN 
        movie_companies mc ON tt.title_id = mc.movie_id
    LEFT JOIN 
        company_type c ON mc.company_type_id = c.id
    GROUP BY 
        tt.aka_id, tt.aka_name, tt.movie_title, tt.production_year
)
SELECT 
    md.aka_name,
    md.movie_title,
    md.production_year,
    md.keywords,
    md.company_types
FROM 
    movie_details md
ORDER BY 
    md.production_year DESC, md.aka_name;