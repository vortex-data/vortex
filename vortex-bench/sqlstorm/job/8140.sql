
SELECT 
    a.name AS actor_name,
    t.title AS movie_title,
    mc.note AS company_note,
    STRING_AGG(DISTINCT k.keyword, ',') AS keywords,
    ci.nr_order AS cast_order,
    ii.info AS movie_info
FROM 
    aka_name a
JOIN 
    cast_info ci ON a.person_id = ci.person_id
JOIN 
    aka_title t ON ci.movie_id = t.movie_id
JOIN 
    movie_companies mc ON t.id = mc.movie_id
JOIN 
    company_name cn ON mc.company_id = cn.id
JOIN 
    movie_keyword mk ON t.id = mk.movie_id
JOIN 
    keyword k ON mk.keyword_id = k.id
JOIN 
    movie_info ii ON t.id = ii.movie_id
WHERE 
    t.production_year >= 2000 
    AND cn.country_code = 'USA'
    AND ii.info_type_id = (SELECT id FROM info_type WHERE info = 'Summary')
GROUP BY 
    a.name, t.title, mc.note, ci.nr_order, ii.info, t.production_year
ORDER BY 
    t.production_year DESC, a.name;
