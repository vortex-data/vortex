SELECT a.name, t.title, c.note, ci.kind
FROM aka_name a
JOIN cast_info c ON a.person_id = c.person_id
JOIN aka_title t ON c.movie_id = t.movie_id
JOIN movie_companies mc ON t.id = mc.movie_id
JOIN company_name cn ON mc.company_id = cn.id
JOIN comp_cast_type ci ON c.person_role_id = ci.id
WHERE t.production_year > 2000
ORDER BY t.production_year DESC;
