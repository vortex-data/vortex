
WITH TagCount AS (
    SELECT 
        TRIM(tag) AS TagName,
        COUNT(*) AS PostCount
    FROM (
        SELECT 
            UNNEST(string_to_array(SUBSTRING(Tags FROM 2 FOR LENGTH(Tags) - 2), '><')) AS tag
        FROM 
            Posts
        WHERE 
            PostTypeId = 1  
    ) AS extracted_tags
    GROUP BY 
        TRIM(tag)
),
TopTags AS (
    SELECT 
        TagName,
        PostCount,
        ROW_NUMBER() OVER (ORDER BY PostCount DESC) AS Rank
    FROM 
        TagCount
    WHERE 
        PostCount > 1  
),
MostActiveUsers AS (
    SELECT 
        Users.DisplayName,
        Users.Reputation,
        COUNT(Posts.Id) AS QuestionsAnswered,
        SUM(COALESCE(Posts.AnswerCount, 0)) AS TotalAnswers,
        SUM(COALESCE(Posts.Score, 0)) AS TotalScore
    FROM 
        Users
    JOIN 
        Posts ON Users.Id = Posts.OwnerUserId
    WHERE 
        Posts.PostTypeId = 2  
    GROUP BY 
        Users.DisplayName, Users.Reputation
),
TagUsage AS (
    SELECT 
        Posts.Id AS PostId,
        Posts.Title,
        Posts.CreationDate,
        UNNEST(string_to_array(SUBSTRING(Posts.Tags FROM 2 FOR LENGTH(Posts.Tags) - 2), '><')) AS TagName,
        Users.DisplayName AS Owner
    FROM 
        Posts
    JOIN 
        Users ON Posts.OwnerUserId = Users.Id
    WHERE 
        Posts.PostTypeId = 1  
)
SELECT 
    TopTags.TagName,
    TopTags.PostCount,
    MostActiveUsers.DisplayName,
    MostActiveUsers.Reputation,
    MostActiveUsers.QuestionsAnswered,
    MostActiveUsers.TotalAnswers,
    MostActiveUsers.TotalScore,
    COUNT(TagUsage.PostId) AS TagPostCount,
    MIN(TagUsage.CreationDate) AS EarliestPostDate,
    MAX(TagUsage.CreationDate) AS LatestPostDate
FROM 
    TopTags
JOIN 
    TagUsage ON TopTags.TagName = TagUsage.TagName
JOIN 
    MostActiveUsers ON TagUsage.Owner = MostActiveUsers.DisplayName
GROUP BY 
    TopTags.TagName, 
    TopTags.PostCount,
    MostActiveUsers.DisplayName, 
    MostActiveUsers.Reputation, 
    MostActiveUsers.QuestionsAnswered, 
    MostActiveUsers.TotalAnswers,
    MostActiveUsers.TotalScore
ORDER BY 
    TopTags.PostCount DESC, MostActiveUsers.TotalScore DESC
LIMIT 10;
