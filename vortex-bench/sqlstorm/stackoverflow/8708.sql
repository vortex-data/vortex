
WITH UserPostStats AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        COUNT(p.Id) AS TotalPosts,
        SUM(CASE WHEN p.PostTypeId = 1 THEN 1 ELSE 0 END) AS TotalQuestions,
        SUM(CASE WHEN p.PostTypeId = 2 THEN 1 ELSE 0 END) AS TotalAnswers,
        SUM(CASE WHEN p.Score > 0 THEN 1 ELSE 0 END) AS PositivePosts,
        SUM(CASE WHEN p.Score < 0 THEN 1 ELSE 0 END) AS NegativePosts,
        SUM(p.ViewCount) AS TotalViews
    FROM 
        Users u
    LEFT JOIN 
        Posts p ON u.Id = p.OwnerUserId
    GROUP BY 
        u.Id, u.DisplayName
), 
TopUsers AS (
    SELECT 
        UserId,
        DisplayName,
        TotalPosts,
        TotalQuestions,
        TotalAnswers,
        PositivePosts,
        NegativePosts,
        TotalViews,
        RANK() OVER (ORDER BY TotalPosts DESC) AS RankByPosts,
        RANK() OVER (ORDER BY TotalViews DESC) AS RankByViews
    FROM 
        UserPostStats
)
SELECT 
    tu.DisplayName,
    tu.TotalPosts,
    tu.TotalQuestions,
    tu.TotalAnswers,
    tu.PositivePosts,
    tu.NegativePosts,
    tu.TotalViews,
    CASE 
        WHEN tu.RankByPosts <= 10 THEN 'Top Contributor'
        WHEN tu.RankByViews <= 10 THEN 'Popular User'
        ELSE 'Regular User' 
    END AS UserCategory
FROM 
    TopUsers tu
WHERE 
    tu.TotalPosts > 50 OR tu.TotalViews > 1000
ORDER BY 
    tu.TotalPosts DESC, 
    tu.TotalViews DESC;
