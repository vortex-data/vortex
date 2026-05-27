WITH UserReputation AS (
    SELECT 
        U.Id AS UserId,
        U.DisplayName,
        U.Reputation,
        U.CreationDate,
        ROW_NUMBER() OVER (ORDER BY U.Reputation DESC) AS ReputationRank
    FROM 
        Users U
),
PostStatistics AS (
    SELECT 
        P.OwnerUserId,
        COUNT(P.Id) AS TotalPosts,
        COUNT(P.AcceptedAnswerId) AS AcceptedAnswers,
        SUM(P.Score) AS TotalScore,
        AVG(P.ViewCount) AS AvgViews
    FROM 
        Posts P
    GROUP BY 
        P.OwnerUserId
),
TopContributors AS (
    SELECT 
        UR.UserId,
        UR.DisplayName,
        PS.TotalPosts,
        PS.AcceptedAnswers,
        PS.TotalScore,
        PS.AvgViews,
        RANK() OVER (ORDER BY PS.TotalScore DESC) AS ScoreRank
    FROM 
        UserReputation UR
    JOIN 
        PostStatistics PS ON UR.UserId = PS.OwnerUserId
    WHERE 
        UR.Reputation > 1000
)

SELECT 
    TC.DisplayName,
    TC.TotalPosts,
    TC.AcceptedAnswers,
    TC.TotalScore,
    TC.AvgViews,
    CASE 
        WHEN TC.ScoreRank <= 10 THEN 'Top Contributor'
        ELSE 'Contributor'
    END AS ContributorType,
    COALESCE(B.BadgeCount, 0) AS BadgeCount
FROM 
    TopContributors TC
LEFT JOIN (
    SELECT 
        UserId,
        COUNT(*) AS BadgeCount
    FROM 
        Badges
    GROUP BY 
        UserId
) B ON TC.UserId = B.UserId
WHERE 
    TC.AvgViews > 50
ORDER BY 
    TC.TotalScore DESC, 
    BadgeCount DESC;
