WITH UserStats AS (
    SELECT 
        U.Id AS UserId,
        U.Reputation,
        U.Views,
        COUNT(DISTINCT P.Id) AS PostCount,
        SUM(CASE WHEN P.PostTypeId = 1 THEN 1 ELSE 0 END) AS QuestionCount,
        SUM(CASE WHEN P.PostTypeId = 2 THEN 1 ELSE 0 END) AS AnswerCount,
        SUM(CASE WHEN P.PostTypeId IN (1, 2) THEN P.Score ELSE 0 END) AS TotalScore
    FROM 
        Users U
    LEFT JOIN 
        Posts P ON U.Id = P.OwnerUserId
    GROUP BY 
        U.Id, U.Reputation, U.Views
),
TopBadgeUsers AS (
    SELECT 
        B.UserId,
        COUNT(B.Id) AS BadgeCount
    FROM 
        Badges B
    GROUP BY 
        B.UserId
)
SELECT 
    U.UserId,
    U.Reputation,
    U.Views,
    U.PostCount,
    U.QuestionCount,
    U.AnswerCount,
    U.TotalScore,
    COALESCE(B.BadgeCount, 0) AS BadgeCount
FROM 
    UserStats U
LEFT JOIN 
    TopBadgeUsers B ON U.UserId = B.UserId
ORDER BY 
    U.Reputation DESC, U.TotalScore DESC
LIMIT 100;