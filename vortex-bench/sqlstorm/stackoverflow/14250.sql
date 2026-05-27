
WITH UserPostStats AS (
    SELECT 
        u.Id AS UserId,
        COUNT(p.Id) AS PostCount,
        SUM(COALESCE(p.Score, 0)) AS TotalScore,
        AVG(COALESCE(p.ViewCount, 0)) AS AvgViewCount,
        AVG(COALESCE(p.AcceptedAnswerId, 0)) AS AcceptedAnswerRatio
    FROM 
        Users u
    LEFT JOIN 
        Posts p ON u.Id = p.OwnerUserId
    GROUP BY 
        u.Id
),
PostHistoryStats AS (
    SELECT 
        ph.PostId,
        COUNT(ph.Id) AS EditCount,
        MAX(ph.CreationDate) AS LastEdited
    FROM 
        PostHistory ph
    GROUP BY 
        ph.PostId
),
FinalStats AS (
    SELECT 
        up.UserId,
        up.PostCount,
        up.TotalScore,
        up.AvgViewCount,
        ph.EditCount,
        ph.LastEdited
    FROM 
        UserPostStats up
    LEFT JOIN 
        PostHistoryStats ph ON up.UserId = ph.PostId
)
SELECT 
    UserId,
    PostCount,
    TotalScore,
    AvgViewCount,
    EditCount,
    LastEdited
FROM 
    FinalStats
ORDER BY 
    TotalScore DESC, PostCount DESC;
