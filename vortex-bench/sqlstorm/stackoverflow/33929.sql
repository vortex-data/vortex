WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        p.Score,
        p.ViewCount,
        p.OwnerUserId,
        ROW_NUMBER() OVER (PARTITION BY p.OwnerUserId ORDER BY p.CreationDate DESC) AS PostRank
    FROM 
        Posts p
    WHERE 
        p.PostTypeId = 1  
),
UserReputation AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        u.Reputation,
        COUNT(DISTINCT p.Id) AS QuestionsCount,
        SUM(COALESCE(v.BountyAmount, 0)) AS TotalBounty
    FROM 
        Users u
    LEFT JOIN 
        Posts p ON u.Id = p.OwnerUserId AND p.PostTypeId = 1  
    LEFT JOIN 
        Votes v ON p.Id = v.PostId AND v.VoteTypeId IN (8, 9)  
    WHERE 
        u.Reputation > 0
    GROUP BY 
        u.Id, u.DisplayName, u.Reputation
),
TopUsers AS (
    SELECT 
        ur.UserId,
        ur.DisplayName,
        ur.Reputation,
        ur.QuestionsCount,
        ur.TotalBounty,
        DENSE_RANK() OVER (ORDER BY ur.Reputation DESC) AS ReputationRank
    FROM 
        UserReputation ur
    WHERE 
        ur.QuestionsCount > 5
),
OpenQuestions AS (
    SELECT 
        p.Id,
        p.Title,
        p.CreationDate,
        c.UserDisplayName AS LastCommenter,
        ROW_NUMBER() OVER (PARTITION BY p.Id ORDER BY c.CreationDate DESC) AS LatestCommentRank
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    WHERE 
        p.PostTypeId = 1 AND p.ClosedDate IS NULL  
),
FinalResults AS (
    SELECT 
        tp.UserId,
        tp.DisplayName,
        tp.Reputation,
        tp.QuestionsCount,
        tp.TotalBounty,
        rq.PostId,
        rq.Title,
        rq.CreationDate,
        rq.Score,
        rq.ViewCount,
        oq.LastCommenter
    FROM 
        TopUsers tp
    JOIN 
        RankedPosts rq ON tp.UserId = rq.OwnerUserId
    LEFT JOIN 
        OpenQuestions oq ON rq.PostId = oq.Id
    WHERE 
        tp.ReputationRank <= 10  
)

SELECT 
    fr.UserId,
    fr.DisplayName,
    fr.Reputation,
    fr.QuestionsCount,
    fr.TotalBounty,
    fr.PostId,
    fr.Title AS PostTitle,
    fr.CreationDate,
    fr.Score,
    fr.ViewCount,
    COALESCE(fr.LastCommenter, 'No comments yet') AS LastCommenter
FROM 
    FinalResults fr
ORDER BY 
    fr.Reputation DESC,
    fr.CreationDate DESC;