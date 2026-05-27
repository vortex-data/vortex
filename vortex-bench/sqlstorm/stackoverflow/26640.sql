
WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title AS PostTitle,
        p.Tags,
        p.CreationDate,
        p.AcceptedAnswerId,
        COUNT(a.Id) AS AnswerCount,
        ROW_NUMBER() OVER(PARTITION BY p.Id ORDER BY p.CreationDate DESC) AS rn
    FROM 
        Posts p
    LEFT JOIN 
        Posts a ON p.Id = a.ParentId
    WHERE 
        p.PostTypeId = 1 
    GROUP BY 
        p.Id, p.Title, p.Tags, p.CreationDate, p.AcceptedAnswerId
),
MostVotedAnswers AS (
    SELECT 
        a.Id AS AnswerId,
        a.ParentId,
        a.Score,
        u.DisplayName AS OwnerName
    FROM 
        Posts a
    JOIN 
        Users u ON a.OwnerUserId = u.Id
    WHERE 
        a.PostTypeId = 2 
    ORDER BY 
        a.Score DESC
),
ClosedPosts AS (
    SELECT 
        ph.PostId,
        ph.CreationDate AS CloseDate,
        c.Name AS CloseReason
    FROM 
        PostHistory ph
    JOIN 
        CloseReasonTypes c ON ph.Comment = CAST(c.Id AS VARCHAR)
    WHERE 
        ph.PostHistoryTypeId = 10 
)
SELECT 
    rp.PostId,
    rp.PostTitle,
    rp.Tags,
    rp.CreationDate AS QuestionDate,
    mp.AnswerId,
    mp.OwnerName AS AnswerOwner,
    mp.Score AS AnswerScore,
    cp.CloseDate,
    cp.CloseReason
FROM 
    RankedPosts rp
LEFT JOIN 
    MostVotedAnswers mp ON rp.PostId = mp.ParentId
LEFT JOIN 
    ClosedPosts cp ON rp.PostId = cp.PostId
WHERE 
    rp.rn = 1 
ORDER BY 
    AnswerScore DESC NULLS LAST,
    QuestionDate DESC;
