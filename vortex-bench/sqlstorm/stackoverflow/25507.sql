WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.Body,
        U.DisplayName AS OwnerDisplayName,
        p.CreationDate,
        p.ViewCount,
        p.AnswerCount,
        p.Tags,
        RANK() OVER (PARTITION BY p.Tags ORDER BY p.CreationDate DESC) AS RankByTags
    FROM 
        Posts p
    JOIN 
        Users U ON p.OwnerUserId = U.Id
    WHERE 
        p.PostTypeId = 1 
        AND p.CreationDate >= '2023-01-01' 
        AND p.ViewCount > 100 
),

HistoryWithComments AS (
    SELECT 
        ph.PostId,
        ph.UserDisplayName,
        ph.CreationDate AS HistoryDate,
        ph.Comment AS EditComment,
        ph.Text AS EditText
    FROM 
        PostHistory ph
    JOIN 
        Posts p ON ph.PostId = p.Id
    WHERE 
        ph.PostHistoryTypeId IN (4, 5, 6) 
        AND ph.CreationDate >= '2023-01-01' 
)

SELECT 
    rp.PostId,
    rp.Title,
    rp.OwnerDisplayName,
    rp.CreationDate AS QuestionDate,
    rp.ViewCount,
    rp.AnswerCount,
    rp.Tags,
    hwc.HistoryDate,
    hwc.UserDisplayName AS Editor,
    hwc.EditComment,
    hwc.EditText
FROM 
    RankedPosts rp
LEFT JOIN 
    HistoryWithComments hwc ON rp.PostId = hwc.PostId
WHERE 
    rp.RankByTags = 1 
ORDER BY 
    rp.Tags, 
    rp.CreationDate DESC;