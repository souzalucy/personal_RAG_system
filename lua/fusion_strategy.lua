--[[
fusion_strategy.lua — Source fusion strategies for combining
Vector RAG and BM25 results into a single ranked list.

Strategies:
  1. "reciprocal_rank" — Reciprocal Rank Fusion (RRF)
     score = 1 / (k + rank_in_vector) + 1 / (k + rank_in_bm25)
     This is the recommended approach for combining semantic + keyword results.

  2. "concatenate" — Simple concatenation: vector results first, then BM25.
     No re-ranking, just append BM25 results that aren't already in vector results.

The strategy is selected via features.lua → fusion.strategy.
--]]

local FusionStrategy = {}

--- Reciprocal Rank Fusion: combine two ranked lists
--- @param vector_results table list of {method="vector_rag", content, source, page, score}
--- @param bm25_results table list of {method="bm25", content, source, page, score}
--- @param k number RRF constant (default 60)
--- @return table fused and re-ranked results
function FusionStrategy.reciprocal_rank(vector_results, bm25_results, k)
    k = k or 60

    -- Build a map of content -> RRF score accumulator
    local rrf_scores = {}  -- content_hash -> {score, method, content, source, page}

    local function content_key(item)
        return (item.source or "") .. "::" .. (item.text or ""):sub(1, 200)
    end

    -- Process vector results
    for rank, item in ipairs(vector_results) do
        local key = content_key(item)
        if not rrf_scores[key] then
            rrf_scores[key] = {
                method = "vector_rag",
                content = item.text,
                source = item.source or item.document,
                page = item.page,
                rrf_score = 0,
                vector_rank = rank,
                bm25_rank = nil,
            }
        end
        rrf_scores[key].rrf_score = rrf_scores[key].rrf_score + (1 / (k + rank))
        rrf_scores[key].vector_rank = rank
    end

    -- Process BM25 results
    for rank, item in ipairs(bm25_results) do
        local key = content_key(item)
        if not rrf_scores[key] then
            rrf_scores[key] = {
                method = "bm25",
                content = item.text,
                source = item.source or item.document,
                page = item.page,
                rrf_score = 0,
                vector_rank = nil,
                bm25_rank = rank,
            }
        else
            -- Already seen from vector — mark as both
            rrf_scores[key].method = "hybrid"
        end
        rrf_scores[key].rrf_score = rrf_scores[key].rrf_score + (1 / (k + rank))
        rrf_scores[key].bm25_rank = rank
    end

    -- Convert to sorted list
    local fused = {}
    for _, v in pairs(rrf_scores) do
        fused[#fused + 1] = {
            method = v.method,
            content = v.content,
            source = v.source,
            page = v.page,
            score = v.rrf_score,
        }
    end

    -- Sort by RRF score descending
    table.sort(fused, function(a, b)
        return a.score > b.score
    end)

    return fused
end

--- Concatenate strategy: vector results first, then BM25 results not already present
--- @param vector_results table
--- @param bm25_results table
--- @return table concatenated results
function FusionStrategy.concatenate(vector_results, bm25_results)
    local seen = {}
    local function content_key(item)
        return (item.source or "") .. "::" .. (item.text or ""):sub(1, 200)
    end

    local result = {}
    for _, item in ipairs(vector_results) do
        result[#result + 1] = item
        seen[content_key(item)] = true
    end

    for _, item in ipairs(bm25_results) do
        if not seen[content_key(item)] then
            result[#result + 1] = item
            seen[content_key(item)] = true
        end
    end

    return result
end

--- Fuse results using the configured strategy
--- @param strategy string "reciprocal_rank" or "concatenate"
--- @param vector_results table
--- @param bm25_results table
--- @param k number|nil RRF constant (only used for reciprocal_rank)
--- @return table fused results
function FusionStrategy.fuse(strategy, vector_results, bm25_results, k)
    if strategy == "reciprocal_rank" then
        return FusionStrategy.reciprocal_rank(vector_results, bm25_results, k)
    elseif strategy == "concatenate" then
        return FusionStrategy.concatenate(vector_results, bm25_results)
    else
        -- Default to reciprocal rank
        return FusionStrategy.reciprocal_rank(vector_results, bm25_results, k)
    end
end

return FusionStrategy
