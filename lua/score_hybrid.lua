--[[
score_hybrid.lua — Hybrid scoring: vector similarity + keyword boost.

This script re-ranks Qdrant search results by combining the vector
similarity score with a keyword match bonus. This is especially useful
for bilingual (EN/PT) search where exact keyword matches should be
preferred even if the semantic similarity is slightly lower.

Formula:
    final_score = vector_score + keyword_boost * keyword_match_ratio

Where:
    - vector_score: the cosine similarity from Qdrant (0.0 – 1.0)
    - keyword_boost: weight given to keyword matching (default 0.15)
    - keyword_match_ratio: fraction of query keywords found in the text

The Python code in vector_rag.py mirrors this logic since Qdrant's
local mode doesn't support arbitrary Lua execution on query results.
--]]

local HybridScorer = {}

-- Default boost weight for keyword matching
local DEFAULT_KEYWORD_BOOST = 0.15

--- Tokenize text into lowercase words
--- @param text string
--- @return table set of words
local function tokenize(text)
    local words = {}
    for word in text:lower():gmatch("%w+") do
        words[word] = true
    end
    return words
end

--- Compute the hybrid score for a single result
--- @param query string the user's search query
--- @param result_text string the text content of the result
--- @param vector_score number the original vector similarity score
--- @param keyword_boost number|nil optional boost weight (default 0.15)
--- @return number the hybrid score
function HybridScorer.compute(query, result_text, vector_score, keyword_boost)
    keyword_boost = keyword_boost or DEFAULT_KEYWORD_BOOST

    local query_words = tokenize(query)
    local text_words = tokenize(result_text)

    -- Count how many query keywords appear in the text
    local match_count = 0
    for word in pairs(query_words) do
        if text_words[word] then
            match_count = match_count + 1
        end
    end

    local total_query_words = 0
    for _ in pairs(query_words) do
        total_query_words = total_query_words + 1
    end

    if total_query_words == 0 then
        return vector_score
    end

    local match_ratio = match_count / total_query_words
    return vector_score + (keyword_boost * match_ratio)
end

--- Re-rank a list of results using hybrid scoring
--- @param query string
--- @param results table list of {text, score, ...}
--- @param keyword_boost number|nil
--- @return table re-ranked results
function HybridScorer.rerank(query, results, keyword_boost)
    for _, result in ipairs(results) do
        result.score = HybridScorer.compute(
            query,
            result.text or "",
            result.score or 0,
            keyword_boost
        )
    end

    -- Sort by new score descending
    table.sort(results, function(a, b)
        return a.score > b.score
    end)

    return results
end

return HybridScorer
