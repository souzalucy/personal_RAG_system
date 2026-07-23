--[[
bm25_scorer.lua — Tokenization helper for BM25.

Only the tokenize() function is used by the Python code (bm25_rag.py).
The actual BM25 scoring is done by the rank_bm25 Python library.
--]]

local BM25Scorer = {}

--- Tokenize text into lowercase words
--- @param text string
--- @return table list of words
function BM25Scorer.tokenize(text)
    if not text or text == "" then
        return {}
    end
    local words = {}
    for word in text:lower():gmatch("%w+") do
        words[#words + 1] = word
    end
    return words
end

return BM25Scorer
