--[[
normalize_query.lua — Query text normalization for bilingual (EN/PT) search.

Loaded by vector_rag.py via lupa and executed in LuaJIT.

What it does:
  1. Lowercases the query
  2. Strips Portuguese accents (a -> a, c -> c, etc.)
  3. Collapses multiple whitespace characters
  4. Removes punctuation that doesn't carry semantic weight

This ensures that "coracao" and "coracao" match the same content,
and that "como funciona?" and "como funciona" are treated identically.
--]]

local QueryNormalizer = {}

-- Characters to remove entirely (punctuation that doesn't add meaning)
local PUNCTUATION_REMOVE = "[%p%%%$%#%*%+%=%<%>%[%]%^%{%}|%~]"

--- Normalize a query string
--- @param query string
--- @return string normalized query
function QueryNormalizer.normalize(query)
    if not query or query == "" then
        return ""
    end

    -- 1. Lowercase
    local text = query:lower()

    -- 2. Replace accented characters using byte-level substitution
    --    LuaJIT handles UTF-8 strings as bytes, so we use gsub with
    --    the actual UTF-8 byte sequences for accented characters.
    local accent_replacements = {
        ["\195\161"] = "a",  -- a
        ["\195\160"] = "a",  -- a
        ["\195\163"] = "a",  -- a
        ["\195\162"] = "a",  -- a
        ["\195\169"] = "e",  -- e
        ["\195\168"] = "e",  -- e
        ["\195\170"] = "e",  -- e
        ["\195\173"] = "i",  -- i
        ["\195\172"] = "i",  -- i
        ["\195\174"] = "i",  -- i
        ["\195\179"] = "o",  -- o
        ["\195\178"] = "o",  -- o
        ["\195\181"] = "o",  -- o
        ["\195\180"] = "o",  -- o
        ["\195\186"] = "u",  -- u
        ["\195\185"] = "u",  -- u
        ["\195\187"] = "u",  -- u
        ["\195\167"] = "c",  -- c
        ["\195\177"] = "n",  -- n
    }

    for utf8_bytes, plain in pairs(accent_replacements) do
        text = text:gsub(utf8_bytes, plain)
    end

    -- 3. Remove punctuation
    text = text:gsub(PUNCTUATION_REMOVE, " ")

    -- 4. Collapse whitespace and trim
    text = text:gsub("%s+", " ")
    text = text:match("^%s*(.-)%s*$") or text

    return text
end

return QueryNormalizer
