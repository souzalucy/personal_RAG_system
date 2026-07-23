--[[
prompt_builder.lua — LLM prompt construction with token budget management.

This script builds the system prompt and user message for the Groq LLM,
handling context truncation within a token budget.

The Python code in llm.py previously did this inline; now it delegates
to this Lua script so the prompt logic can be modified without redeployment.
--]]

local PromptBuilder = {}

--- Truncate text to fit within a character budget (rough ~4 chars/token)
--- @param text string
--- @param max_chars number
--- @return string
local function truncate_to_budget(text, max_chars)
    if not text or #text <= max_chars then
        return text or ""
    end
    -- Truncate at the last space within budget
    local truncated = text:sub(1, max_chars)
    local last_space = truncated:match("^.*()%s")
    if last_space then
        return truncated:sub(1, last_space - 1) .. " [...]"
    end
    return truncated .. " [...]"
end

--- Build the system prompt
--- @return string
function PromptBuilder.system_prompt()
    return [[You are a helpful research assistant. Answer the user's question based on the provided context. Always include the relevant retrieved text in your answer, citing the source method (Vector RAG or BM25) and page number. If the context doesn't contain enough information to fully answer the question, say so but still summarize what the retrieved text does contain.]]
end

--- Build the full messages array for the LLM call
--- @param question string the user's question
--- @param contexts table list of context dicts: {{method, content, source, page}, ...}
--- @param max_context_tokens number max tokens for context (default 9000)
--- @return table messages: {{role = "system", content = string}, {role = "user", content = string}}
function PromptBuilder.build_messages(question, contexts, max_context_tokens)
    max_context_tokens = max_context_tokens or 9000
    local max_chars = max_context_tokens * 4  -- rough chars-to-tokens ratio

    local context_parts = {}
    local budget_remaining = max_chars

    for i, ctx in ipairs(contexts) do
        local method = ctx.method or "unknown"
        local content = ctx.content or ""
        local source = ctx.source or ctx.document or "unknown"
        local page = ctx.page
        local page_str = ""
        if page then
            page_str = " (page " .. tostring(page) .. ")"
        end
        local prefix = "[Source: " .. method .. " - " .. source .. page_str .. "]\n"

        -- Allocate chars for this chunk proportionally
        local remaining = math.max(1, #contexts - #context_parts)
        local chunk_budget = math.max(200, math.floor(budget_remaining / remaining))
        local truncated = truncate_to_budget(content, chunk_budget - #prefix)
        local part = prefix .. truncated
        context_parts[#context_parts + 1] = part
        budget_remaining = budget_remaining - #part

        if budget_remaining <= 0 then
            break
        end
    end

    local context_str = table.concat(context_parts, "\n\n")

    local user_content = "Context:\n" .. context_str .. "\n\nQuestion: " .. question .. "\n\nAnswer based on the context above:"

    return {
        { role = "system", content = PromptBuilder.system_prompt() },
        { role = "user", content = user_content },
    }
end

return PromptBuilder
