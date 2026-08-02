--[[
features.lua — Central feature flag configuration.

All features can be toggled at runtime by editing this file.
Changes take effect on the NEXT request — no redeployment needed.

Usage:
  local features = require("features")
  if features.bm25_enabled.enabled then
    -- run BM25 search
  end
--]]

return {
  -- Hybrid scoring: boost vector results with keyword matches
  hybrid_scoring = {
    enabled = true,
    keyword_boost = 0.15,
  },

  -- Language detection & filtering (English / Portuguese)
  language_filter = {
    enabled = true,
  },

  -- Query normalization: lowercase, remove accents, strip punctuation
  query_normalization = {
    enabled = true,
  },

  -- BM25 keyword-based search (vectorless RAG)
  bm25_enabled = {
    enabled = true,
  },

  -- Fusion strategy for combining Vector RAG + BM25 results
  fusion = {
    enabled = true,
    strategy = "reciprocal_rank",  -- "reciprocal_rank" | "concatenate"
    reciprocal_rank_k = 60,        -- constant for RRF formula
  },

  -- Document deletion: allow removing PDFs from storage and vector DB
  document_deletion = {
    enabled = true,
  },

  -- Debug logging (prints extra info to stdout)
  debug_logging = {
    enabled = false,
  },
}
