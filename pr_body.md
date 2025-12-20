## Summary

This PR adds assertion generation capabilities to the AutoQ pipeline for data-local and data-global questions. Assertions are factual statements that can be used to evaluate the accuracy of RAG system answers.

## Key Changes

### New Features
- **Assertion Generation**: Added `LocalClaimAssertionGenerator` and `GlobalClaimAssertionGenerator` with a map-reduce approach for processing claims into ranked assertions
- **Assertion Validation**: Optional validation step that scores assertions on grounding, relevance, and verifiability (enabled by default with min score 3/5)

### Configuration
- New `assertions` config section with:
  - `max_assertions`: Maximum assertions per question (default: 10)
  - `enable_validation`: Enable/disable validation (default: true)
  - `min_validation_score`: Minimum score threshold (default: 3)
  - `batch_size`: Batch size for processing (default: 50)
  - `max_data_tokens`: Max tokens per batch (default: 32000)

### CLI Updates
- Added assertion configuration options to settings.yaml

### Documentation
- Updated CLI docs with assertion configuration options
- Updated autoq notebook with assertion settings
- Added assertion_gen notebook for generating assertions retrospectively

## Output Format

Assertions are saved to `assertions.json` in each question output folder:
```json
{
  "question_id": "...",
  "question_text": "...",
  "assertions": [
    {
      "statement": "The response should state that...",
      "source_count": 2,
      "score": 10,
      "rank": 1,
      "validation": {
        "is_valid": true,
        "scores": {
          "grounding": 5,
          "relevance": 5,
          "verifiability": 5
        }
      }
    }
  ]
}
```

## Testing
- Tested CLI end-to-end with AP news dataset
- Verified assertion generation for both data-local and data-global questions
- Confirmed validation scoring works correctly
