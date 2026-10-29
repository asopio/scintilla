# Idea for combined workflow using DeepSeek OCR2 + Qwen3.8 

## Model specifics
- Qwen3.8: 27B params, 256k-token context window (CW)
- DSOCR2 (DeepSeek-OCR2): 3B params, ~10k tokens per page

## Advantages of each model
- Qwen3.8: tool calls, fine-controlled output through prompting 
- DSOCR2: Small, fast, handles page-cropping better than vision-language model (VLM) / heuristic approaches

## Plan
- Process individual pages with DSOCR2
- Pass collated document to Qwen3.8 (need to test how many pages can fit into context!) to bring document into desired markdown format 
- Agentic review loop (also Qwen3.8) runs over whole document: ensures markdown compiles and fixes any remaining (not syntax-breaking) formatting errors 