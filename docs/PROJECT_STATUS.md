# L4D2 AI Project Status

## Last Updated: January 6, 2026

## Overview
This project is developing AI models for Left 4 Dead 2 SourcePawn plugin generation using both open-source (TinyLlama) and proprietary (OpenAI GPT-4o-mini) approaches.

## Current Status

### ✅ Completed
1. **Vultr TinyLlama LoRA Training**
   - Model trained on A100 (20GB slice)
   - 971 examples, 3 epochs, 183 steps
   - Final loss: 0.531, mean_token_accuracy: 0.875
   - Adapter size: ~97MB
   - Location: `model_adapters/l4d2-lora/`

2. **OpenAI GPT-4o-mini v1 Fine-tuning**
   - Job ID: `ftjob-tra4SBK5I334Z39ctcjPMpaf`
   - Model: `ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod:CusA2jFo`
   - Training: 921 samples, 50 eval samples, 3 epochs
   - **Result**: 0% task-appropriate responses (complete failure)

3. **Data Quality Analysis**
   - Identified root cause: 69.3% of training prompts were vague "Implement:" style
   - Only 26.9% had clear task descriptions
   - Created filtering script to remove low-quality examples

4. **OpenAI GPT-4o-mini v2 Fine-tuning (In Progress)**
   - Job ID: `ftjob-kKTJkKIRuX0UX4jLskoxgJu4`
   - Cleaned dataset: 517 high-quality examples
   - Added 15 synthetic examples for common patterns
   - Status: Currently training (~30-60 min remaining)

### 🔄 In Progress
1. **TinyLlama Evaluation**
   - Scripts ready: `scripts/evaluation/run_tinyllama_eval.py`
   - Need SSH access to Vultr instance (108.61.127.209)
   - Will run same 15 test cases as OpenAI evaluation

2. **OpenAI v2 Evaluation**
   - Waiting for fine-tuning completion
   - Will evaluate on same 15 test cases
   - Expected significant improvement due to cleaned data

### 📋 Pending
1. **Model Comparison Report**
   - Script ready: `scripts/evaluation/compare_models.py`
   - Waiting for both model evaluations
   - Will include cost analysis and recommendations

2. **Documentation Updates**
   - Add Vultr post-run checklist
   - Note about 20GB A100 slices for Mistral
   - Update batch size recommendations

## Key Findings

### Data Quality is Critical
- v1 model failure directly linked to poor training data
- 69% of examples were documentation snippets, not clear tasks
- v2 uses filtered data (57% retention) + synthetic examples

### Cost Comparison
| Approach | Training Cost | Inference Cost | Quality |
|----------|---------------|----------------|---------|
| OpenAI v1 | $30 | API per token | 0% success |
| TinyLlama | $25 credits | Free | TBD |
| OpenAI v2 | $20 | API per token | TBD |

### Technical Architecture
- Both models use same training data format (ChatML JSONL)
- TinyLlama: 1.1B parameters + LoRA adapter
- OpenAI: GPT-4o-mini base with fine-tuning

## Next Steps

### Immediate (Today)
1. SSH to Vultr and run TinyLlama evaluation
2. Monitor OpenAI v2 completion
3. Run OpenAI v2 evaluation
4. Generate comparison report

### Short Term (This Week)
1. Based on results, decide on production approach
2. If v2 successful: Create deployment guide
3. If v2 fails: Consider RAG or few-shot alternatives
4. Update training pipeline with quality filters

### Long Term
1. Expand dataset with more high-quality examples
2. Add VScript support alongside SourcePawn
3. Create plugin testing framework
4. Investigate larger open-source models

## Resources
- Vultr Credits: $225 remaining
- Evaluation Scripts: `scripts/evaluation/`
- Training Data: `data/openai_finetune/train_v2.jsonl`
- Models: `model_adapters/l4d2-lora/` (local), OpenAI API (cloud)

## Risks
- OpenAI v2 may still underperform if data quality insufficient
- TinyLlama limited by 1.1B parameter base model
- API costs for OpenAI production usage
- Need for consistent evaluation framework

---
*Status updated automatically by evaluation framework*
