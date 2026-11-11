# Reflection on GenAI Usage – Phase I

**AI-aided pair:**

| First name | Last name | Student ID |
|------------|-----------|------------|
| Peter      | Fischer   | 12102665   |
| Abdul Samad      | Saleem Rashid     | 12248151   |

**Group name:** Cell Investigators  
**Model:** Cellpose3 (denoise_cyto3, cyto3)  
**Dataset:** cyto2 (train_cyto2)  
**GenAI tool used:** Claude Sonnet 5.5  
**Detailed record:** see `ai-interactions/`
---

## 1. Coding

### Failure patterns observed

- It often uses global variables where they aren't necessary.
- If you don't define it, the code design often includes large function with poorly defined functionalities doing many things at the same time. Therefore it is often the case that parts of different functions are the same. This leads to many lines of code which are hard to validate and even more difficult to fix if something is wrong.
- Hardcoded values like channel indices broke the code as soon as the function was reused on a different input, for example in the second denoising pass (detailed record: `ai-interactions/interaction_log.md`, Entry 5).
- The code often ran without errors but still compared two unequal pipelines, because channel conventions differed between code paths. Such errors are silent and only show up as strange results (detailed record: `ai-interactions/interaction_log.md`, Entry 4).
- Claude tested its code only on synthetic data and placeholder models, so a passing test said little about correctness on real samples (detailed record: `ai-interactions/interaction_log.md`, Entry 2 and Entry 6).
- Claude added matrices and outputs that we did not ask for and did not confirm, which made the results harder to check (detailed record: `ai-interactions/interaction_log.md`, Entry 6).
- In longer chats Claude drifted away from the project instructions, so we had to remind it to follow them (detailed record: `ai-interactions/interaction_log.md`, Entry 7).


### Emerging Best Practices

- Define code and process architecture as explicit as possible and leave as little room for interpretation as possible.
- Give the advice to ask if something is unclear and not to hallucinate.
- Write down input and output conventions such as channel order, value range and shapes in the prompt and keep them identical in all functions.
- Ask Claude to ask before coding and to propose extras only after confirmation. This worked well in our robustness pipeline (detailed record: `ai-interactions/interaction_log.md`, Entry 6).
- Always run the generated code on real samples and inspect every intermediate output visually before trusting a metric.
- Use Claude as a code reviewer for your own code, but rerun the result yourself. It found the channel swap in our double denoising code (detailed record: `ai-interactions/interaction_log.md`, Entry 5).
- Repeat the most important project rules in the prompt when the chat gets long.

---

## 2. Analysis

### Failure patterns observed

- Summaries: The summary of the cellpose 3 paper was a very light overview. We think just by asking follow-up questions at some point someone would find all the relevant information for the task, but we also think that it's faster just to read the article (also to get all of the important information and don't miss anything) and then check with the LLM if something is unclear. Another problem could be, if we wouldn't read the paper before, it can be that we wouldn't even know what we esplicitly should ask to get all of the information. (detailed record: `ai-interactions/interaction_log.md`, Entry 1)
- downsampling vs using wrong `diameter` value: Claude gave us the advice to evaluate downsampling (downsampling by a factor and then upsampling again) and a false diameter value by its own.
After running the experiments, we realized that this doesn't make sense due to the processing pipeline and the influence of the `diameter` parameter, because the diameter is just used to rescale the input as a pre-procssing step.
- Claude searched the web or asked for an upload although the papers were already in the project knowledge, so answers were sometimes based on weaker sources (detailed record: `ai-interactions/interaction_log.md`, Entry 1 and Entry 2).
- Some explanations were Claude's own translation of the paper and not a statement from it. The Poisson noise formula and the handling of `diameter` in the denoiser could not be verified by Claude, so we had to look them up elsewhere (detailed record: `ai-interactions/interaction_log.md`, Entry 8).
- General metric recommendations did not fit our edge cases. LPIPS worked better for general similarity, but PSNR was more sensitive to a small amount of residual Poisson noise (detailed record: `ai-interactions/interaction_log.md`, Entry 3).
- Claude read the source code of `denoise.py` instead of running it, so its findings about rescaling and channel order were plausible but needed our own check in the documentation (detailed record: `ai-interactions/interaction_log.md`, Entry 4).


### Emerging Best Practices

- Give the advice to think step by step.
- Give the advice to ask if something is unclear and not to hallucinate.
- Read the key paper yourself first and use Claude only to check unclear points or to compare versions.
- Ask Claude to separate statements from the paper from its own inferences and to list what it could not verify. This made it easy to see which claims needed a check (detailed record: `ai-interactions/interaction_log.md`, Entry 1 and Entry 8).
- Check central claims about parameters and formulas in the official documentation or the source code.
- Use several metrics next to each other and compare them on the same edge case instead of relying on one recommendation.
- Question the premise of Claude proposed experiments against the processing pipeline before running them.

---
