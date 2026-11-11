# AI Interaction Record – Phase I

**Group name:** Cell Investigators  
**Pair:** Peter Fischer (12102665), Abdul Samad Saleem Rashid (12248151)  
**Model:** Cellpose3 (denoise_cyto3, cyto3)  
**Dataset:** cyto2 (train_cyto2)  
**GenAI tool used:** Claude Sonnet 5.5  

---
## 1. General Setup

### Creation of a dedicated Claude Project

- **Tool:** Claude (Projects)
- **Phase I step:** Setup (before any analysis)
- **Goal:** Provide the AI with persistent, shared context about our task, model, and working rules, so we do not need to repeat them in every chat and so that answers stay consistent across sessions.
- **What was set up:** A Claude Project named "AI4Vis" containing the project instructions and reference material.


#### 1.1 Project instructions

Original (German):
```text
Denke bei komplexen Problemen schritt für schritt. Sollte etwas unklar sein, frag nach und nimm nichts an. Gib mir genaue Referenzen wo du Informationen gefunden hast.
```
English translation:
```text
Think step by step for complex problems. If anything is unclear, ask and do not make assumptions. Give me exact references for where you found information.
```

#### 1.2 Project knowledge

| File | Content | Purpose |
|------|---------|---------|
| `Task.txt` | Assignment text + our own task description (model, datasets, Phase I/II goals) | Give the AI the full task context |
| `Cellpose1.pdf` | Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x | Source for answers and references |
| `Cellpose2.pdf` | Pachitariu, Marius & Stringer, Carsen. (2022). Cellpose 2.0: how to train your own model. Nature Methods. 19. 1-8. 10.1038/s41592-022-01663-4.  | Source for answers and references |
| `Cellpose3.pdf` | Stringer, C., Pachitariu, M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat Methods 22, 592–599 (2025). https://doi.org/10.1038/s41592-025-02595-5 | Source for answers and references |
*Additional Information: Cellpose 1–3 build on one another and reference earlier versions. For this reason, the older models were also included in the project as context.*

## 2. Interaction Entries

### Entry 1 – Reading and comparing the Cellpose 1/2/3 papers

- **Date / tool / model:** 2026-10-27 / Claude Sonnet 5.5
- **Phase I step:** Literature grounding / hypothesis generation
- **Goal:** Summarize the Cellpose3 paper and find relevant information to edge cases, bias and hallucinations, and compare training/test sets with Cellpose 1, 2 and Cellpose-SAM.
- **Prompt (original):**
  ```text
  Fasse mir das Paper im kontext (Cellpose3) zusammen. gehe dann speziell drauf ein, was für die aufgabe relevant sein könnte. Außerdem fasse mir zusammen welches training und welches test set verwendet wurde. Ist es das gleiche wie bei cellpose 1-2 bzw. cellpose-SAM? Was sind die unterschiede zu cellpose 1-2?

  Follow-up: Das paper ist im kontext im projekt!
  ```
- **Prompt (english):**
  ```text
  Summarize the paper for me in context (Cellpose3). Then go into specific detail about what might be relevant to the assignment. Also, summarize which training and test sets were used. Are they the same as those used in Cellpose 1-2 or Cellpose-SAM? What are the differences compared to Cellpose 1-2?

  Follow-up: The paper is part of the project!
  ```
- **AI output (summary):** 
First answer: the paper is not available, please re-upload. After the follow-up it read the project file and produced a section/page-anchored summary. 
Flagged "618 vs. 540+68" as a probable typo. Didn't recognize that the cited article for Cellpose2 is also in the project knowledge.
- **Our evaluation:** Claude separated paper statements from inferences and listed what it could not verify. Checked the typo and claude was correct. Read the paper before to evaluate the summary: It was okay, but it forgot a lot of important information especially regarding explicit training and preprocessing pipeline, instead it produced a very light overview. 
- **Decision:** refined with follow-up prompt
- **Why:** Follow-up questions on architecture.

---

### Entry 2 – Pipeline code, `cyto2` data loading and `Degradation` class

- **Date / tool / model:** 2026-11-03 / Claude Sonnet 5.5
- **Phase I step:** Dataset handling / degradation pipeline
- **Goal:** Run denoising and segmentation as separate models, understand `channels`/`diameter`/`flow_threshold`, load image/mask pairs and reproduce the paper's degradations.
- **Prompt (original):**
  ```text
  gib mir python code um die cellpose3 pipeline laufen zu lassen. ich habe ip install "cellpose==3.1.1.2"

  ich möchte das denoising modell und das segmentation modell einzeln laden und verwenden.

  warum ist channels und diameter als parameter wichtig? Wie ist die genaue pipeline von cellpose 3? kann ich auch direkt den output vom modell evaluieren? (ohne flow_threshold)?

  [...] Außerdem schreibe eine klasse degradation welche die oben genannten methoden auf das input bild anwendet und wenn ein ground truth mitgegeben wird und es um skalierung oder drehungen geht, dieses auch gleich angewendet wird. ein screenshot vom daten ordner gebe ich mit.
  ```
- **Prompt (english):**
  ```text
   Give me some Python code to run the CellPose 3 pipeline. I have `ip install “cellpose==3.1.1.2”`.

  I want to load and use the denoising model and the segmentation model separately.

  Why are `channels` and `diameter` important as parameters? What is the exact CellPose 3 pipeline? Can I also evaluate the model's output directly? (Without `flow_threshold`)?

   [...] Also, write a class `degradation` that applies the methods mentioned above to the input image, and if a ground truth is provided and scaling or rotation is involved, apply those adjustments immediately. I'm including a screenshot of the data folder.
  ```
- **Context given to the AI:** Project papers; `samplename`/`mask_samplename` file pairs of the downloaded `cyto2` data; folder screenshot;
- **AI output (summary):** Code for separate `DenoiseModel` and `CellposeModel`. Explanation of `channels` and `diameter` and the inference pipeline, with notes on which parts are documented vs. not verified. Module `cellpose_data_tools.py` with `find_samples`/`load_sample` and `Degradation` (photometric ops leave the mask unchanged; geometric ops transform the mask with nearest neighbor), tested on synthetic pairs (alignment, labels, seeds).
- **Evaluation:** Tested by Claude only on synthetic data. Validated by us on real `cyto2` samples. Visual validation of the code by us.
- **Decision:** modified
- **Why:** The class is reused in the analysis pipeline (Entry 9).
- **Effect on subsequent analysis:** Basis for all degradation experiments.
- **Failure pattern:** Claude searched the web although the paper was in the Project (corrected by the user).

---

### Entry 3 – Choosing metrics for denoised vs. original image

- **Date / tool / model:** 2026-11-05 / Claude Sonnet 5.5
- **Phase I step:** Metrics
- **Goal:** Find a suitable metric to compare the denoised image with the original image before the added noise.
- **Prompt (original):**
  ```text
  welche metrik ist gut geeignet um die ähnlichkeit zweier bilder zu identifizieren?

  ich möchte denoised und orignalbild (von augmentation) vergleichen
  ```
- **Prompt (english):**
  ```text
  Which metric is well suited to identify the similarity of two images?

  I want to compare the denoised and the original image (from augmentation).
  ```
- **AI output (summary):** Claude first asked clarifying questions and then recommended SSIM, PSNR as a baseline, LPIPS and AP@0.5 from Cellpose. It argued that pixel metrics may undervalue good restorations, because Cellpose3 is trained with a perceptual and a segmentation loss. It also stated that the validity of LPIPS for microscopy images is unverified.
- **Our evaluation:** We evaluated LPIPS as a better parameter than the baseline for general simularity between the input image and the denoised one. For edge cases (like residual small amount of poisson noise) PSNR showed better performance.
- **Decision:** modified
- **Why:** We used only a subset of the metrics that Claude proposed later (see Entry 6).

---

### Entry 4 – Channel handling and denoiser configuration

- **Date / tool / model:** 2026-11-06 / Claude Sonnet 5.5
- **Phase I step:** Pipeline correctness
- **Goal:** Configure `channels` and `diameter` correctly for our two channel fluorescence images that are stored as RGB.
- **Prompt (original):**
  ```text
  überprüfe die einstellungen der modelle auf diesen input: Shape Original: (512, 480, 3) | Shape denoised: (512, 480)

  das seg. modell ohne denois schafft 42/49 treffer, das seg modell mit nur 40/49. [code omitted]
  ```
- **Prompt (english):**
  ```text
  Check the model settings for this input: Shape Original: (512, 480, 3) | Shape denoised: (512, 480)

  The seg. model without denoise achieves 42/49 hits, the seg. model with it only 40/49. [code omitted]
  ```
- **Context given to the AI:** Our code, the shape output and screenshots of the image.
- **AI output (summary):** Claude recognized a two channel image stored as RGB and pointed out that `channels=[0,0]` averages the RGB channels. It noticed that a comment in our code contradicted its own assumption and asked instead of deciding. By reading the source of `denoise.py` it found a 30/17 rescaling of the second channel and a swapped channel order in the denoised image. It also explained that 42/49 are object counts and not hits and offered an IoU based count.
- **Our evaluation:** After some discussion with claude about the right usage of `channels` we checked it in the documentation.
- **Decision:** refined with follow-up prompt
- **Why:** The first versions compared two unequal pipelines, so several correction rounds were needed.
- **Failure pattern:** The first code version used `channels=[0,0]` and the channel conventions differed between the code paths.

---

### Entry 5 – Effect of double denoising

- **Date / tool / model:** 2026-11-06 / Claude Sonnet 5.5
- **Phase I step:** Experiment design and code review
- **Goal:** Find out what happens to the segmentation when an image passes twice through the denoising network.
- **Prompt (original):**
  ```text
  ich möchte herrausfinden, was doppeltes durchlaufen des denoising networks für einen einfluss hat. wo ist der fehler im code? [code omitted]

  überprüfe: [code omitted]
  ```
- **Prompt (english):**
  ```text
  I want to find out what influence passing through the denoising network twice has. Where is the error in the code? [code omitted]

  Check: [code omitted]
  ```
- **AI output (summary):** Claude found that hardcoded channel indices swap cytoplasm and nucleus in the second pass, which confounds the comparison. After our fix it confirmed the new channel order by reading the code, without running it.
- **Decision:** refined with follow-up prompt
- **Why:** We wrote our own fix and let Claude check it in a second round.

---

### Entry 6 – Degradation and robustness analysis pipeline

- **Date / tool / model:** 2026-11-06 to 2026-11-09 / Claude Sonnet 5.5
- **Phase I step:** Degradation pipeline, metrics and visualization
- **Goal:** Find the degradations or combinations of degradations where the metrics collapse, for example through merging cell boundaries.
- **Prompt (original):**
  ```text
  schreib mir code, welcher diese klasse verwendet um ein sample (das gebe ich) zu verändern (augmentation). danach möchte ich anhand von metriken (IoU für die segmentierung, LPIPS für bildähnlichkeit von original bild zu denoised bild) auswerten bei welchen veränderungen (oder kombinationen aus veränderungen) gewisse einbrüche in den metriken sind (z.b. durch verschmelzen von zellgrenzen). Wenn du ideen hast für andere metriken, dann frag mich zuerst bevor du den code erstellst, ob du sie auch einbauen sollst. der code soll die augmentation (mit der zur verfügung gestellten klasse), die metrik berechnung und die darstellung der resultate.

  A. nimm 1., 3. 4. 5. und 6.     B.: 1. unveränderte original zu denoised 2. Ja füge sonst einen kanal mit 0er hinzu 3. von dir sinnvoll interpretierte paare, auch mehr wie nur 2, aber auch nur parameterreihen. ich möchte so viel wie möglich abdecken, 4. schlage selber vor 5. ja 6. ansich neu festlegen aber auch die möglichkeit falsch gesetzer durchmesser testen (wie einen eigenen "augmentation" parameter) 7. Ja beide, 8. ich gebe dir ein einzelnes sampel. 9. nimm die für den task (z.b. heatmap für kombinationen, lineplots pro operation) die sinnvollste darstellungsart (bestimme selber). 10. Ja. C: 1. verwende auch rotate, diese erkenntnis ist auch wichtig.
  ```
- **Prompt (english):**
  ```text
  Write me code that uses this class to modify a sample (which I provide), i.e. augmentation. Afterwards I want to use metrics (IoU for the segmentation, LPIPS for image similarity from original image to denoised image) to evaluate for which changes (or combinations of changes) certain drops in the metrics occur (e.g. through merging of cell boundaries). If you have ideas for other metrics, ask me first before you create the code whether you should include them too. The code should contain the augmentation (with the provided class), the metric calculation and the presentation of the results.

  A. Take 1., 3., 4., 5. and 6.     B.: 1. unchanged original to denoised 2. Yes, otherwise add a channel with zeros 3. Pairs sensibly interpreted by you, also more than just 2, but only parameter series. I want to cover as much as possible, 4. Propose yourself 5. Yes 6. Basically set anew, but also the possibility to test wrongly set diameters (like an own "augmentation" parameter) 7. Yes both, 8. I give you a single sample. 9. Take the most sensible type of presentation for the task (e.g. heatmap for combinations, line plots per operation), decide yourself. 10. Yes. C: 1. Use rotate as well, this finding is also important.
  ```
- **Context given to the AI:** Our `Degradation` class and one sample with ground truth.
- **AI output (summary):** As requested, Claude asked before coding and proposed six extra metrics and ten setup questions. It then delivered `degradation_analysis.py` with single parameter sweeps, paired grids, a severity ladder, a diameter factor and metrics such as IoU, AP, merge and split rates, boundary F1, LPIPS, SSIM and PSNR. Results come as line plots, heatmaps, example images and CSV files. It tested the code only with synthetic data and placeholder models.
- **Our evaluation:** We visually evaluated the code.
- **Decision:** modified and refined with follow-up prompt
- **Why:** We selected the metrics ourselves and added the wrong diameter test and the rotation.
- **Failure pattern:** In the end it added matrices and outputs we didn't ask it to and also didn't confirm. 
---

### Entry 7 – Visual inspection of the anomalies

- **Date / tool / model:** 2026-11-09 / Claude Sonnet 5.5
- **Phase I step:** Interpreting results and visualization
- **Goal:** Look at the anomalies of Entry 6 as images and test hypotheses about their cause.
- **Prompt (original):**
  ```text
  lies die projekt anweisungen neu. und schreibe mir ein script um die auffälligkeiten als bilder zu begutachten. Speziallisiere dich auf diese auffälligkeiten: [list of anomalies omitted]
  ```
- **Prompt (english):**
  ```text
  Reread the project instructions. And write me a script to inspect the anomalies as images. Specialize on these anomalies: [list of anomalies omitted]
  ```
- **Context given to the AI:** Results of the matrices after visual evaluation and the list of next steps that Claude had proposed before.
- **AI output (summary):** Claude wrote `inspect_anomalies.py` with sections for noise, blur, diameter, smoothing, combinations and CSV plots. Every predicted object is classified as matched, merged, extra or poor. The script tests whether the extra objects are blobs in the red channel only. Claude had marked earlier explanations as hypotheses and listed what it did not know.
- **Why:** To visually evaluate the output of every processing step.
- **Failure pattern:** Had to remember claude to use the project insturctions.

---

### Entry 8 – Poisson noise formula and role of `diameter`

- **Date / tool / model:** 2026-11-09 / Claude Sonnet 5.5
- **Phase I step:** Understanding the degradations
- **Goal:** Understand how Poisson noise is added in Cellpose3 and how `diameter` enters the denoising model.
- **Prompt (original):**
  ```text
  durch welche formel ist das hinzufügen von poisson rauschen erklärbar? (welche wird angewendet)?

  warum ergibt ein kleineres alpha dann mehr rauschen?

  wird der durchmesser parameter bei cellpose 3 beim denoising modell einfach in das modell eingespeist oder ist er vorverarbeitung?
  ```
- **Prompt (english):**
  ```text
  By which formula can the addition of Poisson noise be explained? (Which one is applied?)

  Why does a smaller alpha then result in more noise?

  Is the diameter parameter in Cellpose 3 simply fed into the denoising model, or is it preprocessing?
  ```
- **AI output (summary):** Claude gave the formula with a Gamma distributed scaling factor and stated that the paper has no written formula, so it is Claude's own translation of the text. It corrected the false premise of our second question, because only β varies and α stays fixed at 4. It explained `diameter` as preprocessing through rescaling and noted that a wrong diameter may lead to merging or splitting. It could not verify how `denoise.py` handles the diameter.
- **Our evaluation:** Because of the same problem as in Entry 1 (for the poisson noise explaination) and because it couldn't verify how the diameter variable was handled, we looked it up in the internet.
- **Decision:** refined with follow-up prompt
- **Why:** The second prompt exposed a wrong premise, and the diameter hypothesis matches our diameter factor experiment.

---


