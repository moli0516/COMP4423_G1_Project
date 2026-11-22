# DeskCheck Part C individual report working draft

The Part C application compares two photographs of the same desk and reports which objects changed. The released E0 model produced the same 23 event records through the UI, single-pair CLI and batch runner on four validation pairs. Against the manual annotations, 22 of those 23 records were correct. In a separate training experiment on an RTX 2060 SUPER, increasing input size from 640 to 960 improved validation mAP50–95 from 0.84079 to 0.88475. All eight validation photos came from two capture groups. These are validation results. The complete held-out evaluation and the student's personal account belong in the final individual PDF. [E0 validation](REAL_BATCH_EVALUATION.md#formal-v1-results) · [Resolution experiment](C_E2_RESOLUTION_PROTOCOL.md)

## Introduction

DeskCheck tracks six classes in the order bottle, cup, keyboard, mouse, book and scissors. Each pair contains at most one target instance per class in the basic task. C supplies the image loader, comparison service, Streamlit UI, CLI, exports and the matched resolution experiment. A supplies data, annotations, the trained detector and shared trainer. B supplies filtering, matching, event classification and evaluation. [Source and ownership](../README.md)

## Method

### 1. Program design and testing

C decodes the two originals, applies EXIF orientation and passes canonical RGB arrays to A’s detector. The detector handles model resizing and returns boxes in oriented source pixels. B keeps the selected detection per class. A class on the after side alone is added, one on the before side alone is removed, and a class absent from both yields no record. B matches same-class detections first. For a match, the distance between the two box centers divided by the oriented image diagonal determines movement. A value strictly above 0.02152951369355096 is moved. Equality and lower values are unchanged. E0 uses confidence 0.20, NMS IoU 0.35 and image size 640. [Application settings](../configs/app.json) · [Model and frame contract](REAL_MODEL_INTEGRATION.md)

B selected the movement threshold halfway between the largest untouched-object displacement and the smallest reportable movement in the pilot. The six pilot pairs yielded 34 matched objects: 31 untouched and three intentionally moved. They came from one capture session. For duplicate detections within a class, B keeps the highest confidence and breaks ties by box coordinates, then detection ID. A pair counts as changed when at least one object is added, removed or moved. [Threshold evidence](a_contracts_freeze.md) · [Filtering and tie rules](part_b_filtering.md)

The three user entry points share the comparison service. Synthetic fixtures exercise all four statuses. For the formal E0 validation, the four real pairs produced matching records across all three entry points and all 16 Linux browser downloads matched CLI JSON, CSV and PNG files byte for byte. Independent scoring then compared those records with ground truth. [Formal acceptance](REAL_MODEL_INTEGRATION.md#formal-v1-validation)

After C integrated A’s latest revisions, a live Windows Chrome session at `461db74` repeated the comparison on the four formal pairs. Each of the 16 correctly named downloads matched the same-pair CLI export byte for byte, and the 23 CLI records agreed with B’s batch output. [November 23 verification receipt](evidence/c_e0_entrypoints_20261123.json)

### 2. Robustness in real scenes

The loader rejects corrupt uploads and unequal oriented dimensions. Input changes clear stale results. A browser check after real inference verified that a corrupt upload disabled comparison and that replacing it with a valid photo restored the workflow. On formal pair G01P1 a pair of scissors was visible in both photographs, yet the detector's after-frame score was 0.1929, below 0.20. The application reported a false removal. The diagnostic box had IoU 0.9425 against the annotation. The example shows how a small score change can alter a scene-level decision even when localization is close. [E0 failure analysis](REAL_BATCH_EVALUATION.md#formal-v1-results) · [Integration verification](https://github.com/moli0516/COMP4423_G1_Project/pull/10)

### 3. Problems found and corrected

Ultralytics interprets NumPy prediction inputs as BGR. Sending the loader’s RGB array directly changed detections. The corrected detector converts at the backend boundary with a contiguous buffer. An independent PIL-input prediction reproduced all 43 retained E0 detections on the eight original validation photos. A accepted the fix in its canonical branch. A second defect let direct configuration construction or dataclass replacement associate changed effective settings with an old digest. C added a check that binds effective settings back to their source and regression tests for construction and replacement. [Detector acceptance](REAL_MODEL_INTEGRATION.md) · [A’s response](https://github.com/moli0516/COMP4423_G1_Project/pull/8#issuecomment-7075147335) · [Configuration regression](../tests/c_real/test_real_settings_provenance.py)

### 4. How GenAI assisted implementation

AI-assisted tooling contributed to implementation, regression tests, model runs and the verification of artifacts. The project history distinguishes agent-executed commands, teammate reviews and observed software behavior. For instance, the stale-digest review request led to checks for both configuration construction paths. In the private report the student should identify a specific prompt or request, the files it affected and a decision or verification they personally made. [C contribution evidence](C_CONTRIBUTION_EVIDENCE.md)

### 5. How GenAI addressed the task

One verifiable example concerns experiment inputs. Checking the archive hashes proved which photos had been released. On the first training scan, the pinned backend re-encoded all 28 training JPEGs while the eight validation images and 36 labels retained their original bytes. The preparation workflow now keeps a pristine source, prepares a separate working copy, and compares the effective inputs used by each training arm. A matching set of 72 file hashes supports the resolution comparison. [Input preparation and checks](C_E2_DECISION_BRIEF_20261118.md#set-up-the-formal-dataset-on-the-training-machine)

### 6. Limitations and improvements in the GenAI-assisted solution

An earlier AI-assisted status update asked A to review the channel conversion and output path a second time. A had already confirmed both on the other PR. Reading the two PR threads together corrected the handoff. Similarly, a green CI run on C’s earlier commit could only verify that particular branch. In a separate merge rehearsal, replacing the shared files with A's latest versions allowed a checkpoint containing only bottle and cup to pass C's pre-inference validation. The combined adapter keeps the six-class check and exercises it with a deliberately incomplete checkpoint map. Comparing both branches and testing the failure case caught a problem that a passing happy-path demo could miss. [A’s reply](https://github.com/moli0516/COMP4423_G1_Project/pull/8#issuecomment-7075147335) · [Application regression](../tests/c_real/test_yolo_real_pipeline.py)

Browser testing exposed a separate limitation of mock UI checks. On the earlier UI, four download buttons existed and their data were correct in AppTest. During consecutive downloads of full-resolution photos, a browser could lose the last PNG when another click triggered a Streamlit rerun. C changed the buttons to download the prepared files without rerunning the app and added a regression check. [Download controls](../app/streamlit_app.py) · [UI regression](../tests/c_release/test_release_ui.py)

Agreement among entry points can propagate the same detector error. Independent annotations and separate detector and event scoring exposed this in G01P1. Every validation box was large by the COCO original-pixel area convention. Eight photos, two groups, one training seed and one eligible independently photographed control limit the conclusions about small objects, new desks and false-alarm rate. Additional independent groups with small annotated targets and repeated seeds would allow those questions to be tested. [Validation scope and measurements](C_E2_RESOLUTION_PROTOCOL.md)

## Experimental results

The formal v1 export provided 28 training images. The validation subset contained eight images, 44 annotated boxes and 23 events. Across the four E0 pairs, the native detector validator measured mAP50 0.99500 and mAP50–95 0.87789 before B filtered detections. The validator used confidence 0.001 and NMS IoU 0.70. The application used its frozen thresholds above. At event IoU 0.50, four-status macro-F1 was 0.9103 and change-only macro-F1 was 0.8889. The single independently captured no-change pair produced zero alarms. [Formal E0 results and counts](REAL_BATCH_EVALUATION.md#formal-v1-results)

| Validation measurement | Fresh 640 control | Fresh 960 candidate |
| --- | ---: | ---: |
| Native detector mAP50 | 0.98140 | 0.99500 |
| Native detector mAP50–95 | 0.84079 | 0.88475 |
| Correct localized events on original pairs | 21/23 | 23/23 |
| Four-status macro-F1 | 0.86184 | 1.00000 |
| Peak PyTorch memory reserved | 426 MiB | 746 MiB |
| Median batch-one CUDA prediction | 126.46 ms | 126.54 ms |
| Completed epochs (selected epoch) | 46 (26) | 28 (8) |

Both resolution runs used the same GPU, YOLOv8n starting weights, effective inputs, seed 42, micro-batch 2, SGD and checkpoint-selection rule. The 640 model falsely reported an unchanged book as removed in G01P1 and G01P2. The 960 model classified both correctly. Book AP50–95 increased from 0.8283 to 0.9950 and scissors from 0.7846 to 0.8723, while bottle and keyboard each decreased. The sub-millisecond difference in median GPU latency is smaller than the observed timing variability. The application event check ran on CPU using each desktop-trained checkpoint. A separately trained CPU pair gave mAP50–95 of 0.82783 and 0.86614. [Full per-class scores, hashes and raw timing samples](evidence/c_e2_validation.json)

## Final test

The final group evaluation requires the selected checkpoint and settings, the complete held-out split and fixed movement rules. Record the two detector mAP measures and per-class results, precision/recall/F1 for every status, macro-F1, change-only results with the averaging rule, independent-control false alarms with their denominator, and representative successes and failures. The production configuration recorded in this evidence uses A’s E0 checkpoint. [Production configuration](../configs/app.json)

## Statement of Contribution

The public draft records the C workstream. For the required **Statement of Contribution**, the student must describe what they personally proposed, checked and learned, with a concrete example, and credit A, B and GenAI for their respective work. Names, IDs, the student's six personal responses and the final submission PDF belong in the private course submission.
