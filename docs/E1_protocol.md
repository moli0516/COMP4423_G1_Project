# E1 Protocol — Controlled Augmentation Ablation

## 1. 文件狀態

* Project：DeskCheck / COMP4423。
* Owner：B。
* 狀態：Protocol draft；尚未訓練，尚未產生 E1 結果。
* 本文件將已知專案狀態與建議實驗規則分開；所有 TBC 必須在正式訓練前確認。
* Proposed policy：關閉六項指定 augmentation。此方案尚待 owner 明確批准，不代表現有 config 已經正式凍結。
* 本文件不是 A 的修復驗收報告，也不宣告 PR #8 已 merge。

## 2. 研究問題與實驗邊界

在相同 dataset、split、模型初始化及訓練規則下，將六項指定 training augmentations 關閉，會如何影響 detector 表現，以及固定 B-side pipeline 的 change-analysis 結果？

E1 是整套 augmentation policy 的比較，不是六個 augmentation 各自的因果分析。不預設 E1 必須勝過 E0；無提升或退步亦需如實記錄。

本輪只比較 E0 與一個事前凍結的 E1 policy，不進行額外 augmentation grid search。E2 不屬本 protocol 範圍。

## 3. 開跑前已知狀態

### 3.1 Historical verification

以下為先前階段的驗證紀錄，僅識別當時檢查的 source、資料與 evidence 狀態；不代表現行 main 的執行結果，亦不構成 E1 正式訓練批准。

* Source integration retest：main `3b6d734be5aabc0591cdf278c3c48a1bf1075a3f` + A `f2d06813b6f2e92270f4651961a7b38869f0ad7c`；當時 15 focused tests、549 full-suite tests、pip check、Ruff 通過。這是當時的驗收組合，不表示該 main 已包含 A 改動。
* Epoch row-index fallback finding 當時已關閉；缺失／無效 epoch 的 acceptance cases 驗證 artifact 使用 `selected_epoch=-1`。此結果不證明既有 E0 checkpoint 的實際 selected epoch。
* Dataset inventory：24 pairs、48 source images；train／validation／test 為 14／4／6 pairs、28／8／12 images。
* 隔離 full export 從 dataset CWD 與 project CWD 均成功；fresh outputs 一致，共 269 objects，train／validation／test 分別為 157／44／68。
* 當時發現下載 export 的八張 validation images 與 source photos／fresh rebuild 不同，包括 decoded RGB pixels；原因及 E0 實際使用的 image bytes 尚待釐清。
* 當時 E0 evidence 的 inference snapshot 語義、threshold tie-break，以及 selected epoch 64 的 provenance 尚待補充。最新狀態見 §3.2；不以後續說明改寫當時的驗證範圍。
* Production adapter `45f0e0a` 不包含在上述 source integration retest；其後的 adapter／C integration 驗證另見 §3.2。

### 3.2 Current implementation baseline — 2026-11-24

本節記錄 E1 code implementation 的起點及截至本次更新的驗證狀態。各項測試結果只適用於其明確記錄的 source revision，不跨 revision 宣稱通過。

#### Source 與整合狀態

* C PR #10 已完成 review、修正驗證、Approve 及 merge。
* Merge 後 main integration baseline：
  ```
  b4bfdc48e00f294180c6df40b23f28d52bf90fa7
  ```
* E1 implementation 以此 SHA 建立新的工作分支／worktree；既有 `experiment/E1` checkout、未追蹤 protocol、private scripts 及 review worktrees 均須保留，不直接 reset、覆蓋或清理。
* C 已驗證 candidate：
  ```
  f9d370b6ffbe55d83a00d01daf554e5622051afb
  ```
* 該 candidate 的 native macOS verification：
  * Committed targeted regressions：24 passed。
  * Full dev suite：582 passed、1 optional-vision skip、0 failed。
  * Repository-wide Ruff 及 `pip check` 通過。
  * Supplementary probes 驗證 malformed class-ID rejection、合法 integral class mapping、BGR buffer ownership／caller non-mutation，以及 cache failure／retry。
* Focused suite 曾再現 intermittent Streamlit／Arrow failure；full suite 通過，未識別出可歸因於該三檔修正 delta 的 regression。此項保留為獨立 non-blocking follow-up，不記為所有 suites 均全綠。
* 上述 execution results 屬於 `f9d370b…`，不是 merge 後 `b4bfdc4…` 的獨立重跑紀錄。
* 既有 browser acceptance receipt 保留實際 execution revision `461db74a13319ea4156e889db278adb50f54d45f`；本輪沒有重新執行 browser acceptance，亦沒有將 receipt 改寫成較新 SHA 的執行證據。

#### A source follow-ups

* A follow-up source 已在獨立 detached worktree 核對：
  ```
  a39ee97411b4f576c1b09099967b2922e0ad2109
  ```
* 前一個 A target `44e2322aaebd9302f6792c30501fc8a712c184fe` 是該 target 的 ancestor；本輪只 review 其後三個 commits 的 delta。
* Committed asymmetric-channel BGR regression 獨立通過；額外 probe 驗證 backend input 為 C-contiguous、與 caller data 不共用 memory，且不修改 caller array。
* Partial-cache production fix 已由 source inspection 及 stub probe 驗證：invalid class map 不發布 cache、不執行 prediction；其後 valid retry 成功並可重用 cache。
* Committed cache regression 在 dev-only environment 因直接 patch `ultralytics.YOLO` 而於測試 setup 階段失敗，尚未執行到 production loader。此為 test dependency-isolation follow-up，不代表 production cache fix 未通過驗證。
* 該 A target 的完整 dev-suite 結果為 401 passed、1 skipped、1 failed；Ruff 及 `pip check` 通過。不得記為已獨立重現 A 所稱的 403 passes。
* 此 A target 的驗證與 C／main integration 分開記錄；不得假設 A branch 的所有後續修改已包含於 `b4bfdc4…`。

#### Dataset 與 E0 package 狀態

* 已交付 `best.pt` 的 SHA256 與 published value 及 `artifact.json` 一致：
  ```
  82ed9f8569b705e7074f44d81006515da3e02dde3284a211c4a73a5f9d96c438
  ```
* Checkpoint 僅完成 byte-level hash verification；本輪未 deserialize checkpoint 或執行 real inference。
* Pinned catalog／validator 驗證通過：24 pairs、無 semantic errors、無 capture-group／shared-image split leakage，各 split 均覆蓋六類及四種 statuses，relative annotation paths 均可解析。
* 使用 movement threshold `0.02152951369355096` 的 direct validation 通過，無 warnings；此結果不等同 final downstream evaluation protocol 已完成批准。
* Frozen application inference `conf=0.20`／NMS IoU `0.35`，與 training-time snapshot `conf=0.25`／IoU `0.45` 已在交付 metadata 中分開標示。
* Threshold CSV 已核對為 15 組 unique combinations、六組 tied optimum；所述 ascending tie-break 選出 `.20/.35`。Evidence placeholder，以及 script exact ranking 與文檔 tolerance 的差異仍待收尾。
* `results.csv` 有 84 data rows，explicit epochs 為 1–84；evidence 的 zero-indexed 說明與之不一致。既有 checkpoint 的實際 selected epoch 仍保留 provenance 限制，不由 CSV 首個最大值推定。
* `yolo_export_at_E0_time/` 尚未在本地下載內容中找到；其在 published release archive 中的位置／完整性尚待確認。
* 目前 export 的 train 28／28、test 12／12 image bytes 與 source 一致；validation 0／8 一致。差異來源及 E0 實際使用的 validation bytes 尚未完整建立。
* 兩個原始 release ZIP 未保留於本地下載內容，因此 published archive SHA256 尚未獨立核驗；不得以 extracted-directory hashing 代替 archive hash verification。

#### E1 execution boundary

* 目前允許開始 protocol／config reconciliation、preflight infrastructure、shared trainer integration，以及 stub／dev tests。
* Six-key policy 仍為 proposed；正式 training 前須由 owner 明確批准。
* Historical E0 是否可作公平對照，或是否採用新 matched control，尚未決定；不得默認既有 E0 與新 E1 僅有 augmentation 差異。
* 正式 training 前仍須確認 initialization identity／hash、effective train／validation bytes、完整有效設定及 execution environment。
* 正式 downstream evaluation 前仍須統一並凍結 B-side movement、event matching、counting、zero-division 及 operating-point 規則。
* 本節不構成 E1 training approval、merge 後 main 的 real-model acceptance，或 held-out evaluation 完成聲明。

* Source integration retest：main `3b6d734be5aabc0591cdf278c3c48a1bf1075a3f` + A `f2d06813b6f2e92270f4651961a7b38869f0ad7c`；15 focused tests、549 full-suite tests、pip check、Ruff 通過。這是驗收組合，不代表 main 已包含 A 改動。
* Epoch row-index fallback finding 已關閉；缺失／無效 epoch 的 acceptance cases 驗證 artifact `selected_epoch=-1`。
* 新 dataset inventory：24 pairs、48 source images；train/val/test 為 14/4/6 pairs、28/8/12 images。
* 隔離 full export 從 dataset CWD 與 project CWD 均成功；fresh outputs 一致，共 269 objects（157/44/68）。
* 新下載 export 的 8 張 validation images 與 source photos／fresh rebuild 不同，包括 decoded RGB pixels；原因及 E0 實際用圖待 A 說明。
* E0 evidence 仍有 embedded inference config 衝突、tied threshold choice 說明、selected epoch 64 的 checkpoint provenance 待釐清。
* Production adapter `45f0e0a` 不包含在上述 source retest，尚未完成獨立驗收。

以上狀態不構成正式 E1 training approval。

## 4. E0 身份與待確認基準


| 欄位                                  | 目前記錄                                                           | 正式訓練前要求                                                   |
| ------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| E0 checkpoint SHA256                  | `82ed9f8569b705e7074f44d81006515da3e02dde3284a211c4a73a5f9d96c438` | 保留；僅識別產物，不單獨證明訓練歷史                             |
| Completed epochs                      | 84；results.csv 有 explicit epochs 1–84                           | 保存 CSV 與 config 證據                                          |
| Selected epoch                        | 宣告 64；CSV 64、65 的 recorded mAP50–95 並列最高                 | 取得 backend 證據或明確保留 unknown/unverified 限制              |
| Frozen application conf / NMS IoU     | A 宣告 0.20 / 0.35                                                 | 釐清 embedded config 0.25 / 0.45 的語義並識別 release settings   |
| Threshold selection                   | CSV 支持 0.20 / 0.35 為 tied optimum                               | 記錄實際如何在並列候選中選定                                     |
| E0 training/validation image identity | TBC                                                                | 釐清 actual run dataset 與 source/fresh/downloaded export 的對應 |
| Model architecture                    | TBC；核對 E0 effective config                                      | 不以先前建議代替實際設定                                         |
| Initial pretrained weights + hash     | TBC                                                                | 找出 E0 真正初始化來源                                           |
| Backend / version / device            | TBC                                                                | 保存實際環境與限制                                               |

若無法重建公平對照，先修改 protocol 並記錄限制，再批准執行；不得默默把有差異的比較標成純 augmentation ablation。

## 5. 初始化規則（建議正式採用）

E1 從與 E0 相同的 pretrained initialization 開始，不由已訓練的 E0 `best.pt` fine-tune，也不 resume E0 optimizer/scheduler 狀態。

「Start from E0」在本 protocol 僅指繼承 E0 的實驗設定，不指繼承 E0 訓練後 checkpoint。若 owner 決定改做 fine-tuning，必須另立研究問題／實驗身份，不沿用本比較的純 augmentation 解讀。

## 6. Proposed augmentation policy


| Key    | E0 effective value | E1 proposed value |
| ------ | ------------------ | ----------------- |
| hsv\_h | TBC                | 0.0               |
| hsv\_s | TBC                | 0.0               |
| hsv\_v | TBC                | 0.0               |
| fliplr | TBC                | 0.0               |
| mosaic | TBC                | 0.0               |
| scale  | TBC                | 0.0               |

* 名稱：six-key augmentation ablation；不稱為「完全無 augmentation」。
* 其他 augmentation keys（包括 translate、degrees、shear、perspective、flipud、mixup、copy\_paste 及所用版本其他相關設定）需保存 effective values 並保持一致。
* 若某 key 在 E0 已為 0，記錄為 no-op，不計作實際改變。
* 保存 E0 → E1 semantic config diff；只容許上述 policy 及必要 run identity/output-path metadata 改變。其他差異逐項解釋並事前批准。
* 不在 validation/test 上施加本實驗的隨機 training augmentation。

## 7. 固定訓練條件

以下項目在 effective config 中逐項記錄，數值以 E0 證據為準，不使用未核實的舊建議：

* Architecture、初始化 checkpoint 及 SHA256。
* Image size、batch size、optimizer、learning rate／schedule、weight decay。
* Epoch budget、early-stopping patience、checkpoint selection 規則。
* Seed、deterministic 設定、workers，以及適用的 backend options。
* Dataset version、split manifest、class mapping。
* Python、Ultralytics、torch 版本、device／硬件。

使用相同 epoch budget／early-stopping 規則，不強制兩個 run 有相同實際停止 epoch。記錄實際 completed epochs。

最低執行範圍為一個已識別 E0 run 與一個 E1 run；若只有單次訓練，結果屬 single-run descriptive evidence，不作統計顯著或穩定性宣稱。若要加 seeds，必須先修訂 protocol，為 E0/E1 配對執行，而非只補跑表現較差的一方。

## 8. Dataset 與資料身份

* Train：14 pairs / 28 images。
* Validation：4 pairs / 8 images。
* Test：6 pairs / 12 images。
* 固定 source photos、annotations、split membership 及六類 class mapping。
* 已知 mapping：0 bottle、1 cup、2 keyboard、3 mouse、4 book、5 scissors；執行前再與 source/config 核對。
* Movement threshold：`0.02152951369355096`，保持不變。

為 source photos、pairs.jsonl、splits.json、exported images/labels 保存 hash manifest。YAML root 可以隨本機路徑改變，但內容身份必須可追溯。

不得把 downloaded validation images 悄悄替換成 fresh images，再將既有 E0 metrics 宣稱為同一資料上的比較。先由 A 釐清 actual run identity；如不能解決，記錄限制或另建經批准的公平 baseline。

## 9. 執行前 smoke 與正式 run

Backend-free preflight supports:

```text
.venv/bin/python scripts/preflight_e1.py --control <control.json> --treatment <treatment.json> --baseline <baseline.json> --initialization <local-file> --dataset-yaml <dataset.yaml> --annotations <pairs.jsonl> --splits <splits.json> --run-dir <new-absent-directory> --output <new-receipt-directory>
```

Approval record flags, when appropriate, are `--owner-approved`,
`--six-key-policy-approved`, `--control-comparability-approved`,
`--effective-settings-reviewed`, `--backend-defaults-verified` and
`--downstream-rules-frozen`. They record local assertions only; they are not
signatures or independent evidence. Missing inputs or approvals yield a
non-authorizing receipt. The receipt directory is distinct from the frozen
training run directory.

The only execution interface is the saved-specification form below. It reads
the persisted preflight spec; it does not accept replacement config,
initialization, dataset or run-directory arguments:

```text
.venv/bin/python scripts/preflight_e1.py --execute --specification <receipt-directory>/execution_specification.json --approval-token APPROVED_E1_TRAINING
```

**NOT EXECUTED.** A passing preflight is not formal training approval, and
this draft protocol still has unresolved scientific gates. No training smoke
or formal command is authorized by this document.

## 10. 評估規則

## 10.1 Detector

Primary detector metric：mAP@0.5:0.95。
Secondary：mAP@0.5、共同 application operating point 下的 precision/recall/F1，以及 evaluator 支援的 per-class TP/FP/FN。

mAP validation protocol 與 application inference operating point 分開記錄；不可默認 mAP 必須使用 conf=0.20。兩個模型用相同的 validation 設定、版本及計算規則。

## 10.2 Downstream

Primary downstream metric：現有 evaluator 定義的 event F1；正式凍結前確認 micro/macro、event matching 及 zero-division 規則。
Secondary：added/removed/moved 分項 counts、precision/recall/F1，以及 no-change pairs 的 false-positive event counts（如 evaluator 支援）。

固定 production adapter、canonical ImageFrame 路徑、class mapping、B-side filter/matcher/change analysis、movement threshold 與 event evaluation rules。

主比較建議共用 conf=0.20、NMS IoU=0.35，待 E0 release config 語義確認後凍結。E1 不另行挑選有利 threshold。各模型獨立 threshold tuning 如需進行，另列事前批准的次要分析，不與主比較混合。

Detector 與 downstream 是兩個不同層面的結果；一方提升不保證另一方提升，分開報告。

## 10.3 Validation 與 held-out test

* Validation 用於事前允許的 checkpoint/model-selection 決策及 pipeline smoke。
* 在查看 test prediction results 前凍結 policy、metrics、threshold rules、失敗處理及 evaluation implementation。
* 不用 test 結果選 augmentation、threshold、matching parameters 或額外 seeds。
* E0/E1 都準備好且資料身份已確認後，按同一 frozen protocol 做 held-out comparison。
* 若系統缺陷迫使重跑，保留原結果、修復原因、版本與重跑紀錄；不得抹去不利結果。

## 11. Run evidence 最小交付

* Approved protocol revision / commit。
* Training/evaluation source SHA；若 dirty，保存 diff 並解釋，正式 run 優先使用可識別的 frozen source。
* E0/E1 effective configs 與 semantic diff。
* Initial pretrained checkpoint identity / hash。
* Dataset/split/image/label hash manifest。
* Backend、Python、packages、hardware、seed、命令及 run directory。
* best.pt SHA256、results.csv、完整 logs、artifact metadata。
* Completed epochs、selected epoch、index convention、selection evidence；無證據不得虛構。
* Evaluation commands/configs、per-image detections、event outputs、aggregate metrics（依現有支援）。
* Failure examples、限制及 protocol deviations。

## 12. 結果表（尚無結果）

Validation 與 test 分別填表，不混合。


| Metric                               | E0 | E1 | E1 − E0 |
| ------------------------------------ | -- | -- | -------- |
| Detector mAP50–95                   | — | — | —       |
| Detector mAP50                       | — | — | —       |
| Detector F1 @ common operating point | — | — | —       |
| Downstream event F1                  | — | — | —       |
| Added TP / FP / FN                   | — | — | —       |
| Removed TP / FP / FN                 | — | — | —       |
| Moved TP / FP / FN                   | — | — | —       |
| No-change false-positive events      | — | — | —       |

另附每個 pair 的結果及錯誤類型；不把單次小資料集上的差異表述為一般化保證。不為追求正向結果而事後改 primary metric。

## 13. 開跑 gate 與完成定義

## Formal training 前

* Owner 批准研究問題與 six-key policy。
* E0 effective values、初始化及訓練規則已核實。
* E0 actual validation image identity 已釐清；或明確批准替代 baseline/protocol。
* Dataset manifests、hashes、class mapping 已凍結。
* Source/CLI/environment/command/run directory 已記錄。
* Config diff 只有批准差異。
* Checkpoint selection 規則與 provenance 表示方式已確認。

## 正式評估前

* E0/E1 checkpoint 身份及 evidence 可追溯。
* E0 release operating-point 語義已釐清。
* Production adapter 真模型 smoke 與所需應用路徑 parity 通過。
* Metrics、event counting、thresholds、test protocol 已凍結。

## E1 完成

* Training/evaluation evidence 保存完整。
* E0/E1 使用一致 protocol，或清楚記錄偏差與限制。
* 結果表、raw counts、failure analysis 完成。
* 不論提升／持平／退步均如實報告。
* 文件未將 single-run evidence、未驗證 epoch 或未驗收 package 寫成已證實結論。

## 14. 本機 E1 review 修正狀態（2026-11-25）

* 本機獨立 source review 在 E1 初始 dirty snapshot 發現五項 blocking issues：初始化身份未綁定 preflight、held-out split membership 可由 symlink 繞過、preflight 與 shared trainer 接受的資料 layout 不一致、run directory 未綁定 plan，以及 readiness/approval 欄位語義不一致。修正已實作並由合成測試覆蓋；這不是另一位 reviewer 對修正的 acceptance。
* 執行規格保存於 receipt 目錄的 `execution_specification.json`。規格 hash 綁定 control/treatment/baseline bytes 與 resolved settings、初始化絕對路徑及 SHA256、dataset YAML／catalog identities 與 train/val inventory、source-file hashes、執行環境、approval records 及全新的絕對 run directory。執行重新驗證這些輸入並拒絕改變或搬移初始化；沒有簽章，spec hash 只識別內容，不證明簽署者或 approval 真實性。
* Preflight 與 shared trainer 僅接受明確列出的 `train: images/train` 及 `val: images/val`，並要求固定的 `images/train`, `images/val`, `labels/train`, `labels/val` 目錄。缺少任一 split key、custom image targets 或 `train_labels`／`val_labels` 均在 backend import 前拒絕；不會由 missing keys 推導預設。使用者提供的 YAML 若需補上明確 targets，須另行核對及批准；本工作不改 delivered YAML 或修復 export。它使用既有 `pairs.jsonl` + `splits.json` catalog，按 exporter 的 `<pair_id>__before/after` stems 將每個 train/val member 對應到 declared split；test membership、跨 split aliases、train/val aliases、未知成員及不支援 layout 均拒絕。YAML test target（若有）不會被列舉或讀取。
* Preflight 先核對每張 train/val image 的 authoritative membership，再 hash train/val image/label bytes。Held-out image/label bytes 不被開啟、解碼或 hash。若缺少可信 metadata，不能推斷 test membership，也不能建立 ready inventory；目前沒有跨 held-out bytes 的 duplicate-content guarantee。
* Readiness 定義：`technical_ready` 表示 config、初始化、train/val+split identities、source/environment 與新 run path 齊備；其中 Ultralytics 與 Torch package metadata 不可缺少。`formal_training_approved` 表示 technical readiness 且全部六項 owner/prerequisite records（owner/scientific、six-key、control comparability、effective settings review、backend defaults verified、downstream rules frozen）均記錄為 true；它描述已記錄的 gate，不證明這些聲明真實或構成獨立科學核准。Preflight 的 `execution_intent` 和 `execution_permission` 一律 false。執行 token 只提供當次 explicit intent；只有 saved formal gate、current input match 及 exact token 全部通過，才會呼叫 shared trainer。這些 CLI records 是明示聲明，不是數位簽章或獨立科學證據。
* 執行拒絕已存在的 run path（含 regular file、directory、symlink）；不 resume、不覆蓋。Receipt output 與 run output 分離。Shared trainer 保留 optional initialization defaults；E1 hash mismatch 在 lazy backend import/model construction 前拒絕，並將實際 initialization path/hash 寫入 resolved config。
* Backend-free environment report 只記 Python/platform 與已安裝套件 metadata；不 import 或初始化 Ultralytics/Torch。後端預設值及實際 runtime settings 未由這次修正驗證。即使 approval record 被填寫，也不能把未觀察設定描述成已觀察事實。
* Preflight 支援的命令與 flags 見 §9。`--execute --specification ... --approval-token APPROVED_E1_TRAINING` 為唯一執行形式，**NOT EXECUTED**。本次未訓練、未執行 real inference、未載入 checkpoint 或下載 weights；未發生 real-model acceptance。此前 receipt 不代表此修正後 source snapshot。
* 五項修正的 regression coverage：凍結初始化 A/B、same-path byte mutation、missing initialization 及 trainer model-construction boundary 見 `test_preflight_initialization_a_rejects_changed_bytes_and_relocation`、`test_missing_initialization_cannot_authorize_execution`、`test_trainer_rejects_hash_mismatch_before_backend_or_model`、`test_trainer_rechecks_initialization_immediately_before_model_construction`；authoritative split、test symlink rejection-before-hash、train/val aliases、wrong-split directories 及 shared-trainer-supported layout 見 `test_inventory_requires_catalog_and_maps_disjoint_splits`、`test_symlink_to_test_is_rejected_before_hashing_test_content`、`test_train_val_aliasing_and_wrong_split_directories_rejected`、`test_unsupported_layout_rejected_and_shared_trainer_accepts_supported_layout`；frozen output collision/nesting 見 `test_execution_rejects_frozen_output_collisions`、`test_execution_and_receipt_directories_must_not_nest`；readiness/approval/CLI gates 見 `test_readiness_truth_table_and_spec_identity`、`test_execution_requires_each_approval_record`、`test_cli_approval_flags_cannot_authorize_incomplete_preflight`、`test_execute_cli_rejects_preflight_overrides`。These tests use synthetic fixtures; they are not reruns of the independent review probes.
* Final backend-free verification on the corrected working tree: `.venv/bin/python -m pytest` — 636 passed, 1 skipped; `.venv/bin/python -m pytest -q tests/unit/test_e1_experiment.py tests/unit/test_trainer_provenance.py tests/unit/test_training_config.py` — 82 passed; focused collection — 82 tests; changed-file Ruff and `py_compile` passed; `.venv/bin/python -m pip check` passed. Repository-wide `.venv/bin/ruff check .` reports two findings confined to the unchanged `scripts/private/measure_pilot_tau.py` (non-executable shebang and import ordering); they are not modified here. The exact standalone independent-probe/log directory was not available; the supplied five-finding review summary and preserved source-review export were retained, and no placeholder label was treated as a path.
* 本次沒有真實 owner/scientific approval。正式 training remains unapproved; no real-model acceptance occurred. An incomplete receipt generated after these edits records the current source identity; no earlier receipt is evidence for this corrected snapshot.

## 15. 第二次本機 E1 re-review 的三項修正（2026-11-25）

* 最新 independent re-review 的報告文件未出現在本次提供的 evidence bundle；實際 synthetic probe 與重跑 log 可在 `/tmp/deskcheck-e1-independent-review.eg2t5h/independent_synthetic_probes.py`、`independent-synthetic-probes-rerun2.log` 及 `verification-exits.txt` 查閱。該 probe 重現了 C：省略 train/val 被兩個本機 validator 接受、custom label target 僅由 preflight 拒絕；D：run ancestor 改為 symlink 後 execution 可被導向 receipt；及 receipt audit：malformed version、receipt embedded spec 不一致仍可進 trainer。此修正不宣稱已獲獨立 reviewer acceptance。
* C 修正：E1 preflight 與 shared trainer 共用嚴格 scalar layout validator；明確要求 `train: images/train`、`val: images/val`，並一致拒絕缺少任一 key、custom train/val targets 及 `train_labels`／`val_labels`。authoritative split-catalog membership 和不讀取 held-out content 的既有檢查保留。合成回歸見 `test_dataset_layout_failures_are_consistent_and_pre_backend` 及 `test_unsupported_layout_rejected_and_shared_trainer_accepts_supported_layout`。實際 dataset YAML 未修改；若日後輸入沒有明確 targets，這是執行前獨立的資料準備 prerequisite。
* D 修正：trainer 呼叫前重新解析 frozen canonical run path，要求仍等於 specification path、拒絕現存 symlink path components、重新檢查當前 receipt/run 雙向 nesting，並拒絕已存在的 output。回歸見 `test_execution_rejects_redirected_run_ancestor_before_trainer`、`test_execution_rechecks_receipt_run_nesting_before_trainer` 及 `test_execution_rejects_frozen_output_collisions`；stub 驗證拒絕時 trainer 未被呼叫。未採用預留目錄，避免與 shared trainer 的既有 `run_dir` 建立契約不相容；最後檢查與 shared trainer 建立目錄之間仍有窄小 TOCTOU window，本修正不是 adversarial filesystem race-proofing。
* Receipt/spec audit 修正：只接受 receipt version 3 及 spec schema 1；檢查必要欄位與精確型別、重複 JSON keys、canonical spec hash、receipt 目錄 linkage、receipt embedded spec 與獨立 spec 相等，並核對 receipt provenance fields。回歸見 `test_execution_rejects_malformed_or_unsupported_receipt_version`、`test_execution_rejects_malformed_specification_schema`、`test_execution_rejects_receipt_embedded_spec_disagreement`、`test_execution_rejects_missing_receipt_field_and_incorrect_field_type` 與 valid linked stub case `test_execution_uses_frozen_inputs_and_forwards_same_initialization`。Hash 只檢查本地內容一致；receipt/spec 可被本機編輯，不是簽章，也不獨立證明 owner approval。
* 本次只使用 synthetic dataset／receipt fixtures 及 trainer stubs；保留先前 A/B/E 保護與回歸，不執行 training、real inference、checkpoint loading、dependency installation 或 held-out content access。正式 training 仍需 owner/scientific approvals、明確資料 YAML targets、可信 catalog/split、初始化與資料身份、observed backend/runtime settings、control comparability 及凍結 downstream rules。未發生 real-model acceptance；execution command 仍為 **NOT EXECUTED**。

### Remaining formal-training prerequisites

* Owner approval of the exact scientific protocol and six-key policy.
* A saved, complete specification whose local initialization path and bytes remain unchanged through the shared-trainer boundary.
* Trusted annotations and split manifest for the same export; complete train/val pairing and exact content identity. The current workspace has no real catalog files, so only synthetic membership fixtures were used.
* Effective Ultralytics/backend defaults and runtime-observed settings remain unknown in this backend-free environment and must be established separately.
* A matched or explicitly approved control/E0 comparability decision.
* Frozen downstream event matching, counting, threshold and zero-division rules. Test-content duplicate detection remains outside this inventory's guarantee.

## 16. 依據

本草稿的 historical status sections 保留 owner 提供的 DeskCheck 分工、E1 討論、A 的交付聲明及當時 source/package/rebuild retest 範圍，不把它們改寫為當前驗收。§14 另外記錄 2026-11-25 的本機 E1 source review/fix scope。未執行 E1 訓練或 real-model acceptance；所有 proposed 與 TBC 仍須在 formal run 前確認。
