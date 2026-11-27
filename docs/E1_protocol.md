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

## 17. Evaluation/reporting implementation review — 2026-11-26

* `EvaluationRunner` 使用既有 `EventEvaluator`／`NoChangeEvaluator`，並輸出 `predictions.json`、`ground_truth.json`、`metrics.json`、`run_metadata.json`。若 comparison service 提供 before/after image detections，另保存 `image_detections.json`；其範圍是 B filter 後的應用診斷，不是 detector AP 輸入。事件指標現包含 aggregate 與 per-pair counts／precision／recall／F1。
* Evaluation manifest 參照的 before/after image bytes 會記錄 SHA256，runner 在推論前及後核對 bytes 未改變。由 `prepare_real_validation.py` 產生的 manifest 另保存原始 annotations 與 split-manifest hashes。Real batch provenance 另保存 checkpoint/config hash、split/release/manifest identity、共同應用設定、Python/package versions、source revision/dirty paths 及命令。Dirty paths 不是 patch；正式報告仍須保留 dirty source diff，或使用 frozen clean source。
* `scripts/compare_evaluations.py --control <completed-run> --treatment <completed-run> --output <new-directory>` 可比較已保存的 downstream runs，不載入模型。它要求 split、manifest、pair/path/image hashes、ground truth、evaluation settings、source/environment 及 no-change eligibility 一致；並要求不同 run ID 與 checkpoint bytes。它不比較 training augmentation policy（該項是實驗變因），輸出 aggregate、per-status、per-pair treatment-minus-control、`comparison.json` 及 `summary.md`。Validation 與 test 必須分別輸入；不同 split 會拒絕比較。
* No-change 輸出區分「false-positive change event 數」與「false-alarm pair rate」；後者明確定義為至少有一個 predicted change 的 eligible no-change pairs / eligible no-change pairs，不把事件數當作 rate。
* Event evaluator 同時回報 fixed-four-status `macro_f1` 及 change-only `macro_f1`，per-status zero-division 為 0；本 protocol 尚未選定哪一個作唯一 primary event F1。比較報告不替 owner 選擇，也不據此填造一個 primary 結果。
* Detector AP 尚無可直接用於 E1 的通用對照介面。`scripts/score_c_resolution_pair.py` 有 E2 resolution-specific `model.val()` mAP 路徑及 per-class AP，但綁定不同資料與 640/960 實驗；它不提供本 E1 的共同 operating-point precision/recall/F1、per-class TP/FP/FN 完整證據或可比 E1 報告。因此 detector 評估仍是 implementation gap；不得將該 E2 script 誤記為 E1 已完成。
* Inference failure 目前 fail-fast 並以 pair ID/cause 提示，避免將不完整 inventory 當成成功結果；不發布 partial output。Failure artifact/重跑關聯方式尚未凍結，需在正式 evaluation 前由 owner 決定。此處不改寫 failure policy。
* 本節記錄實作能力與缺口，不是 protocol approval、真模型 parity 或 E1 評估結果。全部驗證僅使用 synthetic fixtures/stub service；未讀取或 hash 真實 held-out inputs，未載入 checkpoint，亦未開始正式 evaluation。

## 18. Owner decisions 與 detector／failure-record 實作（2026-11-26）

* 以下四項為 owner 於 2026-11-26 凍結的決定，本節只記錄其實作與 synthetic 驗證，不代表實驗設計已被重新解讀：
  1. **Primary downstream metric**：change-only macro F1（added／removed／moved）。fixed-four-status macro F1、per-status 指標、no-change false-positive event count 及 pair-level false-alarm rate 一律標為 secondary，且 event count 與 pair rate 不合併。primary 名稱、值與 decision date 會寫入 `metrics.json`、`run_metadata.json` 及 comparison 輸出，comparator 會拒絕缺少或與已計分結果不一致的 primary 聲明。
  2. **Detector P/R/F1 與 per-class TP/FP/FN**：使用自訂 evaluator；matching 為 IoU 0.5、class 必須相同、依 confidence 由高至低 greedy one-to-one 配對；重複偵測已配對 GT 記為 FP、unmatched GT 記為 FN。evaluation operating point 取 explicit config 的 conf 0.20／NMS IoU 0.35，只記錄不重新套用、不調參。overall P/R/F1 由 pooled counts 計算（micro），per-class macro 另標為 secondary。
  3. **mAP**：mAP@0.5:0.95 及 mAP@0.5 使用 backend validator 自身 validation settings，不強制 conf 0.20／NMS IoU 0.35；實際使用設定與 backend version 分開記錄，無法觀察者記為 unobserved，不以預設值代替。mAP 與 application operating point 在輸出中為兩個獨立區段。
  4. **Failure records**：evaluation 失敗時寫入 `failure_record.json`，內容含 reason、exception type、pair/image context、input identities、source revision／dirty paths、evaluation config、command、timestamp 及 run id；failed run 不產生 success-shaped metric files，且不覆蓋既有輸出。rerun 可引用一個或多個較早 failure records 的 identifier 並附 free-text reason；link 在寫入新 run 前重新驗證（檔案存在、bytes 未變、id 相同、無重複），comparator 會列出 linked failures，但不自動接受 rerun 的科學有效性。
* 註（2026-11-26）：本節列出的 `src/deskcheck/evaluation/detector_evaluator.py` 與 `scripts/evaluate_detector_predictions.py` 已由 §19 取代，保留為歷史記錄。
* 實作：`src/deskcheck/evaluation/detector_evaluator.py`（自訂 detector P/R/F1、explicit config、saved-run 路徑）、`src/deskcheck/evaluation/map_evaluator.py`（backend mAP wrapper，注入 validator）、`src/deskcheck/evaluation/failure_record.py`（failure record 與 rerun link）、`src/deskcheck/evaluation/evaluation_runner.py`（failure record 觸發、run id、primary metric metadata）、`src/deskcheck/evaluation/run_comparator.py`（primary／secondary 標記、linked failures）、`scripts/evaluate_detector_predictions.py`。未新增 dependency。
* 既有 localized-event evaluator 沒有 micro 變體，因此輸出將 `micro_metric` 明確記為 unavailable，未新增 micro 規則；secondary 只包含 fixed-four-status macro F1、per-status 指標及 no-change 兩項結果。
* Zero-division：沿用既有 event evaluator 慣例，不新增規則。per-status denominator 為 0 時值為 0.0，兩組 macro F1 均除以固定 status 數，因此無 support 的 status 貢獻 0.0，change-only macro 在整個 split 都沒有 change support 時仍是已定義的 0（不是 unavailable）。Detector evaluator 同樣採 zero_division=0，另以 `unavailable_metrics` 區分「計算結果為 0」與「denominator 為 0、無法計算」。
* 自訂 detector matching 與 Ultralytics mAP 內部 matching 不同，因此兩者的 counts 不預期相等；mAP 不使用此自訂 evaluator 的 threshold。
* E2 的 640/960 resolution 比較維持原路徑（`scripts/score_c_resolution_pair.py`），未共用或改寫；E1 使用獨立的 mAP wrapper，避免改變 E2 既有 test 與 evidence 語義。
* 本節僅表示 synthetic-tested only（stub service、synthetic detections／annotations 與 stub validator）；formal training、real-backend verification、real-model evaluation、A 的 dataset provenance 修正、control comparability decision 及所有 owner approvals 仍為 open。本輪未執行 training、real inference、checkpoint loading、weight download 或 held-out content access，亦未產生任何 E1 結果或 approval。

## 19. Application-pipeline relabelling（2026-11-26）

* 獨立 review 指出：`image_detections.json` 保存的是 backend confidence／NMS、class mapping 及 B `DetectionFilter`（每影像每類最多一個）之後的 detections。以它計算的 precision／recall／F1 不是 detector metric；§18 的 "detector P/R/F1" 命名不成立。本節取代該命名，§18 保留為歷史記錄，不改寫。
* 現行 evaluator 位於 `src/deskcheck/evaluation/application_pipeline_evaluator.py`，輸出 `metric_name = application_pipeline_precision_recall_f1`、`input_scope = post_B_filter_one_per_class`、`scope_limitation`（one per class → recall non-increasing；precision／F1 可升可跌）及 `true_detector_metric = blocked`。CLI 為 `scripts/evaluate_application_pipeline.py`，輸出 `application_pipeline_metrics.json`。
* 舊 public 名稱（`detector_evaluator` module、`DetectorEvaluator`、`DetectorEvaluationConfig`、`evaluate_detector_predictions`、`evaluate_saved_detector_run`、`load_runner_detections`、`evaluate_detector_predictions.py`、`detector_metrics.json`）已由上述名稱取代。它們只存在於前一輪未提交變更，沒有已保存報告或外部 consumer，因此不提供 alias；對應關係即本段所列。
* Evaluator 會讀取並強制 `run_metadata.image_detection_evidence`：只接受 `post_B_filter per-image detections`；缺少、未知、宣稱 raw／pre-filter（目前不存在該 artifact）或與 `image_detections.json` 存在性矛盾者一律拒絕，並把 label 與來源寫入 `inputs.input_scope_source`。B runner 目前沒有把 comparison service 每個 result 內的 `prediction_scope` 持久化，因此 evaluator 只能驗證 run-level 的 evidence label，無法逐對核對 `prediction_scope`；此缺口列為 Task 2 設計項。
* Operating-point 檢查擴充為 run identity 綁定：要求 completed schema 1.0 `event-runner-v1`、存在 `run_id`、pair inventory／`pair_count` 與 saved detections 相符、provenance 的 checkpoint SHA-256 格式正確、operating point 等於 config。仍**未能**驗證：`run_metadata` 沒有 `image_detections.json` 的 digest、checkpoint bytes 未讀取、upstream filter／backend 未重跑；provenance 可被本機編輯，這不是 tamper proofing（已寫入報告 `binding` 區段）。Operating-point 比較為**精確 JSON 相等**，未加入 tolerance 或型別轉換：字串 `"0.2"` 不等於 `0.2`，相差一個 ulp 的數值亦不相等，兩者都會被拒絕；此行為由測試固定，日後若放寬必須是明確決定。
* Confidence 處理：evaluator 不重新套用 threshold；它驗證每個 saved detection 的 confidence 不低於記錄的 operating point，嚴格低於即拒絕。Backend 使用 `conf > 0.20`（strict）之行為來自 adapter 契約與獨立 review 的引用，本輪未重跑 backend 驗證；B `DetectionFilter` 使用 `>=`。正好 0.20 在此被接受（依 B filter 語義），此邊界已寫入報告 `confidence_boundary`。
* 結果表規定：§12 的 Detector mAP50–95／mAP50 行只能由 backend validator wrapper（`map_evaluator.py`）於真實執行時提供；Detector F1 @ common operating point 行**不得**以本 evaluator 填寫，且在 pre-filter artifact 經 A／B sign-off 前必須留空。本 evaluator 只支撐 application-pipeline 診斷。
* `docs/C_DETECTOR_PREDICTIONS_HANDOFF.md` 主張 post-B arrays 不是 AP 輸入；修正後實作與該文件一致，故未修改該 draft（其 A／B review 未完成）。pre-filter artifact 設計見 `docs/proposals/E1_detector_prefilter_design.md`。
* Owner（B）於 2026-11-26 就該設計作出 D1–D6 決定，proposal §10 為正式記錄：D1 = 先做 P1（adapter output、before `DetectionFilter`，不需 A 改動），P2 audit columns 只在 A 同意時以同一 schema version 加入；D2 = non-canonical predictions 不進 primary metric，另列 audit count（沒有 P2 時記為 unavailable，不記為 0）；D3 = 記錄 `>` 語義字串與 backend version、只驗 saved confidence、不 normalise 成 `>=`；**D4 = pre-filter evidence 放在獨立 evidence directory，由 `run_metadata.json` 以 relative path 與 digest 連結（與原 run-directory 建議不同）**；D5 = scope 需 A 與 B 批准；D6 = 兩個 metrics 為分開輸出，rerun 只有在有 explicit rerun link 時才取代 failed run。
* P1／P2 都只提供 detections，**不能產生 mAP**；mAP 只能來自 backend validation result。`src/deskcheck/evaluation/map_evaluator.py` 保持不變，且**目前未連接到任何 script**（僅以注入 validator 的 API 存在，尚未有 CLI 或 runner 呼叫）。
* Pre-filter artifact **尚未實作**。截至 2026-11-27，A 與 C 已在 PR #11 回覆：A 通過 P1，有條件通過 P2（schema 待 B 確認）；C 原則上接受 P1，條件是在拆分 PR 中確認最終共用 schema 與歸屬。實作尚未開始。Detector F1 行在該 artifact 經 A／B 批准前仍必須留空。
* 2026-11-28：B 決定 P1 與 P2 不在本次課程提交範圍內，不予實作；本設計僅作為紀錄保留。
* 仍 open 或 deferred（不在 D1–D6 內）：E0 control identity、trusted test manifest，以及 B 對 A 的 P2 schema 與 C 的 P1 條件的回覆。
* 本節所有變更僅 synthetic-tested；未執行 training、real inference、checkpoint loading、weight download 或 held-out content access。Owner approvals、real-backend verification、A 的 dataset provenance 及 control comparability 仍為 open。
