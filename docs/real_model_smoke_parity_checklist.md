# DeskCheck 真模型 Smoke / Parity Checklist

## 1. 狀態及目的

* 狀態：執行規格草稿；尚未執行真模型 smoke 或 parity。
* 依據：owner 提供的專案狀態及需求；本文件未獨立檢查最新 repo。
* 目的：分別確認真 checkpoint 可執行、各實際支援路徑的 detector 輸出一致、以及 downstream event 輸出一致。
* 不屬本輪：訓練、threshold tuning、held-out evaluation、模型準確率驗收。
* 所有 proposed tolerance 與 TBC 項目必須在比較輸出前凍結；不得看到差異後放寬以求通過。

## 2. 執行 Gate

## Code-discovery 階段可先做

* 唯讀確認 adapter 及 entry points，不載入 checkpoint。
* 記錄 source SHA、工作目錄狀態及依賴版本。
* 確認各路徑是真模型、fake backend、或根本未提供。
* 定義 smoke fixture、config、輸出及比較規格。

## 真模型執行前

* 使用獨立、可識別的 candidate source；不直接改 main 或既有驗收現場。
* Adapter source review 通過；相關必要改動已包含在 candidate。
* Checkpoint SHA256 符合第 3 節。
* Checkpoint 來源可信且已核准載入。
* 固定非 held-out fixture，確認 train membership 及 image hashes。
* Config precedence 已確認，所有路徑可證明 effective values 一致。
* 環境、device、dtype、preprocessing、postprocessing 固定。
* 確定真實可用路徑及所有必要 CLI/API 參數，不使用猜測命令。
* 確定輸出位置並避免覆蓋既有 evidence。
* 比較規則、tolerances 及 required routes 已批准。

E0 training/validation provenance 未釐清，不妨礙單純 code discovery。若仍進行功能 smoke，需明確批准其範圍：只能證明指定 bytes 在指定設定下的執行與 parity，不能證明 E0 歷史訓練／評估聲明。

## 3. Checkpoint 與 Source 身份


| 欄位                       | 值                                                                        |
| -------------------------- | ------------------------------------------------------------------------- |
| Checkpoint                 | E0 best.pt；實際 absolute path：TBC                                       |
| Expected SHA256            | `82ed9f8569b705e7074f44d81006515da3e02dde3284a211c4a73a5f9d96c438`        |
| Candidate adapter revision | `45f0e0a292d906bfa97bb1b837ea3103f77ee060`，待確認 ancestry／完整整合版本 |
| Actual tested source SHA   | TBC                                                                       |
| Integration base           | TBC；不假設舊 main 已包含 adapter                                         |
| Dirty diff / merge parents | TBC；如非 clean commit 必須保存 patch、parents 及檔案身份                 |

Hash 一致只識別 checkpoint bytes，不證明其 selected epoch 或資料歷史。缺 best.pt 不得改用 last.pt；hash 不符則停止。

## 4. Smoke Fixture

## 正向 fixture

從已固定 manifest 的 train split 選至少一個完整 before/after pair，使用 source photos，不使用 held-out test，也不使用目前身份有爭議的 downloaded validation export。


| 欄位                                    | 執行前填寫                                         |
| --------------------------------------- | -------------------------------------------------- |
| Manifest path + SHA256                  | TBC                                                |
| Annotation path + SHA256                | TBC                                                |
| Pair ID / capture group                 | TBC                                                |
| Confirmed split                         | train                                              |
| Before image path / SHA256 / dimensions | TBC                                                |
| After image path / SHA256 / dimensions  | TBC                                                |
| Selection reason                        | 操作方便／資料完整；不以測試後的 parity 結果挑樣本 |

如一個 pair 不足以覆蓋尺寸／方向或 downstream 非空 events，可事前再指定第二個 train pair。不得為求 pass 反覆更換 fixture。

## 邊界情況（獨立記錄）

* 缺 checkpoint 或 hash 不符：應停止，不 fallback。
* 不存在／無法解碼的 input：應有可識別失敗，不偽裝成成功的空結果。
* Zero detections：如有預先識別的非 held-out fixture，可測合法空結果；不得以任意提高 threshold 代替主 parity 設定。
* 未覆蓋情況標為 NOT TESTED，不能當作 PASS。

## 5. Code Discovery：先確認實際支援路徑

逐項記錄檔案、symbol、呼叫鏈、模型建立位置與 config 注入位置。以下是待查類別，不代表 repo 已提供。


| 路徑類別                  | 狀態 | Entry point / symbol | 真 detector 建立位置 | Config precedence | Raw output 可取得？ | 納入 required？ |
| ------------------------- | ---- | -------------------- | -------------------- | ----------------- | ------------------- | --------------- |
| Direct production adapter | TBC  | TBC                  | TBC                  | TBC               | TBC                 | 建議 yes        |
| Service / runner          | TBC  | TBC                  | TBC                  | TBC               | TBC                 | TBC             |
| CLI                       | TBC  | TBC                  | TBC                  | TBC               | TBC                 | TBC             |
| UI                        | TBC  | TBC                  | TBC                  | TBC               | TBC                 | TBC             |

允許狀態：SUPPORTED-REAL / FAKE-ONLY / NOT-IMPLEMENTED / BLOCKED / TBC。

* 不新增一條臨時 production 路徑來聲稱現有介面已支援。
* 路徑不支援屬 N/A 或 feature gap，依既定交付契約判斷；不是 parity PASS。
* 可使用獨立 audit harness 收集既有 API 的 raw outputs，但必須保存 harness，且不能繞過待驗介面的 preprocessing、config 或 downstream。
* UI 如只顯示四捨五入結果，僅能驗 display／functional 行為；需取得 backing raw output 才能宣稱 numeric parity。
* 若未有任何可比較的第二條真模型路徑，只能報 direct smoke，不能宣稱跨介面 parity。

## 6. Config 與環境凍結

## Config

建議本輪顯式指定共同 application operating point：

* confidence\_threshold：0.20。
* nms\_iou\_threshold：0.35。
* movement threshold：0.02152951369355096。

以上為本輪 proposed execution settings；不宣稱已解決 E0 embedded config 歷史衝突。若路徑依賴相互衝突的 metadata，須先明確選擇來源／override，不能猜測。

每條路徑保存：config file path/hash、CLI/API overrides、effective config，以及 precedence。無法證明有效值一致時標 BLOCKED，不進行純 detector parity 歸因。

其他需固定／查證項目：image size、class filters、max detections、agnostic NMS、augmentation/TTA、letterbox/resize、EXIF orientation、RGB/BGR、dtype/range、bbox convention、B-side filters/matcher 及任何其他結果相關設定。

## 環境

保存 OS、Python、Ultralytics、torch、相關影像依賴、device、dtype、硬件、seed/deterministic options（如適用）。

主 parity 使用相同環境／device／dtype；跨 CPU/GPU 或不同 backend 比較另列擴展測試，不混入主結論。

## 7. 執行順序

1. 保存 source、fixture、checkpoint、config、environment 身份。
2. Direct production adapter 對 before/after 各執行一次，保存 raw outputs。
3. 同一路徑／相同設定重複執行一次，檢查可重複性；不要只比較首次 cold-start timing。
4. 對每條 required、SUPPORTED-REAL 路徑使用相同 input bytes、checkpoint 與 effective settings。
5. 保存原始 detections；另保存各層可取得的 filtered/matched/events outputs。
6. 先比較 detector；通過後再定位 downstream 一致性。
7. 如 detector 不一致，保留所有輸出並報告；不得只看 final events 相同就掩蓋 detector 差異。
8. 結束後確認 source、input、checkpoint 未被修改。

Exact commands：code discovery 後填入；本草稿不發明 entry points 或 flags。

## 8. 比較格式與語義

## Detector record

至少包含：route、pair\_id、before/after、image hash、原圖尺寸、checkpoint hash、effective config identity，以及每個 detection 的 class\_id、confidence、bbox。

Canonical comparison bbox：原圖 pixel coordinates 的 xyxy，float；座標原點／邊界約定必須與契約一致。若 adapter 使用其他格式，保存 raw 值與明確 conversion；不得重複縮放、提前整數化或用 display-rounded values 比較。

## Detection matching

* 同一張 image 比較，class\_id 必須完全一致。
* Detection count 必須一致。
* 不以 list order 或 route-local detection ID 判定不同。
* 用同 class 下的 deterministic one-to-one matching 比對 bbox；演算法、tie-break 事前固定並保存。
* 通過須存在滿足所有欄位 tolerance 的完整一對一對應；不允許多對一配對。
* Unmatched detections、confidence 或 bbox 超限均報告，不以放寬 threshold 消除差異。

## Proposed strict tolerances

適用於相同環境/device/dtype、使用相同模型路徑的主要比較，執行前需批准：


| 欄位                                                 | 規則                                                           |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| Image/checkpoint/config identities                   | 完全一致；僅允許已聲明、不影響語義的 path/output metadata 差異 |
| Detection count / class\_id                          | 完全一致                                                       |
| Confidence                                           | absolute difference <= 1e-5                                    |
| Bbox 每個 xyxy 座標                                  | absolute difference <= 1e-3 original-image pixels              |
| NaN / Inf                                            | 不接受                                                         |
| Event category / matched semantic endpoints / counts | 完全一致                                                       |
| Normalized downstream displacement（若輸出）         | absolute difference <= 1e-6                                    |

這些是 proposed acceptance limits，不是已驗證的 backend 保證。若 route 只保存低精度值，標記 numeric parity 無法完整驗證，不能事後放寬 tolerance 製造 PASS。額外的 IoU／pixel-error summary 可用於診斷，不取代上述 pass criteria。

## 9. Downstream Parity

固定相同 B-side implementation 與所有 thresholds。比較每個 pair：

* Filter 後的物件集合與欄位（可取得時）。
* Before/after object matching 的語義對應。
* Added / removed / moved events 與對應物件。
* 如現有 schema 明確輸出 unchanged，亦比較；不新增假定狀態。
* Event counts、status 及可用的 displacement／score。

Route-local IDs 可以不同；先透過 detections 的一對一對應建立 ID mapping，再比較 object/event identity。不得只比較 aggregate counts，因為相同數量可能對應不同物件。

若 direct adapter 不提供 downstream，使用共同 pipeline 的 direct reference harness 作參考，並清楚標示這不是 adapter 本身輸出 events。

## 10. 分層 Pass Criteria

## L1 — Functional smoke

Checkpoint 成功載入，指定輸入完成真 inference，schema／數值範圍／座標空間符合契約。合法空 detections 可通過功能測試，但不得用來聲稱有覆蓋非空 detection/event parity。

## L2 — Detector parity

所有 required、可取得 raw detections 的路徑均符合 detection matching/tolerance 規則。同一路徑 repeatability 結果獨立記錄。

## L3 — Downstream parity

所有 required downstream 路徑的 object matching／events 語義一致。Detector-only 路徑為 N/A；UI 只有 screenshot 不足以支持 raw numerical parity。

各層獨立報 PASS / FAIL / BLOCKED / NOT TESTED / N/A。L1 PASS 不代表 L2/L3 PASS，亦不代表 detector accuracy 或 held-out quality 通過。

## 11. Evidence 保存

Output root：執行前指定專用新目錄，建議 `artifacts/acceptance/real-model-smoke/<run-id>/`；需確認不會意外提交大檔／敏感路徑。此為建議 layout，不宣稱 repo 已有相同目錄。

保存：

* run\_manifest.json：時間、source/checkpoint/fixture/config hashes、路徑清單、required routes、批准的 tolerances。
* environment.txt、commands.txt、source-status.txt、必要 source patch。
* config/：各 route 的 input config 與 effective values。
* raw/<route>/：原始 detections、events、stdout/stderr／exit status。
* canonical/<route>/：正規化 comparison records 與 ID mapping。
* comparisons/：每個欄位差異、unmatched objects、max errors、逐 route verdict。
* summary.md：分層結果、limitations、未覆蓋範圍與 findings。
* UI screenshot（如需要）只作補充，不作唯一 parity evidence。

不得覆蓋 E0 package、原 dataset 或舊 acceptance 證據；不需要複製 checkpoint 到每個輸出目錄，保留原位置與 hash 即可。

## 12. Failure Handling

* Hash/config/input mismatch：停止該比較，標 BLOCKED。
* Error 被吞掉變成空 detections/events：FAIL，保留 logs。
* 同一輸入 detector output 不一致：先檢查版本、effective args、preprocessing、NMS、device/dtype、serialization；不直接改 B-side thresholds。
* Detector 一致但 events 不同：檢查 filtering、matching、movement threshold、state／cache 與 event conversion。
* 只在 threshold 邊界附近不同：仍記錄差異，不因「接近 threshold」自動通過。
* 修復後使用新 run-id 重跑受影響範圍，保留原失敗 evidence；若修改 acceptance rules，記錄 revision 並重新比較全部受影響路徑。

## 13. 結果表


| Route          | Source/config verified | L1 functional | Repeatability | L2 detector parity | L3 downstream parity     | Evidence / limitation |
| -------------- | ---------------------- | ------------- | ------------- | ------------------ | ------------------------ | --------------------- |
| Direct adapter | TBC                    | NOT TESTED    | NOT TESTED    | Reference / TBC    | N/A or reference harness | TBC                   |
| Runner/service | TBC                    | NOT TESTED    | NOT TESTED    | NOT TESTED         | NOT TESTED               | TBC                   |
| CLI            | TBC                    | NOT TESTED    | NOT TESTED    | NOT TESTED         | NOT TESTED               | TBC                   |
| UI             | TBC                    | NOT TESTED    | NOT TESTED    | NOT TESTED         | NOT TESTED               | TBC                   |

## 14. 完成 Checklist

* 實際支援／required 路徑已由 code 確認。
* 所有 TBC execution fields 已補齊，或明確標出不能執行原因。
* Checkpoint、input、config、source 身份完整。
* 所有執行與比較都有 raw evidence。
* L1/L2/L3 分開結論，未將 N/A 當 PASS。
* 所有差異／未覆蓋範圍均如實記錄。
* 未觸碰 held-out test，未進行 training/tuning。
* 未以本輪功能驗收取代 E0 provenance 或 dataset-package acceptance。
