# M³Diff — Tái hiện evaluation trên Modal

Bộ script chạy **M³Diff** trên **OmniDiff** bằng GPU của Modal, phục vụ thực nghiệm cho bài báo [OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OmniDiff_A_Comprehensive_Benchmark_for_Fine-grained_Image_Difference_Captioning_ICCV_2025_paper.html).

Bài toán **Image Difference Captioning** nhận hai ảnh và sinh caption mô tả những thay đổi giữa chúng, chẳng hạn vật thể xuất hiện, biến mất, đổi màu hoặc di chuyển.

Repository này **tái hiện evaluation bằng checkpoint đã huấn luyện của tác giả**; chưa tái hiện training và chưa triển khai cải tiến mô hình. Đã chạy thành công toàn bộ 1.570 cặp test trên Modal.

## 1. Nguồn code và phạm vi

| Thành phần | Nguồn / vai trò |
|---|---|
| Kiến trúc M³Diff, MDP, loader và inference | [Repository chính thức của tác giả](https://github.com/Yuan-Liu-01/M3Diff) |
| Checkpoint merged | [IVC-liuyuan/M3Diff](https://huggingface.co/IVC-liuyuan/M3Diff) |
| Ảnh test | [IVC-liuyuan/OmniDiff](https://huggingface.co/datasets/IVC-liuyuan/OmniDiff) |
| Prompt và reference captions | Repository chính thức, giữ nguyên theo revision đã chọn |
| `modal_reproduce.py` | Script bổ sung để chuẩn bị tài nguyên và điều phối chạy trên Modal |
| `repro_helpers.py` | Script bổ sung để kiểm tra dữ liệu, chấm điểm và tổng hợp kết quả |

Các script phụ trợ được xây dựng với sự hỗ trợ của OpenAI Codex. Phương pháp M³Diff thuộc về tác giả bài báo; repository này không phải implementation chính thức của tác giả.

Inference gọi trực tiếp evaluator của code gốc. Phần chấm điểm dùng wrapper riêng với PTBTokenizer và các scorer của `pycocoevalcap`: BLEU-4, METEOR, ROUGE-L, CIDEr. Không gọi trực tiếp `metrics.py` gốc và không tính SPICE vì Bảng 3 không báo metric này.

Code được cố định tại commit `4e346e52185620ec6fa17235d93b0c26a9d47a71`, **không tự cập nhật theo nhánh `main`**. Các revision tài nguyên được ghi trong `assets.lock.json`; script thực thi sử dụng bản tương ứng trong biến `LOCK` của `modal_reproduce.py`.

## 2. Cấu trúc thư mục

```text
.
├── README.md
├── .gitignore
├── requirements-client.txt
├── assets.lock.json
├── modal_reproduce.py
├── repro_helpers.py
├── test_repro_helpers.py
└── results/                    # Sinh sau khi chạy; không đưa lên Git mặc định
    └── full-<timestamp>-<id>.zip
```

Không cần đặt source M³Diff, dataset hay model weights trong repository này. Script tự tải chúng về môi trường Modal.

## 3. Yêu cầu

- Máy cá nhân có Python 3.10 trở lên và Internet. Không cần GPU, CUDA hoặc PyTorch trên máy cá nhân.
- Tài khoản [Modal](https://modal.com/) có thể sử dụng GPU A100 40GB.
- Các lệnh dưới đây dành cho **Windows PowerShell hoặc CMD**, chạy tại thư mục chứa `modal_reproduce.py`.

Modal build môi trường Linux/Python 3.10 và cài các thư viện GPU trên cloud. Chạy các stage có thể phát sinh chi phí GPU, CPU, RAM và lưu trữ; xem [bảng giá Modal](https://modal.com/pricing) và hạn mức tài khoản trước khi chạy.

## 4. Cài đặt trên máy cá nhân

Tải repository bằng **Code → Download ZIP** rồi giải nén, hoặc clone bằng Git. Mở terminal tại thư mục chứa các file trong mục 2.

**Chạy từng lệnh riêng, nhấn Enter và chờ hoàn tất trước khi chạy lệnh tiếp theo.** Không nối các lệnh thành một dòng.

Tạo môi trường Python:

```powershell
py -m venv .venv
```

Cài Modal client:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt
```

Đăng nhập Modal và hoàn tất xác thực trên trình duyệt:

```powershell
.\.venv\Scripts\python.exe -m modal setup
```

Kiểm tra các tham số của bộ chạy, chưa khởi chạy thực nghiệm:

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --help
```

Không cần activate `.venv` vì các lệnh gọi trực tiếp Python trong môi trường này.

## 5. Chạy thực nghiệm

Thứ tự cho lần đầu: **prepare → smoke → full**.

### Bước 1 — Chuẩn bị tài nguyên

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage prepare
```

Bước này tải checkpoint M³Diff khoảng 16,2 GB, vision tower SigLIP và ảnh test cần dùng; không tải toàn bộ dữ liệu training. Script kiểm tra ID, ảnh và cấu hình model, rồi lưu tài nguyên trong Modal Volume `omnidiff-reproduction-v1`.

Tài nguyên được tải trên CPU. Những lần sau có thể dùng lại cache nếu revision không thay đổi. Lần đầu Modal cần build image; bước cài FlashAttention có thể mất thời gian.

### Bước 2 — Chạy thử 10 cặp ảnh

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage smoke
```

Chạy 5 cặp Real và 5 cặp Render với seed chọn mẫu cố định `2025`. Kết quả được tải về `results/smoke-<timestamp>-<id>.zip`.

Giải nén và xem `Real/examples.json`, `Render/examples.json` để kiểm tra model đã sinh caption. Smoke test chỉ kiểm tra pipeline, **không tính điểm benchmark**. Chỉ chạy bước tiếp theo khi smoke test hoàn tất thành công.

### Bước 3 — Chạy toàn bộ test set

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage full
```

Chạy lần lượt **803 cặp Real** và **767 cặp Render** trên một A100 40GB. Sau inference, chương trình chấm điểm trên CPU, in bảng so sánh với bài báo và tải ZIP về `results/`.

Lượt thực nghiệm ngày 01/11/2026 mất khoảng **58,7 phút cho hai bước inference**, bao gồm thời gian tải model trong mỗi bước. Đây không phải tổng thời gian build, chuẩn bị dữ liệu, chấm điểm hay cam kết thời gian cho lần chạy khác.

Khi terminal hiển thị `Saved: ...zip` và `App completed`, lượt chạy đã hoàn tất. Mỗi lần gọi `full` tạo một lượt inference mới; nếu chỉ cần lấy lại kết quả đã hoàn thành, dùng `collect` bên dưới.

### Lấy lại kết quả hoặc chấm điểm lại

Thay `RUN_ID_CUA_BAN` bằng đúng chuỗi xuất hiện sau `RUN ID:` trong log:

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage collect --run-id RUN_ID_CUA_BAN
```

`collect` dùng prediction đã lưu trên Volume, không chạy inference lại. Nó chỉ hoạt động khi inference đã hoàn tất; không khôi phục một split đang chạy dở.

## 6. Thiết lập thực nghiệm

| Thiết lập | Giá trị |
|---|---|
| GPU | 1 × NVIDIA A100 40GB |
| Checkpoint | M³Diff merged chính thức |
| Precision khi inference | BF16 |
| PyTorch / CUDA image | 2.2.2 / 12.1.1 |
| FlashAttention | 2.5.8 |
| Decoding | Greedy; `temperature=0`, `num_beams=1` |
| Giới hạn output | `max_new_tokens=1024` |
| Conversation mode | `qwen_2` |
| KV cache | Bật |
| Test set | Real: 803 cặp; Render: 767 cặp |
| Metric | BLEU-4, METEOR, ROUGE-L, CIDEr |

Kiến trúc, trọng số, prompt và thứ tự hai ảnh được giữ theo nguồn đã cố định. Chỉ đường dẫn vision tower trong config được đổi sang cache trên Modal. Không cần tải thêm base LLaVA để chạy checkpoint merged này.

## 7. Đọc kết quả

Giải nén ZIP trong `results/`:

| File trong ZIP | Nội dung |
|---|---|
| `comparison.csv` | Điểm bài báo, điểm tái hiện và chênh lệch |
| `Real/predictions.jsonl`, `Render/predictions.jsonl` | Caption model sinh cho từng cặp |
| `Real/examples.json`, `Render/examples.json` | ID, đường dẫn ảnh, prompt, prediction và reference |
| `Real/metrics_raw.json`, `Render/metrics_raw.json` | Điểm gốc do scorer trả về |
| `Real/metrics_x100.json`, `Render/metrics_x100.json` | Điểm nhân 100 để trình bày như bảng bài báo |
| `run.json`, `assets.json` | Cấu hình lượt chạy, revision, số mẫu, thời gian và hash |
| `pip-freeze.txt`, `scoring-pip-freeze.txt`, `nvidia-smi.txt` | Môi trường inference, scoring và thông tin GPU |
| `Real/command.json`, `Render/command.json` | Lệnh inference thực tế |
| `Real/inference.log`, `Render/inference.log` | Log của từng split |

Các file metric và `comparison.csv` được tạo cho lượt `full`. ZIP chứa đường dẫn ảnh, không chứa ảnh gốc; ảnh nằm trong dataset trên Modal và có thể xem từ nguồn Hugging Face.

Chương trình kiểm tra đủ ID, không trùng, không dư và không có caption rỗng trước khi chấm điểm. Giữ caption tiếng Anh để đánh giá với reference tiếng Anh; chỉ dịch ví dụ khi trình bày báo cáo.

## 8. Kết quả baseline đã chạy

Lượt chạy: `full-20261101T074459Z-253021`, ngày **01/11/2026**, đủ 803 cặp Real và 767 cặp Render. Các điểm bên dưới đã nhân 100, làm tròn hai chữ số thập phân.

| Split | Kết quả | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---:|---:|---:|---:|
| Real | Bài báo, Bảng 3 | 14.30 | 18.90 | 32.90 | 31.30 |
| Real | Tái hiện | 14.67 | 19.54 | 32.80 | 36.28 |
| Render | Bài báo, Bảng 3 | 15.70 | 19.90 | 35.30 | 28.30 |
| Render | Tái hiện | 17.16 | 20.17 | 35.59 | 30.89 |

Đây là kết quả evaluation của checkpoint công bố, **không phải kết quả cải tiến phương pháp**. Chưa xác định được nguyên nhân chính xác của chênh lệch với bài báo; không khẳng định môi trường và tài nguyên công bố trùng hoàn toàn với thực nghiệm lịch sử của tác giả.

Khi phát triển cải tiến, dùng kết quả tái hiện này làm baseline, giữ cùng test set và cách chấm điểm. Chọn cấu hình trên validation set, rồi đánh giá phương án cuối cùng trên test set.

## 9. Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `venv: error: unrecognized arguments` | Các lệnh có thể đã bị nối liền. Chạy riêng từng lệnh ở mục 4. |
| Không tìm thấy `requirements-client.txt` hoặc script | Mở terminal tại đúng thư mục chứa `modal_reproduce.py`. |
| `No module named modal` | Cài requirements và gọi đúng `.\.venv\Scripts\python.exe`. |
| Chưa xác thực Modal | Chạy lại lệnh `modal setup` ở mục 4. |
| Chưa có tài nguyên đã chuẩn bị | Hoàn tất stage `prepare` trước `smoke` hoặc `full`. |
| Inference xong nhưng chấm điểm/tải ZIP lỗi | Dùng `collect` với đúng run ID. |
| Inference thiếu mẫu, lỗi hoặc timeout | Xem log trên Modal và `inference.log`; `collect` không tiếp tục sinh các caption còn thiếu. |
| Volume chứa revision khác | Nếu chủ động thay revision, đồng bộ `LOCK` và `assets.lock.json`, rồi dùng tên Volume mới trong `VOLUME_NAME`. |

Timeout GPU là 15 phút cho smoke và 2 giờ cho full; không tự retry. Bộ chạy hiện chưa hỗ trợ resume giữa một split.

## 10. Đưa bộ chạy lên GitHub

Đặt các file sau tại thư mục gốc của repository:

- `README.md` và `.gitignore`.
- `modal_reproduce.py`, `repro_helpers.py`, `test_repro_helpers.py`.
- `requirements-client.txt` và `assets.lock.json`.

Có thể tạo repository trên GitHub và dùng **Add file → Upload files** để tải các file này lên. Nếu upload bằng giao diện web, chỉ chọn các file nêu trên: `.gitignore` không lọc các file được tải lên thủ công.

`.gitignore` đi kèm loại trừ `.venv/`, cache, cấu hình IDE, thông tin đăng nhập, model weights, dataset và `results/` khỏi việc thêm file thông thường bằng Git. Không cần đưa thư mục `M3Diff_Modal/` lồng bên trong hoặc ZIP bộ chạy vào repository.

Nếu muốn công bố bằng chứng thực nghiệm, có thể đính kèm ZIP kết quả đã kiểm tra vào một GitHub Release. Bảng baseline ở mục 8 có thể dùng để đọc nhanh kết quả ngay trên trang repository.

## 11. Trích dẫn và ghi nhận nguồn

```bibtex
@inproceedings{Liu_2025_ICCV,
  author    = {Liu, Yuan and Hou, Saihui and Hou, Saijie and Du, Jiabao and Meng, Shibei and Huang, Yongzhen},
  title     = {OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {21440--21449}
}
```

Tham khảo [project page](https://yuan-liu-omnidiff.github.io/), [source code chính thức](https://github.com/Yuan-Liu-01/M3Diff), [checkpoint](https://huggingface.co/IVC-liuyuan/M3Diff) và [dataset](https://huggingface.co/datasets/IVC-liuyuan/OmniDiff). Code, checkpoint và dataset chịu các điều khoản sử dụng tương ứng từ nguồn phát hành.
