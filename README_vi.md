# Tái hiện evaluation của M³Diff trên Modal

Mục tiêu đầu tiên là chạy checkpoint công bố của tác giả trên toàn bộ test set OmniDiff, tính các metric trong Bảng 3 và so sánh với bài báo. Đây là **tái hiện evaluation bằng checkpoint có sẵn**. Bộ chạy này chưa thực hiện lại training và chưa thay đổi phương pháp của tác giả.

**Trạng thái bàn giao:** đã kiểm tra dữ liệu đánh giá, chạy 15 bài kiểm tra CPU và kiểm tra lệnh với Modal SDK 1.6.3. Chưa build image trên Modal, chưa inference bằng GPU, chưa có kết quả thực nghiệm. Máy hiện tại chưa có profile Modal đăng nhập. Không có job cloud nào được khởi chạy trong lúc chuẩn bị bộ này.

## 1 Phạm vi thực nghiệm

| Nội dung | Thiết lập |
|---|---|
| Model | Checkpoint merged chính thức `IVC-liuyuan/M3Diff` |
| Test set | OmniDiff-Real 803 cặp; OmniDiff-Render 767 cặp |
| Tổng số cặp | 1.570; các cặp tái sử dụng 319 ảnh riêng biệt |
| GPU đề xuất | Một A100 40GB trên Modal; cần smoke test để xác nhận bộ nhớ và thời gian |
| Precision | BF16, theo mặc định của evaluator hiện được công bố |
| Attention | FlashAttention 2.5.8 |
| Generation | Greedy, temperature 0, một beam, tối đa 1.024 new token, cache bật |
| Prompt | Giữ nguyên prompt từng mẫu và thứ tự hai ảnh của tác giả |
| Metric | BLEU-4, METEOR, ROUGE-L, CIDEr |

Code, checkpoint, vision tower và dataset được cố định revision trong `assets.lock.json`. Chỉ đường dẫn đến vision tower trong config được chuyển thành đường dẫn cache cục bộ; kiến trúc và trọng số được giữ nguyên. Không cần tải base LLaVA để đánh giá checkpoint merged này.

## 2 Chuẩn bị trên máy của bạn

Giải nén trọn bộ vào một thư mục, mở PowerShell tại thư mục đó. Máy cá nhân chỉ điều khiển công việc; GPU và các thư viện nặng nằm trên Modal. Cần Python 3.10 trở lên và tài khoản Modal sử dụng được GPU.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv\Scripts\python.exe -m modal setup
```

Hoàn thành đăng nhập qua trang Modal do lệnh mở ra. Không gửi token hoặc mật khẩu vào chat. Nếu đã có một môi trường Python cài Modal 1.6.3 và đăng nhập, có thể sử dụng môi trường đó.

Kiểm tra lệnh trước khi chạy:

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --help
```

## 3 Tải tài nguyên trên CPU

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage prepare
```

Lệnh này tải khoảng 16,2 GB trọng số M³Diff, vision tower SigLIP và chỉ các ảnh test cần dùng. Tài nguyên được lưu trong Modal Volume `omnidiff-reproduction-v1` để dùng lại ở các lượt sau. Không tải toàn bộ dữ liệu training. Bước này kiểm tra số cặp, ID, file ảnh và cấu hình MDP trước khi đánh dấu chuẩn bị xong. Model được tải trên CPU để tránh giữ GPU trong lúc tải.

Lần đầu Modal cần build môi trường; FlashAttention có thể mất thời gian biên dịch. Chi phí CPU có thể phát sinh ngay cả khi chưa chạy GPU. Nếu báo lỗi build, giữ lại log; chưa kết luận model không chạy được.

## 4 Chạy thử 10 cặp ảnh

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage smoke
```

Chạy 5 cặp Real và 5 cặp Render, chọn bằng seed cố định 2025. Mục tiêu là xác nhận tải đúng checkpoint, sinh được caption, lưu kết quả và giải phóng GPU. Smoke test có timeout 15 phút; không tự retry. Không dùng 10 mẫu này để khẳng định tái hiện được điểm số toàn bộ benchmark.

Kết quả được tải về `results/smoke-<thời gian>-<mã>.zip`. Mở `Real/examples.json` và `Render/examples.json` để xem prompt, đường dẫn hai ảnh, caption dự đoán và reference. Kiểm tra `inference.log` nếu xuất hiện cảnh báo missing weights, unexpected weights hoặc lỗi generation trước khi chạy toàn bộ.

## 5 Chạy toàn bộ test set và tính metric

Sau khi smoke test thành công:

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage full
```

Hai tập được chạy lần lượt trên một GPU; không dùng mặc định 8 tiến trình của shell launcher gốc. Inference dùng trực tiếp evaluator Python của tác giả. Chấm điểm chuyển sang CPU để tránh giữ GPU trong bước tính metric. Timeout GPU là 2 giờ; không tự retry. Timeout là giới hạn mỗi lần gọi, không phải cam kết về tổng hóa đơn.

Archive kết quả gồm:

- `comparison.csv`: số bài báo, số tái hiện và chênh lệch theo từng metric.
- `Real/predictions.jsonl`, `Render/predictions.jsonl`: caption thực tế của từng cặp.
- `metrics_raw.json`, `metrics_x100.json`: metric gốc và metric đã nhân 100 theo cách trình bày của bài báo.
- `examples.json`: từng mẫu với prompt, ảnh, prediction và reference để phân tích hoặc làm slide.
- `run.json`, `assets.json`, `pip-freeze.txt`, `nvidia-smi.txt`, `command.json`, `inference.log`: bằng chứng cấu hình, phiên bản và thời gian.

Bảng mục tiêu dưới đây là **kết quả của tác giả**, chưa phải kết quả đã chạy:

| Test set | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| OmniDiff-Real | 14,3 | 18,9 | 32,9 | 31,3 |
| OmniDiff-Render | 15,7 | 19,9 | 35,3 | 28,3 |

Không sửa output để khớp bảng. Nếu điểm lệch, kiểm tra revision, precision, prompt, tokenizer, preprocessing và metric rồi ghi nhận nguyên nhân. Code hiện được công bố có thể khác môi trường thực nghiệm lịch sử; chỉ sau khi chạy mới biết mức khớp.

## 6 Tính đúng metric

`repro_helpers.py` dùng cùng PTB tokenizer và các scorer BLEU, METEOR, ROUGE, CIDEr của `pycocoevalcap` như entry point đánh giá chính thức. Chỉ bỏ SPICE vì Bảng 3 không báo metric đó; không thay scorer CIDEr. Tên output vẫn là CIDEr, dù paper gọi CIDEr-D trong phần evaluation protocol.

Chương trình yêu cầu đủ toàn bộ ID, không trùng, không dư và không có caption rỗng trước khi chấm điểm. Điều này tránh việc vô tình tính điểm trên một phần test set. Đặc biệt, CIDEr trên một subset không thể coi là điểm toàn bộ test set. Giữ caption tiếng Anh khi chấm vì reference gốc là tiếng Anh; có thể dịch ví dụ để trình bày slide sau.

## 7 Chi phí và phạm vi training

Theo bảng giá Modal kiểm tra ngày 31/10/2026, A100 40GB có đơn giá GPU khoảng **2,10 USD/giờ**, chưa gồm CPU, RAM và các khoản áp dụng khác. Smoke test có tối đa 15 phút thực thi GPU; full test có tối đa 2 giờ thực thi GPU. Còn có thời gian khởi tạo, build image và xử lý CPU. Thời lượng thực tế chỉ biết sau khi chạy thử; không có dự toán tổng tiền đã được xác nhận. Xem giá và hạn mức tài khoản trên Modal trước khi gọi các lệnh chạy.

Training trong bài báo dùng 8 A100 40GB trong khoảng 26 giờ, tương đương 208 GPU-giờ. Nếu áp cùng đơn giá GPU nói trên thì riêng GPU xấp xỉ 437 USD; đây chỉ là phép quy đổi từ thời gian tác giả công bố, không phải báo giá hay thời gian bảo đảm trên Modal. Chưa nên bắt đầu full training khi mục tiêu đầu tiên là có pipeline và kết quả evaluation hợp lệ.

Nếu giảng viên yêu cầu **tự huấn luyện lại** để được tính là tái hiện, evaluation bằng checkpoint chỉ là bước chuẩn bị. Khi đó cần bổ sung training theo dữ liệu và cấu hình bài báo, hoặc thống nhất phạm vi thu nhỏ và nêu rõ khác biệt. Không có bảo đảm điểm số môn học từ bộ chạy này.

## 8 Khi có lỗi hoặc mất kết nối

Console in `RUN ID` ngay trước lúc inference. Dữ liệu cũng ở Modal Volume. Nếu inference đã xong nhưng lấy archive hoặc chấm điểm bị lỗi, dùng lại run ID:

```powershell
.\.venv\Scripts\python.exe -m modal run modal_reproduce.py --stage collect --run-id full-THAY_BANG_RUN_ID
```

`collect` chỉ chấm điểm và đóng gói kết quả đã có, không inference lại. Nếu inference thất bại hoặc thiếu mẫu, chương trình từ chối chấm như một lượt full-test hoàn chỉnh. Các log đã commit vẫn có thể tải bằng Modal CLI hoặc xem trên dashboard. Không dùng lại cùng run ID để ghi đè lần chạy trước.

Nếu full run vượt timeout, cần chia test set thành các shard có ID cố định và ghép đủ trước khi chấm; bản hiện tại chưa tự resume giữa một split. Không tăng timeout hoặc chạy lặp tự động trước khi xem log và thời gian smoke test.

## 9 Nội dung cần thu cho slide sau thực nghiệm

1. Bài toán và đầu vào/đầu ra của M³Diff.
2. Kiến trúc MDP và nguồn code/checkpoint.
3. Môi trường, test split và decoding settings thực tế.
4. Bảng đối chiếu tác giả và kết quả chạy, kèm chênh lệch.
5. Ví dụ thành công và thất bại, không chỉ chọn các mẫu tốt.
6. Giới hạn tái hiện và hướng cải tiến tiếp theo.

Chưa tạo slide có kết quả vì hiện chưa thực hiện inference.

## Nguồn

- [Source code chính thức](https://github.com/Yuan-Liu-01/M3Diff)
- [Checkpoint M³Diff và hướng dẫn loader](https://huggingface.co/IVC-liuyuan/M3Diff)
- [OmniDiff dataset](https://huggingface.co/datasets/IVC-liuyuan/OmniDiff)
- [Bài báo ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OmniDiff_A_Comprehensive_Benchmark_for_Fine-grained_Image_Difference_Captioning_ICCV_2025_paper.html)
- [Modal GPU](https://modal.com/docs/guide/gpu), [Volumes](https://modal.com/docs/guide/volumes), [bảng giá](https://modal.com/pricing), [đăng nhập](https://modal.com/docs/guide)

Các script phụ trợ trong bộ này phục vụ điều phối Modal và kiểm tra kết quả; mã mô hình được tải từ repo tác giả ở revision đã cố định. Tuân thủ license của code, model và dataset tương ứng.
