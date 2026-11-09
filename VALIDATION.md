# Kiểm tra trước khi chạy trên Modal

Ngày chuẩn bị: 31/10/2026.

Đã thực hiện:

- Đọc source chính thức tại commit `4e346e52185620ec6fa17235d93b0c26a9d47a71`.
- Đối chiếu 803 prompt/reference Real và 767 prompt/reference Render, mỗi cặp có hai ảnh đúng thứ tự.
- Kiểm tra metadata công khai: checkpoint merged, MDP, layer selection, trọng số vision tower và cơ chế nạp trained vision weights.
- 15 kiểm tra CPU về tính đầy đủ của ID, từ chối ID trùng/thiếu/dư, caption rỗng, bảo toàn thứ tự ảnh: đạt.
- Python compilation của các script: đạt.
- Modal SDK 1.6.3 nạp module và hiển thị CLI `--help`: đạt, không gọi cloud.
- Rà soát độc lập cấu hình Modal, tham số evaluator và cách tính metric; sửa việc phụ thuộc file JSON bên cạnh module khi import từ xa.

Chưa thực hiện:

- Build container image trên Modal.
- Cài/biên dịch FlashAttention trong image từ xa.
- Tải checkpoint và ảnh test vào Modal Volume.
- Inference GPU, kiểm tra VRAM/thời gian thực tế, hoặc xác nhận metric số.
- Training hoặc cải tiến mô hình.

Không coi kiểm tra CPU hoặc việc hiển thị CLI thành công là bằng chứng model đã chạy được trên GPU.

Chạy lại kiểm tra dữ liệu nếu đã clone source:

```powershell
py -X utf8 test_repro_helpers.py --repo DUONG_DAN_TOI_M3Diff
```

Nếu không có source ở đường dẫn được chỉ định, các kiểm tra dữ liệu chính thức sẽ được đánh dấu bỏ qua; các kiểm tra prediction vẫn chạy. Kết quả 15 bài đạt ở trên được thực hiện với source thực tế, không có bài bỏ qua.
