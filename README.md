# License Plate Detection

Project nay xay dung he thong phat hien va nhan dang bien so xe Viet Nam tu video. Ung dung dung YOLO de detect vung bien so, crop bien so tu frame, tien xu ly anh cho OCR, nhan dang chuoi ky tu bang Fast Plate OCR, sau do validate dinh dang bien so va hien thi ket qua tren Streamlit.

## Tong quan pipeline

Theo `Report.pdf`, pipeline tong the gom cac buoc:

1. Thu thap du lieu tu camera RTSP, frame co timestamp/camera ID va cac nguon public.
2. Tien xu ly du lieu: khu nhieu, can chinh do sang/do tuong phan, crop ROI bien so va resize ve kich thuoc dau vao cua model.
3. Detection: dung YOLO11 de sinh bounding box bien so.
4. OCR: crop bien so da detect, tien xu ly anh va nhan dang ky tu bang Fast Plate OCR da fine-tune.
5. Post-processing: lam sach chuoi, validate format bien so Viet Nam, gop ket qua theo track va hien thi real-time.

Trong source code, pipeline nay duoc hien thuc trong `appStreamlit.py`:

- Upload video bang Streamlit.
- Doc frame bang OpenCV va xu ly moi `skip_frames = 5` frame.
- Detect/tracking bien so bang `ultralytics.YOLO` voi ByteTrack.
- Crop bien so co padding 10%.
- Tien xu ly anh crop truoc OCR.
- OCR bang `fast_plate_ocr.LicensePlateRecognizer`.
- Vote nhieu lan doc trong cung mot track de tang do on dinh.
- Hien thi frame da annotate, danh sach detection, bien so hop le va confidence.

## Data Preprocessing

Project su dung hai tap du lieu rieng:

### 1. Detection dataset

Dataset detection duoc to chuc theo format YOLO:

```text
Data/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
|   +-- train/
|   +-- val/
|   +-- test/
+-- dataset.yaml
```

Moi anh co file label `.txt` tuong ung, chua bounding box cua class `license_plate`. Tat ca anh duoc resize ve `640x640` khi train YOLO. Tap du lieu duoc chia theo ty le train/validation/test va duoc augment de tang kha nang chiu nhieu trong moi truong thuc te.

### 2. OCR dataset

Dataset OCR gom cap anh bien so va text nhan dang:

```text
OCR_LicensePlate_Dataset/
+-- train/
|   +-- images/
|   +-- train_annotations.csv
+-- valid/
|   +-- images/
|   +-- valid_annotations.csv
+-- test/
    +-- images/
    +-- test_annotations.csv
```

Theo report, tap OCR co 3,763 anh:

| Split | So luong | Ty le |
| --- | ---: | ---: |
| Train | 2,993 | 79.54% |
| Validation | 381 | 10.12% |
| Test | 389 | 10.34% |

Du lieu duoc lam sach bang cach loai bo anh mo, bien so bi che khuat hoac khong doc duoc. Cac augmentation duoc dung trong qua trinh train/fine-tune gom:

- Brightness & Contrast Adjustment
- Motion Blur
- Coarse Dropout
- Horizontal Flip
- ShiftScaleRotate
- ISONoise
- ColorJitter
- ToGray

Trong ung dung Streamlit, anh bien so sau khi crop tiep tuc duoc tien xu ly truoc OCR:

- Bo qua crop qua nho de tranh OCR sai.
- Resize anh nho bang `INTER_CUBIC`.
- Chuyen sang grayscale.
- Tang tuong phan cuc bo bang CLAHE (`clipLimit=2.0`, `tileGridSize=(4, 4)`).
- Chuyen lai BGR va khu nhieu bang `cv2.fastNlMeansDenoisingColored`.

## Model Detection

Model detection trong project la file:

```text
DetectLisence_YOLO11.pt
```

Model duoc load trong `appStreamlit.py` bang:

```python
self.model = YOLO(model_path)
```

Theo report, model detect dua tren YOLO11n voi 3 thanh phan chinh:

- **Backbone**: trich xuat dac trung anh bang cac block convolution/C3/SPPF.
- **Neck/FPN**: ket hop dac trung da ty le de detect bien so o nhieu kich thuoc khac nhau.
- **Head/Detect**: sinh bounding box, confidence score va class label.

Model detection chi co 1 class la `license_plate`, duoc train 20 epochs. YOLO11n co thiet ke nhe, khoang 2.59M tham so, phu hop cho xu ly nhanh va gan real-time.

Trong app, model co hai che do:

- `model.track(..., tracker="bytetrack.yaml")`: tracking bien so theo frame, giu ID de vote OCR.
- `model(...)`: fallback neu tracking khong hoat dong.

Bounding box sau detection se duoc crop voi padding 10% de giu du thong tin vien bien so truoc khi dua sang OCR.

## Model Recognition

Model recognition trong project gom:

```text
cct_s_v1_vn.onnx
cct_s_v1_vn_plate_config.yaml
```

App load model bang `fast_plate_ocr`:

```python
LicensePlateRecognizer(
    onnx_model_path="cct_s_v1_vn.onnx",
    plate_config_path="cct_s_v1_vn_plate_config.yaml",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
```

Theo config hien co:

- Input OCR: RGB `64x128`
- `max_plate_slots`: 9
- Alphabet: `0-9`, `A-Z`, va ky tu padding `_`
- Resize interpolation: linear
- Khong giu nguyen aspect ratio khi resize dau vao OCR


## Post-processing va validation

Sau OCR, app thuc hien cac buoc hau xu ly:

- Chuan hoa text: uppercase, bo dau cach, dau gach, underscore va ky tu khong hop le.
- Validate format bien so bang regex:
  - `^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$`
  - `^[0-9]{2}[A-Z][0-9]{1,2}[0-9]{4,5}$`
- Luu nhieu lan doc OCR theo `track_id`.
- Vote tung vi tri ky tu dua tren confidence de tao bien so on dinh hon.
- Gop cac detection trung bien so va giu ket qua co OCR confidence cao nhat.

## Cau truc project

```text
Detect_Plate/
+-- appStreamlit.py
+-- DetectLisence_YOLO11.pt
+-- cct_s_v1_vn.onnx
+-- cct_s_v1_vn_plate_config.yaml
+-- Report.pdf
+-- Video_Test/
    +-- Download.mp4
    +-- cam_20250922_181322.mp4
    +-- video1.mp4
```

## Cai dat

Tao moi truong Python va cai cac thu vien can thiet:

```bash
pip install streamlit ultralytics opencv-python fast-plate-ocr onnxruntime
```

Neu may co GPU va muon chay OCR bang GPU, co the cai them ban onnxruntime phu hop voi CUDA:

```bash
pip install onnxruntime-gpu
```

## Chay ung dung

Tai thu muc project, chay:

```bash
streamlit run appStreamlit.py
```

Sau do upload video `.mp4`, `.avi`, `.mov` hoac `.mkv` tren giao dien Streamlit va bam **Start Detection**.

## Output

Ung dung hien thi:

- Video/frame da annotate bounding box va bien so nhan dang.
- Tong so detection.
- So bien so hop le.
- So bien so unique.
- Valid rate.
- Anh crop bien so, frame, timestamp, YOLO confidence, OCR/vote confidence va so lan doc OCR.


