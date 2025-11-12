import cv2
import torch
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from tqdm import tqdm


MODEL_PATH = Path(r"D:\Diploma AI\try_2\best_model_with_classes.pth")
VIDEO_FOLDER = Path(r"D:\Diploma AI\try_2")
OUTPUT_ROOT = VIDEO_FOLDER
SKIP_FRAMES = 60
CONFIDENCE_THRESHOLD = 0.95

state = torch.load(MODEL_PATH, map_location='cpu')
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 4)  # 4 класса
model.load_state_dict(state['model_state_dict'])
CLASS_NAMES = state['classes']

TARGET_CLASSES = ['audi_a6_c4', 'shkoda_fabia', 'uaz_images']
FOLDER_NAMES = {
    'audi_a6_c4': 'audi',
    'shkoda_fabia': 'shkoda',
    'uaz_images': 'uaz'
}

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === ТРАНСФОРМАЦИИ — ТОЧНО КАК ПРИ ОБУЧЕНИИ! ===
transform = transforms.Compose([
    transforms.Resize((500, 500)),  # ← КРИТИЧНО: 500x500
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

videos = list(VIDEO_FOLDER.glob("*.mp4"))
if not videos:
    raise FileNotFoundError("Нет .mp4 файлов в текущей папке!")

print(f"Найдено видео: {len(videos)}")

for video_path in videos:
    video_name = video_path.stem  # "1" или "2"
    print(f"\nОБРАБОТКА: {video_name}.mp4")


    output_dir = OUTPUT_ROOT / video_name
    output_dir.mkdir(exist_ok=True)


    for folder in FOLDER_NAMES.values():
        (output_dir / folder).mkdir(exist_ok=True)

    saved_count = {cls: 0 for cls in TARGET_CLASSES}
    processed_frames = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ОШИБКА: не открывается {video_name}.mp4")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"  {video_name}.mp4", leave=False)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        if frame_idx % SKIP_FRAMES != 0:
            continue

        processed_frames += 1

        try:

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)


            input_tensor = transform(pil_img).unsqueeze(0).to(device)


            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, idx = torch.max(probs, 1)
                conf_val = conf.item()
                pred_class = CLASS_NAMES[idx.item()]


            if pred_class in TARGET_CLASSES and conf_val >= CONFIDENCE_THRESHOLD:
                folder_name = FOLDER_NAMES[pred_class]
                save_path = output_dir / folder_name / f"{video_name}_f{frame_idx:06d}_c{conf_val:.2f}.jpg"
                cv2.imwrite(str(save_path), frame)
                saved_count[pred_class] += 1

            pbar.set_postfix({
                'audi': saved_count['audi_a6_c4'],
                'shkoda': saved_count['shkoda_fabia'],
                'uaz': saved_count['uaz_images'],
                'conf': f"{conf_val:.2f}"
            })

        except Exception as e:
            print(f"  Ошибка на кадре {frame_idx}: {e}")

    cap.release()
    pbar.close()

    print(f"  ГОТОВО: {video_name}.mp4")
    for cls in TARGET_CLASSES:
        folder = FOLDER_NAMES[cls]
        print(f"    → {folder}: {saved_count[cls]} кадров → {output_dir / folder}")
    print(f"    Обработано: {processed_frames} кадров")

print("\n" + "="*60)
print("ВСЁ ЗАВЕРШЕНО!")
print(f"Результаты: {OUTPUT_ROOT}")
print("="*60)



import shutil
from pathlib import Path

def clean_empty_dirs():
    current_dir = Path(".")
    

    def remove_empty(path: Path):
        if not path.is_dir():
            return False
        

        children_empty = True
        for child in path.iterdir():
            if child.is_dir():
                if not remove_empty(child):
                    children_empty = False
            else:
                children_empty = False  # найден файл


        if children_empty:
            try:
                path.rmdir()  # только для пустых
                print(f"Удалена пустая папка: {path}")
                return True
            except Exception as e:
                print(f"Ошибка при удалении {path}: {e}")
                return False
        return False  # не пуста


    for item in sorted(current_dir.iterdir(), key=lambda x: len(x.parts), reverse=True):
        if item.is_dir():
            remove_empty(item)


clean_empty_dirs()
