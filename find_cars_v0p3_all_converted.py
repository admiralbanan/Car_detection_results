
import cv2
import torch
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


MODEL_PATH = Path(r"D:\Diploma AI\best_model_with_classes.pth")
VIDEOS_ROOT = Path(r"D:\Diploma AI\Videos")
OUTPUT_ROOT = Path(r"D:\Diploma AI\founded_batch")
SKIP_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.80


state = torch.load(MODEL_PATH, map_location='cpu')
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 4)
model.load_state_dict(state['model_state_dict'])
CLASS_NAMES = state['classes']


TARGET_CLASSES = ['audi_a6_c4', 'shkoda_fabia', 'uaz_images']
CLASS_FOLDERS = {cls: cls.split('_')[0] for cls in TARGET_CLASSES}

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


date_folders = [f for f in VIDEOS_ROOT.iterdir() if f.is_dir()]
converted_paths = []

for date_folder in date_folders:
    converted_folder = date_folder / "converted"
    if converted_folder.exists():
        videos = list(converted_folder.glob("*_fixed.mp4"))
        if videos:
            converted_paths.append((date_folder.name, converted_folder, videos))

if not converted_paths:
    raise FileNotFoundError("Не найдено папок 'converted' с видео *_fixed.mp4")

print(f"Найдено дат с видео: {len(converted_paths)}")
print("-" * 80)


for date_name, conv_folder, video_files in converted_paths:
    print(f"\nОБРАБОТКА ДАТЫ: {date_name}")
    print(f"Видео: {len(video_files)}")


    date_output = OUTPUT_ROOT / date_name
    date_output.mkdir(parents=True, exist_ok=True)
    for folder in CLASS_FOLDERS.values():
        (date_output / folder).mkdir(exist_ok=True)

    total_saved = {cls: 0 for cls in TARGET_CLASSES}
    total_processed = 0


    for video_path in video_files:
        print(f"  → {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    ОШИБКА: не открывается {video_path.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        video_saved = {cls: 0 for cls in TARGET_CLASSES}
        pbar = tqdm(total=total_frames, desc=f"    {video_path.stem}", leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pbar.update(1)

            if frame_idx % SKIP_FRAMES != 0:
                continue

            total_processed += 1

            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, idx = torch.max(probs, 1)
                    conf_val = conf.item()
                    pred_class = CLASS_NAMES[idx.item()]

                # Сохраняем только целевые классы
                if pred_class in TARGET_CLASSES and conf_val >= CONFIDENCE_THRESHOLD:
                    folder_name = CLASS_FOLDERS[pred_class]
                    save_path = date_output / folder_name / f"{video_path.stem}_frame_{frame_idx:06d}_c{conf_val:.2f}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    video_saved[pred_class] += 1
                    total_saved[pred_class] += 1

                pbar.set_postfix({
                    'audi': video_saved['audi_a6_c4'],
                    'fabia': video_saved['shkoda_fabia'],
                    'uaz': video_saved['uaz_images'],
                    'conf': f"{conf_val:.2f}"
                })

            except Exception as e:
                print(f"    Ошибка на кадре {frame_idx}: {e}")

        cap.release()
        pbar.close()

        print(f"    → Найдено: audi={video_saved['audi_a6_c4']}, fabia={video_saved['shkoda_fabia']}, uaz={video_saved['uaz_images']}")


    print(f"\nИТОГО за {date_name}:")
    for cls in TARGET_CLASSES:
        folder = CLASS_FOLDERS[cls]
        count = total_saved[cls]
        print(f"  {cls:12} → {count:4d} кадров → {date_output / folder}")
    print(f"  Обработано кадров: {total_processed}")


print("ЗАВЕРШЕНО!")
print(f"Результаты: {OUTPUT_ROOT}")