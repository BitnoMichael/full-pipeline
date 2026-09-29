#!/bin/bash
set +e

echo "=== [0/5] Подготовка ==="
# Определяем имя входного файла
INPUT_FILE=$(ls /input/*.nii.gz | head -1)
INPUT_BASENAME=$(basename "$INPUT_FILE" .nii.gz)
echo "Input: $INPUT_FILE"
echo "Basename: $INPUT_BASENAME"

# Для зубов обрезаем, имя будет с суффиксом _cropped_0000
CROPPED_NAME="${INPUT_BASENAME}_cropped_0000"
echo "Cropped name: $CROPPED_NAME"

# === [1/5] Копирование весов ===
echo "=== [1/5] Копирование весов ==="
mkdir -p /home/user/TIPs/nnResults
cp -r /models-tips/* /home/user/TIPs/nnResults/

DENTAL_TARGET="/home/user/dental-nnUNet-results/Dataset112_DentalSegmentator"
mkdir -p "$DENTAL_TARGET"
cp -r /models-dental/* "$DENTAL_TARGET/"

# === [2/5] Обрезка CBCT для TIPs ===
echo "=== [2/5] Обрезка CBCT ==="
mkdir -p /home/user/data_tips
python3 /home/user/crop_smart.py "$INPUT_FILE" "/home/user/data_tips/${CROPPED_NAME}.nii.gz"

# === [3/5] TIPs: сегментация зубов ===
echo "=== [3/5] TIPs: сегментация зубов ==="
cd /home/user/TIPs
python3 TIPs.py /home/user/data_tips

# === [4/5] DentalSegmentator: сегментация челюстей ===
echo "=== [4/5] DentalSegmentator ==="

CKPT_DIR="$DENTAL_TARGET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CKPT_DIR/checkpoint_best.pth" ]; then
    CKPT="checkpoint_best.pth"
elif [ -f "$CKPT_DIR/checkpoint_final.pth" ]; then
    CKPT="checkpoint_final.pth"
else
    echo "!!! Не найден чекпоинт в $CKPT_DIR"
    ls -la "$CKPT_DIR"
    exit 1
fi
echo "Используем чекпоинт: $CKPT"

mkdir -p /home/user/dental-output
env nnUNet_results="$DENTAL_TARGET/.." \
    /home/user/dental-venv/bin/nnUNetv2_predict \
    -i /input \
    -o /home/user/dental-output \
    -d 112 -c 3d_fullres -tr nnUNetTrainer \
    -chk "$CKPT" -f 0 -step_size 0.5 \
    -npp 1 -nps 1

# === [5/5] Объединение ===
echo "=== [5/5] Объединение масок ==="
python3 /home/user/merge_labels.py "$INPUT_BASENAME"

# === Копирование результатов ===
echo "=== Копирование в /output ==="
mkdir -p /output

# Копируем всё что есть — с любыми именами
cp /home/user/combined_labels.nii.gz /output/combined_labels.nii.gz || echo "!!! Нет combined"

# TIPs маска — берём первый .nii.gz
TIPS_OUT=$(ls /home/user/data_tips_resample_teeth_instance/*.nii.gz 2>/dev/null | head -1)
if [ -n "$TIPS_OUT" ]; then
    cp "$TIPS_OUT" /output/teeth_only.nii.gz
    echo "Скопирована TIPs маска: $TIPS_OUT"
fi

# Dental маска — берём первый .nii.gz
DENTAL_OUT=$(ls /home/user/dental-output/*.nii.gz 2>/dev/null | head -1)
if [ -n "$DENTAL_OUT" ]; then
    cp "$DENTAL_OUT" /output/jaws_only.nii.gz
    echo "Скопирована Dental маска: $DENTAL_OUT"
fi

echo "=== Готово ==="
ls -la /output