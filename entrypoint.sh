#!/bin/bash
set -e

echo "=== [1/4] Crop для TIPs ==="
python3 /home/user/crop_smart.py

echo "=== [2/4] TIPs: сегментация зубов ==="
mkdir -p /home/user/data_teeth
cp /tmp/patient01a_tips_0000.nii.gz /home/user/data_teeth/patient01a_cropped_0000.nii.gz

mkdir -p /home/user/TIPs/nnResults
cp -r /models-tips/* /home/user/TIPs/nnResults/
cd /home/user/TIPs
python3 TIPs.py /home/user/data_teeth

echo "=== [3/4] DentalSegmentator: сегментация челюстей ==="
TARGET="/home/user/dental-nnUNet-results/Dataset112_DentalSegmentator"
mkdir -p "$TARGET"
cp -r /models-dental/* "$TARGET/"

CKPT_DIR="$TARGET/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CKPT_DIR/checkpoint_best.pth" ]; then
    CKPT="checkpoint_best.pth"
else
    CKPT="checkpoint_final.pth"
fi

export nnUNet_results="/home/user/dental-nnUNet-results"
mkdir -p /home/user/dental-output
/home/user/dental-venv/bin/nnUNetv2_predict \
  -i /input \
  -o /home/user/dental-output \
  -d 112 -c 3d_fullres -tr nnUNetTrainer \
  -chk "$CKPT" -f 0 -step_size 0.5 \
  -npp 1 -nps 1

echo "=== [4/4] Объединение масок ==="
python3 /home/user/merge_labels.py

echo "=== Копирование в /output ==="
mkdir -p /output
cp /home/user/data_teeth_resample_teeth_instance/*.nii.gz /output/teeth_only.nii.gz
cp /home/user/dental-output/*.nii.gz /output/jaws_only.nii.gz
cp /home/user/combined_labels.nii.gz /output/combined_labels.nii.gz

echo "=== Готово ==="
ls -la /output