import SimpleITK as sitk
import numpy as np
import os
import sys
import glob

if len(sys.argv) < 2:
    print("Usage: merge_labels.py <input_basename>")
    sys.exit(1)

input_basename = sys.argv[1]
cropped_name = f"{input_basename}_cropped_0000"

# === Пути ===
# === Динамический поиск файлов ===
FULL_CBCT = f"/input/{input_basename}.nii.gz"
if not os.path.exists(FULL_CBCT):
    fulls = glob.glob("/input/*.nii.gz")
    if not fulls:
        print("!!! Нет файлов в /input/")
        sys.exit(1)
    FULL_CBCT = fulls[0]

tips_files = glob.glob("/home/user/data_tips_resample_teeth_instance/*.nii.gz")
if not tips_files:
    print("!!! Нет TIPs маски в /home/user/data_tips_resample_teeth_instance/")
    sys.exit(1)
TIPS_MASK = tips_files[0]

dental_files = glob.glob("/home/user/dental-output/*.nii.gz")
if not dental_files:
    print("!!! Нет Dental маски в /home/user/dental-output/")
    sys.exit(1)
DENTAL_MASK = dental_files[0]

OUTPUT = "/home/user/combined_labels.nii.gz"

print(f"Full CBCT:   {FULL_CBCT}")
print(f"TIPs mask:   {TIPS_MASK}")
print(f"Dental mask: {DENTAL_MASK}")

print(f"Full CBCT:   {FULL_CBCT}")
print(f"TIPs mask:   {TIPS_MASK}")
print(f"Dental mask: {DENTAL_MASK}")

full = sitk.ReadImage(FULL_CBCT)
print(f"Full size: {full.GetSize()}, spacing: {full.GetSpacing()}")

out_origin = full.GetOrigin()
out_direction = full.GetDirection()
out_spacing = full.GetSpacing()

tips = sitk.Cast(sitk.ReadImage(TIPS_MASK), sitk.sitkUInt8)
dental = sitk.Cast(sitk.ReadImage(DENTAL_MASK), sitk.sitkUInt8)

print("Resampling to full space...")
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(full)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampler.SetDefaultPixelValue(0)

tips_full = resampler.Execute(tips)
del tips
dental_full = resampler.Execute(dental)
del dental, full

tips_arr = sitk.GetArrayFromImage(tips_full)
del tips_full
dental_arr = sitk.GetArrayFromImage(dental_full)
del dental_full

print(f"Arrays: tips={tips_arr.shape}, dental={dental_arr.shape}")

combined = np.zeros(tips_arr.shape, dtype=np.uint8)

combined[dental_arr == 1] = 51
combined[dental_arr == 2] = 52
combined[dental_arr == 5] = 55
del dental_arr

teeth_mask = tips_arr > 0
combined[teeth_mask] = tips_arr[teeth_mask]
del tips_arr, teeth_mask

print(f"Saving: {OUTPUT}")
out = sitk.GetImageFromArray(combined)
out.SetOrigin(out_origin)
out.SetDirection(out_direction)
out.SetSpacing(out_spacing)
sitk.WriteImage(out, OUTPUT, True)

size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Size: {size_mb:.1f} MB")

print("\n=== Labels ===")
counts = np.bincount(combined.ravel(), minlength=56)
for u, c in enumerate(counts):
    if c == 0:
        continue
    if u == 0:
        name = "Фон"
    elif u == 51:
        name = "Верхняя челюсть"
    elif u == 52:
        name = "Нижняя челюсть"
    elif u == 55:
        name = "Канал"
    elif 11 <= u <= 48:
        name = f"Зуб FDI {u}"
    else:
        name = "?"
    print(f"  {u:>3} ({name}): {c:>10}")

print("\nГотово!")