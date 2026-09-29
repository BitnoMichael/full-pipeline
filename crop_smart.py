import nibabel as nib
import numpy as np
import os
import glob

# === Настройки ===
TARGET = 600          # целевой размер по каждой оси (600 вокселей × 0.2 мм = 120 мм)
PAD_KEEP = 40         # padding на передней/нижней стороне (где зубы)
PAD_CUT = 5           # минимальный padding с затылка/макушки

# === Поиск входного файла ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Если скрипт запущен с аргументом - используем его
import sys
if len(sys.argv) > 1:
    INPUT = sys.argv[1]
else:
    # Иначе ищем .nii.gz в папке со скриптом
    candidates = [f for f in glob.glob(os.path.join(SCRIPT_DIR, "*.nii.gz"))
                  if "_0000" in f and "_tips" not in f and "_small" not in f]
    if not candidates:
        raise FileNotFoundError(f"Нет .nii.gz в {SCRIPT_DIR}")
    INPUT = candidates[0]

# Если второй аргумент - путь вывода
if len(sys.argv) > 2:
    OUTPUT = sys.argv[2]
else:
    OUTPUT = os.path.join(SCRIPT_DIR, "patient01a_tips_0000.nii.gz")

print(f"Input:  {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Target: {TARGET} voxels per axis")

# === Загрузка ===
img = nib.load(INPUT)
data = img.get_fdata(dtype=np.float32)
print(f"Original shape: {data.shape}")
print(f"Spacing: {img.header.get_zooms()}")

# === Анатомическая ориентация ===
axcodes = nib.aff2axcodes(img.affine)
print(f"Axis codes: {axcodes}")
# P=posterior (затылок), A=anterior (перед), S=superior (верх), I=inferior (низ)
# L=left, R=right

# === Bounding box твёрдых тканей ===
threshold = np.percentile(data, 85)
print(f"Threshold (85th percentile): {float(threshold):.1f}")

mask = data > threshold
coords = np.argwhere(mask)
print(f"Bone voxels: {mask.sum()}")

min_c = coords.min(axis=0)
max_c = coords.max(axis=0)
print(f"Bone bbox: {min_c} → {max_c}, size: {max_c - min_c}")

# === Начальный bbox с симметричным padding ===
start = np.maximum(min_c - PAD_KEEP, 0).astype(int)
end = np.minimum(max_c + 1 + PAD_KEEP, np.array(data.shape)).astype(int)

# === Асимметричное подрезание: режем с posterior/superior ===
for axis in range(3):
    code = axcodes[axis]

    if code == 'P':      # + направление = затылок
        end[axis] = min(end[axis], int(max_c[axis]) + PAD_CUT)
    elif code == 'A':    # + направление = перед → затылок на - стороне
        start[axis] = max(start[axis], int(min_c[axis]) - PAD_CUT)
    elif code == 'S':    # + направление = верх
        end[axis] = min(end[axis], int(max_c[axis]) + PAD_CUT)
    elif code == 'I':    # + направление = низ → верх на - стороне
        start[axis] = max(start[axis], int(min_c[axis]) - PAD_CUT)
    # L/R — оставляем симметрично

# === Обрезка до TARGET (с posterior/superior сторон) ===
for axis in range(3):
    code = axcodes[axis]
    n = end[axis] - start[axis]

    if n <= TARGET:
        continue

    excess = n - TARGET
    print(f"Axis {axis} ({code}): {n} → cut {excess}")

    if code in ('P', 'S'):
        # Режем с положительного конца (затылок/макушка)
        end[axis] -= excess
    elif code in ('A', 'I'):
        # Режем с отрицательного конца (затылок/макушка)
        start[axis] += excess
    else:
        # L/R — симметрично
        half = excess // 2
        start[axis] += half
        end[axis] -= (excess - half)

# === Проверка нижней границы ===
# Если после обрезки получилось меньше TARGET — добавляем padding обратно
# (только с разрешённых сторон, не с зубов)
for axis in range(3):
    code = axcodes[axis]
    n = end[axis] - start[axis]

    if n >= TARGET:
        continue

    deficit = TARGET - n
    print(f"Axis {axis} ({code}): {n} < {TARGET}, добавляю {deficit} padding")

    if code in ('P', 'S'):
        end[axis] = min(end[axis] + deficit, data.shape[axis])
    elif code in ('A', 'I'):
        # для передней/нижней стороны — не трогаем, добавляем на противоположной
        pass
    else:
        # L/R — симметрично
        half = deficit // 2
        start[axis] = max(start[axis] - half, 0)
        end[axis] = min(end[axis] + (deficit - half), data.shape[axis])

print(f"\nFinal crop: {start} → {end}")
print(f"Final shape: {tuple(end - start)}")

# === Обрезка ===
slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
cropped = data[slices]

# === Коррекция affine (сдвигаем origin) ===
shift = start.astype(float)
new_affine = img.affine.copy()
new_affine[:3, 3] = new_affine[:3, 3] + new_affine[:3, :3] @ shift

# === Сохранение ===
new_img = nib.Nifti1Image(cropped.astype(np.int16), new_affine, img.header)
new_img.header.set_data_shape(cropped.shape)
nib.save(new_img, OUTPUT)

spacing = img.header.get_zooms()
print(f"\nSaved: {OUTPUT}")
print(f"Physical size: {cropped.shape[0]*spacing[0]:.0f} × "
      f"{cropped.shape[1]*spacing[1]:.0f} × "
      f"{cropped.shape[2]*spacing[2]:.0f} мм")

# === Прогноз после ресемпла TIPs ===
print(f"\nAfter TIPs resample to 0.3 мм:")
for i, (s, sp) in enumerate(zip(cropped.shape, spacing)):
    after = int(s * sp / 0.3)
    marker = "✓" if after <= 600 else "✗ ПРЕВЫШЕНИЕ"
    print(f"  Axis {i}: {s} × {sp} мм → {after} voxels  {marker}")