import nibabel as nib
import numpy as np
import os
import sys

# === Аргументы ===
if len(sys.argv) < 3:
    print("Usage: crop_smart.py <input.nii.gz> <output.nii.gz>")
    sys.exit(1)

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

TARGET = 600
PAD_KEEP = 40
PAD_CUT = 5

print(f"Input:  {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Target: {TARGET} voxels per axis")

img = nib.load(INPUT)
data = img.get_fdata(dtype=np.float32)
print(f"Original shape: {data.shape}")
print(f"Spacing: {img.header.get_zooms()}")

axcodes = nib.aff2axcodes(img.affine)
print(f"Axis codes: {axcodes}")

threshold = np.percentile(data, 85)
print(f"Threshold: {float(threshold):.1f}")

mask = data > threshold
coords = np.argwhere(mask)
print(f"Bone voxels: {mask.sum()}")

min_c = coords.min(axis=0)
max_c = coords.max(axis=0)
print(f"Bone bbox: {min_c} → {max_c}, size: {max_c - min_c}")

start = np.maximum(min_c - PAD_KEEP, 0).astype(int)
end = np.minimum(max_c + 1 + PAD_KEEP, np.array(data.shape)).astype(int)

for axis in range(3):
    code = axcodes[axis]
    if code == 'P':
        end[axis] = min(end[axis], int(max_c[axis]) + PAD_CUT)
    elif code == 'A':
        start[axis] = max(start[axis], int(min_c[axis]) - PAD_CUT)
    elif code == 'S':
        end[axis] = min(end[axis], int(max_c[axis]) + PAD_CUT)
    elif code == 'I':
        start[axis] = max(start[axis], int(min_c[axis]) - PAD_CUT)

for axis in range(3):
    code = axcodes[axis]
    n = end[axis] - start[axis]
    if n <= TARGET:
        continue
    excess = n - TARGET
    print(f"Axis {axis} ({code}): {n} → cut {excess}")
    if code in ('P', 'S'):
        end[axis] -= excess
    elif code in ('A', 'I'):
        start[axis] += excess
    else:
        half = excess // 2
        start[axis] += half
        end[axis] -= (excess - half)

for axis in range(3):
    code = axcodes[axis]
    n = end[axis] - start[axis]
    if n >= TARGET:
        continue
    deficit = TARGET - n
    print(f"Axis {axis} ({code}): {n} < {TARGET}, padding {deficit}")
    if code in ('P', 'S'):
        end[axis] = min(end[axis] + deficit, data.shape[axis])
    elif code in ('L', 'R'):
        half = deficit // 2
        start[axis] = max(start[axis] - half, 0)
        end[axis] = min(end[axis] + (deficit - half), data.shape[axis])

print(f"Final crop: {start} → {end}")
print(f"Final shape: {tuple(end - start)}")

slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
cropped = data[slices]

shift = start.astype(float)
new_affine = img.affine.copy()
new_affine[:3, 3] = new_affine[:3, 3] + new_affine[:3, :3] @ shift

new_img = nib.Nifti1Image(cropped.astype(np.int16), new_affine, img.header)
new_img.header.set_data_shape(cropped.shape)
nib.save(new_img, OUTPUT)

spacing = img.header.get_zooms()
print(f"Saved: {OUTPUT}")
print(f"Physical size: {cropped.shape[0]*spacing[0]:.0f} × {cropped.shape[1]*spacing[1]:.0f} × {cropped.shape[2]*spacing[2]:.0f} мм")

print(f"\nAfter TIPs resample to 0.3 мм:")
for i, (s, sp) in enumerate(zip(cropped.shape, spacing)):
    after = int(s * sp / 0.3)
    marker = "✓" if after <= 600 else "✗ ПРЕВЫШЕНИЕ"
    print(f"  Axis {i}: {s} × {sp} мм → {after} voxels  {marker}")