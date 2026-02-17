#set page(width: auto, height: auto, margin: 20pt, fill: rgb("#0d1117"))
#set text(font: "DejaVu Sans", size: 9pt, fill: rgb("#e6edf3"))

#let accent = rgb("#58a6ff")
#let code-fg = rgb("#79c0ff")
#let subtle = rgb("#8b949e")
#let surface = rgb("#161b22")

#let code(body) = text(font: "DejaVu Sans Mono", size: 8.5pt, fill: code-fg, body)
#let hdr(body) = text(fill: accent, weight: "bold", size: 9pt, body)

#table(
  columns: (auto, auto),
  stroke: none,
  inset: (x: 12pt, y: 7pt),
  fill: (_, row) => if row == 0 { surface } else if calc.odd(row) { rgb("#0d1117") } else { rgb("#121920") },
  hdr[Function], hdr[Description],
  code[load_image(path, type)], [Load NIfTI or DICOM],
  code[create_nii_from_medimage(im, path)], [Export to NIfTI],
  code[save_med_image(im, path)], [Save to HDF5],
  code[load_med_image(path)], [Load from HDF5],
  code[resample_to_spacing(im, spacing, interp)], [Change voxel resolution],
  code[resample_to_image(fixed, moving, interp)], [Align to target geometry],
  code[change_orientation(im, code)], [Reorient voxel axes],
  code[rotate_mi(im, axis, angle, interp)], [3D rotation],
  code[crop_mi(im, start, size, interp)], [Crop with origin adjustment],
  code[pad_mi(im, before, after, val, interp)], [Pad with origin adjustment],
  code[translate_mi(im, offset, axis, interp)], [Translate along axis],
  code[scale_mi(im, factor, interp)], [Uniform scaling],
)
