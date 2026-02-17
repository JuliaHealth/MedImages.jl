#set page(width: auto, height: auto, margin: 20pt, fill: rgb("#0d1117"))
#set text(font: "DejaVu Sans", size: 9pt, fill: rgb("#e6edf3"))

#let accent = rgb("#58a6ff")
#let subtle = rgb("#8b949e")
#let code-fg = rgb("#79c0ff")
#let surface = rgb("#161b22")

#let code(body) = {
  text(font: "DejaVu Sans Mono", size: 8.5pt, fill: code-fg, body)
}

#let hdr(body) = text(fill: accent, weight: "bold", size: 9pt, body)

#table(
  columns: (auto, auto, auto),
  stroke: none,
  inset: (x: 12pt, y: 7pt),
  fill: (_, row) => if row == 0 { surface } else if calc.odd(row) { rgb("#0d1117") } else { rgb("#121920") },
  hdr[Field], hdr[Type], hdr[Description],
  code[voxel_data], text(fill: subtle)[Array], [N-dimensional image array],
  code[origin], code[NTuple\{3,Float64\}], [World-space position of first voxel (mm)],
  code[spacing], code[NTuple\{3,Float64\}], [Voxel size per axis (mm)],
  code[direction], code[NTuple\{9,Float64\}], [3x3 direction cosines (row-major)],
  code[image_type], code[Image_type], [#code[MRI_type] #text(fill:subtle)[|] #code[PET_type] #text(fill:subtle)[|] #code[CT_type]],
  code[image_subtype], code[Image_subtype], [#code[T1_subtype]#text(fill:subtle)[,] #code[CT_subtype]#text(fill:subtle)[, ...]],
  code[patient_id], code[String], [Patient identifier],
  code[current_device], code[current_device_enum], [#code[CPU_current_device]#text(fill:subtle)[,] #code[CUDA_current_device]],
)
