#set page(width: auto, height: auto, margin: 20pt, fill: rgb("#0d1117"))
#set text(font: "DejaVu Sans", size: 9pt, fill: rgb("#e6edf3"))

#let accent = rgb("#58a6ff")
#let subtle = rgb("#8b949e")
#let code-fg = rgb("#79c0ff")
#let green = rgb("#3fb950")
#let orange = rgb("#d29922")
#let surface = rgb("#161b22")

#let code(body) = text(font: "DejaVu Sans Mono", size: 8.5pt, fill: code-fg, body)
#let hdr(body) = text(fill: accent, weight: "bold", size: 9pt, body)

#table(
  columns: (auto, auto, auto),
  stroke: none,
  inset: (x: 12pt, y: 7pt),
  fill: (_, row) => if row == 0 { surface } else if calc.odd(row) { rgb("#0d1117") } else { rgb("#121920") },
  hdr[Method], hdr[Speed], hdr[Best for],
  code[Nearest_neighbour_en], text(fill: green)[Fast], [Segmentation masks, label maps],
  code[Linear_en], text(fill: orange)[Medium], [General CT / MRI processing],
  code[B_spline_en], text(fill: subtle)[Slow], [Publication figures],
)
