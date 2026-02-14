#set page(width: auto, height: auto, margin: 24pt, fill: rgb("#0d1117"))
#set text(font: "DejaVu Sans", size: 10pt, fill: rgb("#e6edf3"))

#let accent = rgb("#58a6ff")
#let green = rgb("#3fb950")
#let purple = rgb("#bc8cff")
#let orange = rgb("#d29922")
#let pink = rgb("#f778ba")
#let subtle = rgb("#8b949e")
#let surface = rgb("#161b22")
#let border = rgb("#30363d")

#let node(body, col: accent) = {
  box(
    fill: surface,
    stroke: 1pt + col,
    radius: 6pt,
    inset: (x: 10pt, y: 6pt),
    text(fill: col, weight: "bold", size: 9pt, body)
  )
}

#let group-box(title, col, ..children) = {
  let items = children.pos()
  box(
    stroke: 1pt + col.transparentize(60%),
    radius: 8pt,
    inset: (x: 12pt, y: 10pt),
    fill: col.transparentize(92%),
    {
      text(fill: col, weight: "bold", size: 8pt, upper(title))
      v(6pt)
      for (i, child) in items.enumerate() {
        node(child, col: col)
        if i < items.len() - 1 { h(6pt) }
      }
    }
  )
}

#let arrow-right = text(fill: subtle, size: 14pt, weight: "bold")[ #sym.arrow.r ]
#let arrow-down = text(fill: subtle, size: 14pt, weight: "bold")[ #sym.arrow.b ]
#let arrow-lr = text(fill: subtle, size: 14pt, weight: "bold")[ #sym.arrow.l.r ]

#align(center)[
  // Top row: Input -> MedImage -> Operations
  #grid(
    columns: (auto, auto, auto, auto, auto),
    align: (center + horizon),
    column-gutter: 12pt,
    group-box("Input Formats", green, "NIfTI", "DICOM", "HDF5", "MHA"),
    arrow-right,
    // Core MedImage
    box(
      stroke: 2pt + accent,
      radius: 10pt,
      inset: (x: 16pt, y: 12pt),
      fill: accent.transparentize(90%),
      {
        align(center)[
          #text(fill: accent, weight: "bold", size: 12pt)[MedImage]
          #v(6pt)
          #text(fill: subtle, size: 8pt)[
            voxel\_data #sym.dot origin #sym.dot spacing \
            direction #sym.dot metadata
          ]
        ]
      }
    ),
    arrow-right,
    group-box("Operations", purple, "Rotate", "Crop", "Pad", "Scale", "Resample", "Reorient"),
  )

  #v(14pt)

  // Bottom row: Output, Backends, Autodiff
  #grid(
    columns: (auto, 36pt, auto, 36pt, auto),
    align: (center + horizon),
    column-gutter: 0pt,
    group-box("Output", green, "NIfTI", "HDF5"),
    [],
    group-box("Backends", orange, "CPU", "CUDA", "AMD", "oneAPI"),
    [],
    group-box("Autodiff", pink, "Zygote", "Enzyme"),
  )
]
