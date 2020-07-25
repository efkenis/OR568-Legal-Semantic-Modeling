(TeX-add-style-hook
 "paper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "letterpaper" "margin=1in") ("enumitem" "shortlabels")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "geometry"
    "enumitem"
    "amsmath"
    "graphicx"
    "setspace"
    "titling")
   (LaTeX-add-labels
    "table:LR"
    "table:NN"))
 :latex)

