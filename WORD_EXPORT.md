# Word export

The ICLR submission source remains `paper/iclr2027_conference.tex`. The
repository-root `word_export.tex` is a conversion copy: its dependencies have
root-relative paths, its figures use PNG copies, and its equation numbers and
cross-references are materialized for Word. It is not the submission source.

In Overleaf, select `word_export.tex` as the **Main document**, then use the
Word export. For the normal ICLR PDF, select `paper/main.tex` again. The Word
copy is a snapshot; regenerate it after changing the submission source.

To regenerate locally, first compile `paper/iclr2027_conference.tex` and retain
its `.aux` file. Then run `paper/generate_word_export.ps1` with `-AuxPath` set to
that file and `-PdftoppmPath` set to a Poppler `pdftoppm.exe`. The script updates
only `word_export.tex` and `word_export_figures/*.png`.
