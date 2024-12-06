# Concatenate all notebooks in a directory into a single notebook
import nbformat
import os


def merge_notebooks(notebooks, output_file):
    merged = nbformat.v4.new_notebook()
    for notebook in notebooks:
        with open(notebook, 'r') as f:
            nb = nbformat.read(f, as_version=4)
            for cell in nb.cells:
                merged.cells.append(cell)
    with open(output_file, 'w') as f:
        nbformat.write(merged, f)
    print(f"Notebooks merged into {output_file}")


def main():
    informe = 'informe_introduccion.ipynb'
    tn_ident = 'tn-identification.ipynb'
    notebooks = [informe, tn_ident]
    output_file = 'informe_completo.ipynb'
    merge_notebooks(notebooks, output_file)


if __name__ == '__main__':
    main()
