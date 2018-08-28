"""
Based on the location of this script, recursively convert all
jupyter notebooks to html file.
Potential use case is to host these static html file on
github pages or else where.

Examples
--------
python convert_to_html.py

References
----------
Rendering Jupyter Notebooks (IPython Notebook) on Github Pages Sites
- http://notesofaprogrammer.blogspot.com/2017/02/rendering-jupyter-notebooks-ipython.html
"""
import os
from subprocess import call


def main(base_dir):
    convert_to_html_command = 'jupyter nbconvert --to html --template full {}'

    for root_dir, sub_dir, filenames in os.walk(base_dir):
        # ignore notebook checkpoints
        if '.ipynb_checkpoints' in root_dir:
            continue

        for filename in filenames:
            _, file_extension = os.path.splitext(filename)
            if file_extension == '.ipynb':
                notebook_path = os.path.join(root_dir, filename)
                call(convert_to_html_command.format(notebook_path), shell=True)


if __name__ == '__main__':
    base_dir = '.'
    main(base_dir)
