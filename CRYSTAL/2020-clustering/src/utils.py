""" First Cell contents:
#@markdown ###First Cell 
#@markdown *Please paste this cell into any colab notebook for this project.* <br>
#@markdown ***
#@markdown **Contents**: Contents. <br>
#@markdown ***
#@markdown Major Edit History: 
#@markdown - Author, Date: Created as copy of Old Notebook (`Old Title`).
#@markdown ***
#@markdown <br> 
#@markdown Please change `FIELDDAY_DIR` if it is located differently in your drive. This cell will error if `FIELDDAY_DIR` is incorrect.
# mount drive
from google.colab import drive
drive.mount('/content/drive')

# Change working directory
import os
FIELDDAY_DIR = '/content/drive/My Drive/Field Day' #@param {type:"string"}
JUPYTER_DIR = os.path.join(FIELDDAY_DIR,'Research and Writing Projects/2020 CHI Play - Lakeland Clustering/Jupyter')
os.chdir(JUPYTER_DIR)
print(f'---\nCWD: {os.getcwd()}')

#@markdown Change pandas `max_rows` and `max_columns`
import pandas as pd
pd.options.display.max_columns = 100 #@param {type:"integer"}
pd.options.display.max_rows = 60 #@param {type:"integer"}

#@markdown *Note: There may be other variables to manually change. Look the "Set Variables" section.*

# import utils
import sys
sys.path.append('.')
import utils
"""

"""Save Cell contents:
save = True #@param {type:"boolean"}
savedir = 'Data/Full Data Tables/' #@param ["/", "Data/", "Data/Full Data Tables"] {allow-input: true}
savedir = savedir if os.path.isdir(savedir) else 'Data/Full Data Tables'
savename = '' #@param {type:"string"}
savename = savename or f'Final CM-Cluster Output {DATE}.csv'
savepath = os.path.join(savedir,savename)
if save:
  print(f'Saving to: {savepath}')
else:
  print(f'Not saving.')
"""

TEST_STR = 'hello'


def apply_functions_to_df(df, function_list, verbose=False):
    ret = df
    if verbose:
        print(f'df0 len = {len(ret)}')
    for i, f in enumerate(function_list):
        ret = f(ret)
        if type(ret) is not int and verbose:
            print(f'df{i + 1} len = {len(ret)}')
    return ret


def equal(col, val):
    return lambda df: df.loc[df[col] == val, :]


def match(col, regex, case=True, flags=0):
    return lambda df: df.loc[df[col].str.match(regex, case, flags), :]


def search(col, regex, case=True, flags=0):
    return lambda df: df.loc[df[col].str.contains(regex, case, flags), :]


def search2colsOR(col1, col2, regex):
    return lambda df: df.loc[df[col1].str.contains(regex) | df[col2].str.contains(regex), :]


def searchPair(col1, col2, regex1, regex2):
    return lambda df: df.loc[(df[col1].str.contains(regex1) & df[col2].str.contains(regex2)) |
                             (df[col1].str.contains(regex2) & df[col2].str.contains(regex1)), :]


def sum_col(col):
    return lambda df: df.loc[:, col].sum()


def len_df():
    return lambda df: len(df)


def identity():
    return lambda df: df


"""
Code snippet to add simple file selections to a Jupyter notebook:

base_path = "/content/drive/My Drive/Field Day/Research and Writing Projects/2020 Lakeland EDM/Jupyter/Data/"
folders = [fdir for fdir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, fdir))]
folder_selector = widgets.Select(
    options = folders,
    value = folders[0],
    description = "Folder",
    layout = widgets.Layout(width='80%')
)
file_path = base_path + folder_selector.value
files = [fdir for fdir in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, fdir))]
file_selector = widgets.Select(
    options = files,
    value = files[0],
    description = "File",
    layout = widgets.Layout(width='80%')
)

def updateFolder(change):
  file_path = base_path + folder_selector.value
  files = [fdir for fdir in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, fdir))]
  file_selector.options = files
  file_selector.value = files[0]
folder_selector.observe(updateFolder, names="value")

display(folder_selector, file_selector)
"""


def write_csv_with_meta(df, path, meta_strings, mode='w+'):
    with open(path, mode=mode) as f:
        for l in meta:
            f.write(f'# {l}\n')
        df.to_csv(f)


import os


def init_path():
    import src.settings
    JUPYTER_DIR = os.path.join(src.settings.FIELDDAY_DIR,
                               'Research and Writing Projects/2020 CHI Play - Lakeland Clustering/Jupyter')
    os.chdir(JUPYTER_DIR)
    # print(f'---\nCWD: {os.getcwd()}')


if __name__ == '__main__':
    init_path()
