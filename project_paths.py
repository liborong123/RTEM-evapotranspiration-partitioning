import os,shutil
from pathlib import Path
CODE=Path(__file__).resolve().parent
DATA=Path(os.environ.get('RTEM_DATA_ROOT',str(CODE.parent/'data')))
RESULTS=Path(os.environ.get('RTEM_RESULTS_ROOT',str(CODE.parent/'results')))
def working_copy(relative):
    source=DATA/relative;target=RESULTS/relative
    if source.resolve()==target.resolve():
        raise ValueError('Results directory must differ from the archived data directory')
    target.mkdir(parents=True,exist_ok=True)
    for p in source.rglob('*'):
        q=target/p.relative_to(source)
        if p.is_file() and not q.exists():
            q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    return target
def prepared_input():
    return Path(os.environ.get('RTEM_PREPARED_INPUT',str(DATA/'01_processed_inputs/RF_ready/current_preprocessed_template.nc')))
def final_result():
    explicit=os.environ.get('RTEM_RESULT_FILE')
    if explicit:return Path(explicit)
    root=DATA/'02_final_RF_outputs'
    for name in ['rtem_result.nc','tera_result_11vars.nc']:
        if (root/name).exists():return root/name
    raise FileNotFoundError('Final RTEM NetCDF was not found in data/02_final_RF_outputs')
