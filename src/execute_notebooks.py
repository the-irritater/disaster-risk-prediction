import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(filepath):
    print(f"--- Running {filepath} ---")
    with open(filepath, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    
    # We set path to the root folder so python imports work
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    with open(filepath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Finished {filepath} successfully!\n")

if __name__ == "__main__":
    # List of active notebooks to run
    notebooks = [
        "notebooks/01_data_generation.ipynb",
        "notebooks/02_master_analysis.ipynb"
    ]
    
    # Clean up old notebooks to satisfy the consolidation request
    old_notebooks = [
        "notebooks/02_data_validation_and_cleaning.ipynb",
        "notebooks/03_exploratory_and_spatial_analysis.ipynb",
        "notebooks/04_risk_index_construction.ipynb",
        "notebooks/05_statistical_analysis.ipynb",
        "notebooks/06_disaster_occurrence_modelling.ipynb",
        "notebooks/07_impact_modelling.ipynb",
        "notebooks/08_explainability_and_clustering.ipynb",
        "notebooks/09_power_bi_export.ipynb"
    ]
    
    for old_nb in old_notebooks:
        if os.path.exists(old_nb):
            try:
                os.remove(old_nb)
                print(f"Removed old notebook: {old_nb}")
            except Exception as e:
                print(f"Could not remove {old_nb}: {e}")

    # Set python path to current directory
    os.environ["PYTHONPATH"] = os.getcwd()
    
    for nb in notebooks:
        try:
            run_notebook(nb)
        except Exception as e:
            print(f"Error executing {nb}: {e}")
            break
