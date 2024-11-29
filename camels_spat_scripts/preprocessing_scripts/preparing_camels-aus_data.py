from pathlib import Path
import sys

# Get the current working directory of the script
current_dir = Path(__file__).resolve().parent
# Add the parent directory of the notebook to the Python path
root_dir = Path(current_dir).resolve().parents[1]

sys.path.append(root_dir)

from neuralhydrology.datasetzoo.camelsaus import preprocess_camels_aus_dataset

# Define the path to the raw CAMELS_AUS dataset
data_dir = Path("/gladwell/hydrology/SUMMA/summa-ml-models/CAMELS_AUS/")

# Call the preprocessing function
try:
    preprocess_camels_aus_dataset(data_dir)
except FileExistsError as e:
    print(f"Error: {e}")