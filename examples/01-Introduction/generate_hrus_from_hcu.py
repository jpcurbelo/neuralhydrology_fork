import os

DATA_DIR = "../../../../../gladwell/hydrology/SUMMA/summa-ml-models/CAMELS_US/usgs_streamflow"

# FORCINGS = ["daymet", "nldas", "maurer"]


HCUS = ['01', '03', '11', '17']


if __name__ == "__main__":
    
    for hcu in HCUS:

        hcu_dir = os.path.join(DATA_DIR, hcu)
        hrus = sorted(os.listdir(hcu_dir))
        
        hru_ids = []
        for hru in hrus:
            hru_ids.append(hru.split("_")[0])
            
        # Save HRU ids to file
        with open(f"basin_hcu_{hcu}.txt", "w") as f:
            for hru_id in hru_ids:
                f.write(f"{hru_id}\n")