# Additional data preparations
Create supplementary data information that is saved in ../../data/schaefer/ .

## schaefer_accessible_data_overview.py
Overview for which subjects Schaefer timeseries exist: Iterates through downloaded subject directories + creates an entry for each subject ID -- 'True' if Schaefer files exist for this ID, 'False' if they don't. 

--> saved as  *schaefer_exists.csv*

Run with: `$ python3 schaefer_accessible_data_overview.py`

## schaefer_parcellation_names_overview.py
Overview of network names: For each of the chosen network/parcellation combinations (i.e., variants), it creates an overview which networks the parcellations belong to (index -> network name). 

--> saved as *label_names_[VARIANT].csv*

Run with: `$ python3 schaefer_parcellation_names_overview.py`