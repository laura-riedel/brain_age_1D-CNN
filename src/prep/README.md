# Additional data preparations
Create supplementary data information that is saved in ../../data/schaefer/ (except for `create_fc_matrices.py` and `add_split_shortcuts.py` which save to /ritter/share/projects/laura_riedel_thesis/).

## schaefer_accessible_data_overview.py
Overview for which subjects Schaefer timeseries exist: Iterates through downloaded subject directories + creates an entry for each subject ID -- 'True' if Schaefer files exist for this ID, 'False' if they don't. 

--> saved as  *[VARIANT]/schaefer_exists.csv* and *[VARIANT]/empty_files.csv*

Run with: `$ python3 schaefer_accessible_data_overview.py`

## schaefer_parcellation_names_overview.py
Overview of network names: For each of the chosen network/parcellation combinations (i.e., variants), it creates an overview which networks the parcellations belong to (index -> network name). 

--> saved as *[VARIANT]/label_names_[VARIANT].csv*

Run with: `$ python3 schaefer_parcellation_names_overview.py`

## schaefer_zero_ts_overviews.py
Create overviews: for each Schaefer variant, which parcellations contain only 0-timeseries, and how often?

--> saved as *[VARIANT]/zero_ts_overview.csv*

Run with: `$ python3 schaefer_zero_ts_overviews.py`

## create_fc_matrices.py
Iterate through all known UKBB subjects. If they have Schaefer parcellation timeseries,
calculate the functional connectivity (FC) matrix and save the upper triangle as HDF5 file.

--> saved as *schaefer_fc_matrices.hdf5*

Run with: `$ python3 create_fc_matrices.py`

## define_heldout_test_set.py
Sample subject IDs randomly to use as held-out final test set and save the list.

--> saved with terminal-determined name, default: *heldout_test_set.csv*

Run with: `$ python3 define_heldout_test_set.py`

## add_split_shortcuts.py
Add train/val/test set shortcuts to HDF5 data file for FC matrices.
Creates those shortcuts for the defined split + variants of interest.

--> saved in *schaefer_fc_matrices.hdf5*

Run with: `$ python3 add_split_shortcuts.py [OPTIONS]`