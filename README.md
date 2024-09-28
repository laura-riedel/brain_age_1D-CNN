# Master Thesis: Evaluating a Lightweight 1D-CNN for Brain Age Prediction

_This work fulfills the requirements of the final master thesis (30 ECTS) of the MSc Cognitive Systems at the University of Potsdam. It was conducted in the form of an internship at the [Machine learning in Clinical Neuroimaging Lab](https://psychiatrie-psychotherapie.charite.de/en/research/translation_and_neurotechnology/machine_learning/) ([@ritterlab](https://github.com/ritterlab)) at the Charité Universitätsmedizin Berlin._

In this thesis, I present a novel approach for brain age prediction which uses BOLD activation timeseries from rs-fMRI recordings of the UK Biobank ([Sudlow et al., 2015](https://doi.org/10.1371/journal.pmed.1001779)) as input to a lightweight one-dimensional convolutional neural network (1D-CNN) model.
Two model variations are investigated: a shallow 1D-CNN with rapid feature abstraction and a deep 1D-CNN with slow feature abstraction. Both architectures achieve state-of-the-art performance already, but especially the deep 1D-CNN has the potential to improve its performance even further through additional model optimisation.
The lightweight models seem to learn meaningful features in relation to functional networks and specific brain areas which are documented in a body of neuroscientific literature. 
Additionally, explainability analyses indicate that there might be indicative age-dependent temporal variance in the BOLD signal that the proposed models picked up on. Overall, using lightweight 1D-CNNs to predict brain age based on rs-fMRI activation timeseries seems to be a promising approach which warrants further investigation in the future.

## Structure
- **data:** doesn't contain the data itself but supplementary files.
- **src:** all things code; further divided into my brain_age_prediction package, data preparations, and different experimental settings.
- **viz:** contains visualisations for the written thesis.

More detailed information can be found in the directories themselves.

## Requirements
You should be able to create a working environment using the provided environment files.

e.g. with conda: `conda env create -n ENVNAME --file environment.yml`
(If no ENVNAME is provided, the envirnoment name will default to brain_age_1DCNN.)

You might need to add this repository's src directory to your package/module paths so you can use ukbb_package properly. E.g.:
`conda develop /path/to/repo/brain_age_1D-CNN/src/`