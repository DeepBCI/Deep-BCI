# Mobile BCI Data

Example codes for "Mobile BCI dataset of scalp- and ear-EEGs with ERP and SSVEP paradigms while standing, walking, and running."

The dataset is available in Open Science Framework repository (https://doi.org/10.17605/OSF.IO/R7S9B) under the terms of Attribution 4.0 International Creative Commons License (http://creativecommons.org/licenses/by/4.0/).


## Description
A mobile dataset obtained from electroencephalography (EEG) of the scalp and around the ear as well as from locomotion sensors by 24 participants moving at four different speeds while performing two brain-computer interface (BCI) tasks. The data were collected from 32-channel scalp-EEG, 14-channel ear-EEG, 4-channel electrooculography, and 9-channel inertial measurement units placed at the forehead, left ankle, and right ankle.

## Code List
- load_all_data: loading the dataset of all subjects, modalities, and speeds for each paradigm using BBCI toolbox. (Matlab)
- evaluation_ERP_using_LDA: Evaluating the dataset of ERP for all subjects and speeds using Linear Discriminator Analysis (LDA). (Matlab)
- evaluation_SSVEP_using_CCA: Evaluating the dataset of SSVEP for all subjects and speeds using Canonical Correlation Analysis (CCA). (Matlab)

### Developing Environment
- Matlab 2019b
- BBCI toolbox (https://github.com/bbci/bbci_public)

Contact:
Young-Eun Lee
ye_lee@korea.ac.kr
