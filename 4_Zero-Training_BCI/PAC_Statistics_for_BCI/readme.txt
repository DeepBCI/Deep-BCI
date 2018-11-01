
* Programming Language: MATLAB
* Contact: Kyungho Won (kyunghowon0712@gist.ac.kr)


These scripts allow users to understand and modify easily when calculating phase-amplitude coupling (PAC) and corresponding statistics. There are many opensource scripts related to PAC and statistics, but most of them are too complex to read and modify code. The proposed scripts contains various PAC methods, permutation test, and multiple comparison correction methods (Bonferrnoi, cluster, min-max correction). In addition, it contatins example script of using PAC in case of detection and comparing conditions so that users could understand which statistics should be used for each analysis method (e.g. between group/condition, none).

It uses external dependencis: EEGLAB, FieldTrip toolbox, which are most famous EEG processing toolbox. (EEGLAB for drawing scalp topography, FieldTrip for zero-phase lag IIR bandpass filter))

Please note that this code was modified, user-friendly re-coded, and added custom-built statistics from the original code written and maintained by Robert Seymour, June 2017 for verifying this script.

See https://github.com/neurofractal/sensory_PAC

CAUTION: This script does not contain data pre-processing steps, so users
should pre-process data before using the script.

For any code, you can get detailed information by using "help "
e.g. >>help kh_detect_PAC

[How to use]

1. unzip "external_dependencies.zip"
2. unzip "kh_subsets.zip"
3. unzip "samples_comparison/sub01-08.zip" and "samples_comparison/sub09-16"
2. copy address of eeglab14_1_2b and fieldtrip-20161112
3. paste into eeglab_path and fieldtrip_path in  ".m"
4. run "add_dependencies.m"

[Example code - detecting PAC]
run kh_detect_PAC_example.m

[Example code - comparing PAC comodulation]
run kh_calc_PAC_comodulation.m
CAUTION: it takes about 3 hours

[Example code - multiple comparison correction of PAC]
run kh_multiple_comparison_example.m



