# Deep BCI SW ver. 1.0 is released.
An open software package to develop Brain-Computer Interface (BCI) based brain and cognitive computing technology for recognizing user's intention using deep learning


Website: http://deepbci.korea.ac.kr/


We provide detailed information in each forder and every function.

1. 'Intelligent_BCI': contains deep learning-based intelligent brain-computer interface-related function that enables high-performance intent recognition.

- Domain Adversarial NN for BCI: functions related to domain adversarial neural networks
- EEG based Meta RL Classifier: functions related to model-based reinforcement learning
- GRU based Large Size EEG Classifier: data and functions related to gated recurrent unit
- etc

2. 'Ambulatory_BCI': contains general brain-computer interface-related functions that enable high-performance intent recognition in ambulatory environment

- Channel Selection Method based on Relevance Score: functions related to electrode selection method by evaluating electrode's contribution to motor imagery based on relevance score and CNNs
- Correlation optimized using rotation matrix: functions related to cognitive imagery analysis using correlation feature
- SSVEP decoding in ambulatory environment using CNN: functions related to decoding scalp- and ear-EEG in ambulatory environment
- etc

3. 'Cognitive_BCI': contains cognitive state-related function that enable to estimate the cognitive states from multi-modality and user-custermized BCI

- multi-threshold graph metrics using a range of critiera: functions related to entrain brainwaves based on a combined auditory stimulus with a binaural beat
- EEG_Authentication_Program: identifying individuals based on resting-state EEG
- Ear_EEG_Drowsiness_Detection: identifying individuals based on resting-state EEG using convolutional neural network
- etc

4. 'Zero-Training_BCI': contains zero-training brain-computer interface-related functions that enable to minimize additional training
- ERP-based_BCI_Algorithm_for_Zero_Training: functions related to Event Related Potential (ERP) analysis including feature extraction, classification, and visualization
- SSVEP_based_Mind_Mole_Catching: functions allowing users to play mole cathcing game using their brain activity on single/two-player mode
- SSVEP_based_BCI_speller: functions related to SSVEP-based speller containing nine classes
- etc

Acknowledgement: This project was supported by Institute for Information & Communications Technology Promotion (IITP) grant
funded by the Korea government(No. 2017-0-00451, Development of BCI based Brain and Cognitive Computing Technology for Recognizing User’s Intentions using Deep Learning).
