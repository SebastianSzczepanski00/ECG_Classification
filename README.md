## ECG_Classification

Copyright 2023 by Sebastian Szczepański, Warsaw University of Technology.

This is a part of my engineering thesis. My goal is to classify person id by only ECG features so next step will be replacing data about persons age and sex by FFT spectrum features. First function is responsible for creating dataframe/csv file with persons features. The second one to create neural network and supervised classification. Data source: [PhysioNet ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/).

W tej części pracy założono cel odgadnięcia przez sieć neuronową osoby znajdującej się za kierownicą po zmierzonych cechach sygnału EKG tej osoby. W pierwszej części jest tworzony plik w formacie csv zawierający cechy EKG oraz pewne cechy kierowcy. W drugiej części bazując na tych cechach model sieci neuronowej uczy się i przewiduje, kto znajduje się za kierownicą pojazdu. W dalszej części będą podjęte starania oparcia uczenia się modelu tylko na danych wyekstraktowanych z sygnału EKG. Źródło danych: [PhysioNet ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/).

### Files:
- [ECG_Classification.py](https://github.com/SebastianSzczepanski00/ECG_Classification/blob/main/ECG_Classification.py) - script contains ECG signal processing and first original version of machine learning implementation from the thesis.
- [ECG_Classification_data6.csv](https://github.com/SebastianSzczepanski00/ECG_Classification/blob/main/ECG_Classification_data6.csv) - is the latest csv output file from ECG_Classification.get_files() method, which contains ECG signals attributes processed from raw ECG signals files.
- [Keras_ECG_Classification.ipynb](https://github.com/SebastianSzczepanski00/ECG_Classification/blob/main/Keras_ECG_Classification.ipynb) - script contains machine learning model implementation in Keras, which is better than model used in the thesis. Main reason was mine better understanding of machine learning process and usage common resulotions for the issue.
- [error_list.txt](https://github.com/SebastianSzczepanski00/ECG_Classification/blob/main/error_list.txt) - file contains list with signals, that ECG_Classification.get_files() method wasn't able to process.
- [requirements.txt](https://github.com/SebastianSzczepanski00/ECG_Classification/blob/main/requirements.txt) - pip list with packages and their versions used in the project.
- [ruff.toml](https://github.com/SebastianSzczepanski00/ECG_Classification/blob/main/ruff.toml) - ruff configuration file for ECG_Classification.py script file.
