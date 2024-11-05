## ECG_Classification

Copyright 2023 by Sebastian Szczepański, Warsaw University of Technology.

This is a part of my engineering thesis. My goal is to classify person id by only ECG features so next step will be replacing data about persons age and sex by FFT spectrum features. First function is responsible for creating dataframe/csv file with persons features. The second one to create neural network and supervised classification. Data source: [PhysioNet ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/).

W tej części pracy założono cel odgadnięcia przez sieć neuronową osoby znajdującej się za kierownicą po zmierzonych cechach sygnału EKG tej osoby. W pierwszej części jest tworzony plik w formacie csv zawierający cechy EKG oraz pewne cechy kierowcy. W drugiej części bazując na tych cechach model sieci neuronowej uczy się i przewiduje, kto znajduje się za kierownicą pojazdu. W dalszej części będą podjęte starania oparcia uczenia się modelu tylko na danych wyekstraktowanych z sygnału EKG. Źródło danych: [PhysioNet ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/).
