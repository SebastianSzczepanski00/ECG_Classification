# Copyright 2023 by Sebastian Szczepański, Warsaw University of Technology.

# This is a part of my engineering thesis. My goal is to classify person id by only ECG features so next step will be replacing data about persons age and
# sex by FFT spectrum features. First function is responsible for creating dataframe/csv file with persons features. The second one to create neural
# network and supervised classification. Data source: https://physionet.org/content/ecgiddb/1.0.0/.

# W tej części pracy założono cel odgadnięcia przez sieć neuronową osoby znajdującej się za kierownicą po zmierzonych cechach sygnału EKG tej osoby.
# W pierwszej części jest tworzony plik w formacie csv zawierający cechy EKG oraz pewne cechy kierowcy. W drugiej części bazując na tych cechach
# model sieci neuronowej uczy się i przewiduje, kto znajduje się za kierownicą pojazdu. W dalszej części będą podjęte starania oparcia uczenia się
# modelu tylko na danych wyekstraktowanych z sygnału EKG. Źródło danych: https://physionet.org/content/ecgiddb/1.0.0/.

import os
import statistics

import neurokit2 as nk
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.neural_network
import wfdb
from pyecg import ECGRecord
from scipy.fft import rfft, rfftfreq
from scipy.signal import argrelextrema, butter, find_peaks, lfilter

catalog_dir = r"D:\ecg-id-database-1.0.0"
saving_data_dir = r"C:\Users\sebex\Desktop\Python_Projects\ECG_Classification_project/ECG_Classification_data7.csv"

def get_files(catalog_dir: str, saving_data_dir: str) -> None:
    """Create csv file from catalogs with ECG signal data.

    Parameters
    ----------
    catalog_dir: str
    Directory to get catalog with numerated catalogs with hea files.
    saving_data_dir: str
    Directory to save csv file with ECG features.

    """
    file = open("error_list.txt", "w")  # text file with error messages and wrong processed ECG signals / plik tekstowy z błędami oraz przypisanymi im
    # nieprawidłowo przetworzonymi sygnałami EKG
    Classification_ECG_data = pd.DataFrame(columns=["PR", "RT", "PR/RT", "P", "Q", "R", "S", "T", "Age", "Sex", "Person"])
    print(Classification_ECG_data)
    index = 0
    for i in os.listdir(catalog_dir):
        person_nr = i[-1] if i[-2] == "0" else i[-2:]
        person=os.path.join(catalog_dir,i)
        for j in os.listdir(person):
            print("Signal nr:", index, "Exploring file:", j)
            if j[-3:] == "hea":
                record = ECGRecord.from_wfdb(os.path.join(person, j))
                # lead_name = record.lead_names # signal names in .hea file/ nazwy sygnałów zawartych w pliku .hea
                labels = wfdb.rdheader(os.path.join(person, j)[:-4]) # exctracting sampling frequency, age and sex from hea file /
                # ekstraktowanie z pliku .hea częstotliwości próbkowania, wieku i płci badanego
                age = str(labels.comments[0]).replace("Age: ", "")
                sex = str(labels.comments[1]).replace("Sex: ", "")
                f_samp = labels.fs # number of samples / ilosc probek na sekunde
                T = len(record) / (f_samp) # total time in seconds / calkowity czas w sekundach T=n/fs
                # dt = 1 / f_samp # time step / krok czasowy
                # t = np.arange(0, T, dt) # time table / tablica czasowa
                # f = np.arange(0.01, f_samp / 2, 0.01) # frequency table/ tablica czestotliwosciowa
                signal = record.get_lead('I')

                # Clean data with bandpass filter / Czyszczenie danych filtrem typu bandpass

                nyq = 0.5* f_samp # signal frequency 2 times less than sampling frequency/ częstotliwość sygnału EKG 2 razy mniejsza od
                # częstotliwośći próbkowania
                low_cut = 1 # dla 0.5 i 0.05 sygnał "pływa", więcej dałoby "gładszy" charakter sygnału, ale zabrałoby część informacji
                high_cut = 40
                low = low_cut / nyq
                high = high_cut / nyq
                b, a = butter(N=2, Wn=[low, high], btype="bandpass")  # N - rząd filtru, a, b - składowe wektory współczynników
                y = lfilter(b, a, signal)

                # Find R peaks/ Znajdowanie peaków R
                try:
                    rpeaks, _ = find_peaks(y, prominence=0.6) # the function is finding local maximums
                    rpeaks = list(rpeaks)
                    del rpeaks[rpeaks == 'nan']
                    del rpeaks[rpeaks == 'NaN']
                    rpeaks = np.array(rpeaks)
                    # t_rpeaks = rpeaks / f_samp

                # Find other characteristic peaks/ Odnajdywanie innych charakterystycznych peaków
                    _, waves_peak = nk.ecg_delineate(y, rpeaks, sampling_rate=f_samp, method="peak", show=False, show_type='peaks')
                    del waves_peak['ECG_P_Peaks'][waves_peak['ECG_P_Peaks'] == 'nan']
                    del waves_peak['ECG_Q_Peaks'][waves_peak['ECG_Q_Peaks'] == 'nan']
                    del waves_peak['ECG_S_Peaks'][waves_peak['ECG_S_Peaks'] == 'nan']
                    del waves_peak['ECG_T_Peaks'][waves_peak['ECG_T_Peaks'] == 'nan']
                    # t_p_peaks = np.array(waves_peak['ECG_P_Peaks']) / f_samp
                    # t_q_peaks = np.array(waves_peak['ECG_Q_Peaks']) / f_samp
                    # t_s_peaks = np.array(waves_peak['ECG_S_Peaks']) / f_samp
                    # t_t_peaks = np.array(waves_peak['ECG_T_Peaks']) / f_samp
                
                # Take PR and RT intervals durations/ Ekstraktowanie czasu trawania odcinków PR i RT
                    PR_intervals = []
                    RT_intervals = []

                    if waves_peak['ECG_P_Peaks'][0] < rpeaks[0]: # sprawdzamy, który peak występuje jako pierwszy
                        for i in range(len(rpeaks)):
                            PR_interval = rpeaks[i] - waves_peak['ECG_P_Peaks'][i]
                            PR_intervals.append(PR_interval)
                    else:
                        for i in range(len(rpeaks)-1):
                            PR_interval=rpeaks[i+1]-waves_peak['ECG_P_Peaks'][i]
                            PR_intervals.append(PR_interval)
                    
                    if rpeaks[0] < waves_peak['ECG_T_Peaks'][0]:
                        for i in range(len(rpeaks) - 1):
                            RT_interval = waves_peak['ECG_T_Peaks'][i] - rpeaks[i]
                            RT_intervals.append(RT_interval)
                    else:
                        for i in range(len(rpeaks) - 1):
                            RT_interval = waves_peak['ECG_P_Peaks'][i + 1] - rpeaks[i]
                            RT_intervals.append(RT_interval)

                    PR = statistics.median(PR_intervals)
                    RT = statistics.median(RT_intervals)

                except ValueError:
                    print("ValueError", print(i, j))
                    file.write("ValueError\n")
                    file.write(str(i))
                    file.write(str(j))
                    continue

                except ZeroDivisionError:
                    print("ZeroDivisionError", print(i, j))
                    file.write("ZeroDivisionError\n")
                    file.write(str(i))
                    file.write(str(j))
                    continue

                except IndexError:
                    print("IndexError", print(i, j))
                    file.write("IndexError\n")
                    file.write(str(i))
                    file.write(str(j))
                    continue

                try:
                    P = np.mean(y[waves_peak['ECG_P_Peaks']])
                except Exception:
                    lenght = len(waves_peak['ECG_P_Peaks'])
                    try:
                        for ind in range(lenght):
                            if type(waves_peak['ECG_P_Peaks'][ind]) != 'numpy.int64':  # noqa: E721
                                del waves_peak['ECG_P_Peaks'][ind] 
                    except IndexError:
                        break

                    P = np.mean(y[waves_peak['ECG_P_Peaks']])
                
                try:
                    Q = np.mean(y[waves_peak['ECG_Q_Peaks']])
                except Exception:
                    del waves_peak['ECG_Q_Peaks'][list(waves_peak['ECG_Q_Peaks']).index(np.nan)]
                    Q = np.mean(y[waves_peak['ECG_Q_Peaks']])
                
                try:
                    R = np.mean(y[rpeaks])
                except Exception:
                    del rpeaks[list(rpeaks).index(np.nan)]
                    R = np.mean(y[rpeaks])
                
                try:
                    S = np.mean(y[waves_peak['ECG_S_Peaks']])
                except Exception:
                    del waves_peak['ECG_S_Peaks'][list(waves_peak['ECG_S_Peaks']).index(np.nan)]
                    S = np.mean(y[waves_peak['ECG_S_Peaks']])
                
                try:
                    T = np.mean(y[list(waves_peak['ECG_T_Peaks'])])
                except Exception:
                    # del waves_peak['ECG_T_Peaks'][list(waves_peak['ECG_T_Peaks']).index(np.nan)]
                    lenght = len(waves_peak['ECG_T_Peaks'])
                    try:
                        for ind in range(lenght):
                            if type(waves_peak['ECG_T_Peaks'][ind]) != "numpy.int64":  # noqa: E721
                                del waves_peak['ECG_T_Peaks'][ind] 
                    except IndexError:
                        break
                    T = np.mean(y[waves_peak['ECG_T_Peaks']])
                    
                # FFT
                hamming_window = np.hamming(len(y))
                # n = len(y)
                hamming_window = hamming_window / np.linalg.norm(hamming_window) # np.linalg.norm() normalizowanie wektora/macierzy
                y = y * hamming_window 
                Y = rfft(y) # funkcja rfft() oblicza widmo rzeczywistego sygnału
                pwr = Y * Y.conj() # Oblicz moc jako iloczyn unormowanej transformaty i jej sprzężenia zespolonego. 
                pwr = pwr / f_samp # Unormuj widmo dzieląc przez częstość próbkowania
                pwr = pwr.real # Do dalszych operacji wybierz tylko część rzeczywistą mocy. 
                #rfft obliczaja transformate dla dodatnniej czesci osi
                if len(y) % 2 == 0: # dokładamy moc z ujemnej części widma 
                    pwr[1:-1] *= 2

                else:
                    pwr[1:] *= 2

                F = rfftfreq(len(y), 1 / f_samp) # oblicza częstotliwość dla każdego punktu widmowego, argumentami funkcji są: liczba wartości widma
                # (długość transformaty) oraz okres próbkowania.
                cut = int(len(F) * high_cut / (f_samp / 2))
                m = argrelextrema(pwr[:cut], np.greater) #array of indexes of the locals maxima
                f_m = []
                # max_sorted_f = []
                max_1 = 0
                max_1_i = 0
                max_2 = 0
                max_2_i = 0
                max_3 = 0
                max_3_i = 0
                for i in range(len(pwr[:cut])):
                    if pwr[i] in pwr[m]:
                        f_m.append(i)
                        if pwr[i] >= max_1:
                            max_3 = max_2
                            max_3_i = max_2_i
                            max_2 = max_1
                            max_2_i = max_1_i
                            max_1 = pwr[i]
                            max_1_i = i

                        elif pwr[i] >= max_2:
                            max_3 = max_2
                            max_3_i = max_2_i
                            max_2 = pwr[i]
                            max_2_i = i

                        elif pwr[i] >= max_3:
                            max_3 = pwr[i]
                            max_3_i = i

                # fft_max_p = [max_1, max_2, max_3]
                fft_max_f = [F[max_1_i], F[max_2_i], F[max_3_i]]
                # Sorting by frequency
                if max(fft_max_f) == F[max_1_i]:
                    first_f = fft_max_f.pop(fft_max_f.index(F[max_1_i]))
                    first_pwr = max_1

                elif max(fft_max_f) == F[max_2_i]:
                    first_f = fft_max_f.pop(fft_max_f.index(F[max_2_i]))
                    first_pwr = max_2

                elif max(fft_max_f) == F[max_3_i]:
                    first_f = fft_max_f.pop(fft_max_f.index(F[max_3_i]))
                    first_pwr = max_3

                if max(fft_max_f) == F[max_1_i]:
                    second_f = fft_max_f.pop(fft_max_f.index(F[max_1_i]))
                    second_pwr = max_1

                elif max(fft_max_f) == F[max_2_i]:
                    second_f = fft_max_f.pop(fft_max_f.index(F[max_2_i]))
                    second_pwr = max_2

                elif max(fft_max_f) == F[max_3_i]:
                    second_f = fft_max_f.pop(fft_max_f.index(F[max_3_i]))
                    second_pwr = max_3

                if max(fft_max_f) == F[max_1_i]:
                    third_f = fft_max_f.pop(fft_max_f.index(F[max_1_i]))
                    third_pwr = max_1

                elif max(fft_max_f) == F[max_2_i]:
                    third_f = fft_max_f.pop(fft_max_f.index(F[max_2_i]))
                    third_pwr = max_2

                elif max(fft_max_f) == F[max_3_i]:
                    third_f = fft_max_f.pop(fft_max_f.index(F[max_3_i]))
                    third_pwr = max_3

                # Add data to pandas dataframe / dodawanie danych do dataframe
                Classification_ECG_data.loc[index,'PR'] = PR
                Classification_ECG_data.loc[index,'RT'] = RT
                Classification_ECG_data.loc[index,'Age'] = age
                Classification_ECG_data.loc[index,'Sex'] = sex
                Classification_ECG_data.loc[index,'Person'] = person_nr
                Classification_ECG_data.loc[index,"PR/RT"] = PR / RT if RT != 0 else "NaN"
                Classification_ECG_data.loc[index,'P'] = P
                Classification_ECG_data.loc[index,'Q'] = Q
                Classification_ECG_data.loc[index,'R'] = R
                Classification_ECG_data.loc[index,'S'] = S
                Classification_ECG_data.loc[index,'T'] = T
                Classification_ECG_data.loc[index,'FFT_max_1_Pwr'] = first_pwr
                Classification_ECG_data.loc[index,'FFT_max_1_F'] = first_f
                Classification_ECG_data.loc[index,'FFT_max_2_Pwr'] = second_pwr
                Classification_ECG_data.loc[index,'FFT_max_2_F'] = second_f
                Classification_ECG_data.loc[index,'FFT_max_3_Pwr'] = third_pwr
                Classification_ECG_data.loc[index,'FFT_max_3_F'] = third_f
                index += 1

    print(Classification_ECG_data)
    print(Classification_ECG_data["Person"].value_counts())
    Classification_ECG_data.dropna(inplace=True)
    file.close()
    Classification_ECG_data.to_csv(saving_data_dir)

# ............................................. Machine Learning Part ......................................

def Machine_Learning_part(data_dir: str) -> None:
    """Include neural network to train and test ECG features data which is trained to predict persons id by these features.

    Parameters
    ----------
    data_dir: str
    Directory to data to be computed.

    """
    dt = pd.read_csv(data_dir, index_col="Unnamed: 0")
    dt = dt.iloc[:52]
    dt["Sex"] = dt["Sex"].apply(lambda x: 0 if x == "male" else 1)
    print(dt.head(50))
    print("Amount of signals", len(dt))
    Persons = dt['Person']
    Individuals = list(dict.fromkeys(Persons))
    print("Individuals", Individuals)

    # Removing single ECG signals per person/ usuwanie pojednyczych sygnałów EKG przypadających na jedną osobę
    catalogs = []
    for i in range(len(dt)):
        catalogs.append(dt['Person'].iloc[i])
    for i in range(len(catalogs)):
        if catalogs.count(catalogs[i]) == 1:
            dt.drop(dt.loc[dt["Person"] == catalogs[i]].index, axis=0, inplace=True)
    # dt["FFT_F"] = (dt["FFT_max_1_F"] + dt["FFT_max_2_F"] + dt["FFT_max_3_F"]) / 3
    dt["FFT_Pwr"] = (dt["FFT_max_1_Pwr"] + dt["FFT_max_2_Pwr"] + dt["FFT_max_3_Pwr"]) / 3
    
    X = dt.drop(["FFT_max_2_F", "FFT_max_3_F", "FFT_max_2_Pwr", "FFT_max_3_Pwr", "Sex", "Age", "Person"], axis=1)

    # Investigate another possibilities
    # X = dt.drop(["FFT_max_1_F","FFT_max_2_F","FFT_max_3_F","FFT_max_1_Pwr","FFT_max_2_Pwr","FFT_max_3_Pwr","Sex","Age",'Person'],axis=1)
    # X = dt.drop(["PR/RT","P","Q","R","S","T",'Person'],axis=1)
    # X = dt.drop(["P","Q","R","S","T",'Person'],axis=1)
    y = dt["Person"]
    print("Features", X.columns)
    # random splitting data to train and test data/ losowe dzielenie danych na uczące i testowe
    print(dt.head(52))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=40, shuffle=True)

    # supervised splitting data to train and test data giving oppurnity to train the model with minimum one ECG signal per person/
    # kontrolowane dzielenie danych na uczące i testowe dając możliwość nauki przynajmniej jednego sygnłu na osobę
    
    dt_copy = dt.copy()
    # dt_copy = dt_copy.drop(["FFT_max_2_F","FFT_max_3_F","FFT_max_2_Pwr","FFT_max_3_Pwr","Sex","Age"],axis=1)
    dt_copy = dt_copy.drop(["FFT_max_1_Pwr", "FFT_max_2_Pwr", "FFT_max_3_Pwr", "Sex", "Age"], axis=1)
    # dt_copy = dt_copy.drop(["Sex","Age"], axis=1)
    catalogs = []

    for i in range(len(dt)):
        catalogs.append(dt['Person'].iloc[i])

    catalogs = list(dict.fromkeys(catalogs))
    dt_train = pd.DataFrame()
    dt_test = pd.DataFrame()

    for i in catalogs:
        dt_train = pd.concat([dt_train,dt_copy.loc[dt_copy["Person"] == i].head(1)], ignore_index=True) # min 1 signal per person to train data
        dt_copy.drop(dt_copy.loc[dt_copy["Person"] == i].head(1).index, axis=0, inplace=True)
        dt_test = pd.concat([dt_test, dt_copy.loc[dt_copy["Person"] == i].head(1)], ignore_index=True) # min 1 signal per person to test data
        dt_copy.drop(dt_copy.loc[dt_copy["Person"] == i].head(1).index, axis=0, inplace=True)

    X_train_sv = dt_train.drop("Person", axis=1) # supervised X and y splitting/ kontrolowany podział danych na X i y
    y_train_sv = dt_train["Person"]
    X_test_sv = dt_test.drop("Person", axis=1)
    y_test_sv = dt_test["Person"]
    X_sv = dt_copy.drop("Person", axis=1) # rest signals/ reszta sygnałów
    y_sv = dt_copy["Person"]

    # random adding the rest of signals/ losowe dodawanie reszty sygnałów
    X_train_rest, X_test_rest, y_train_rest, y_test_rest = sklearn.model_selection.train_test_split(X_sv,
                                                                                                    y_sv,
                                                                                                    test_size=0.2,
                                                                                                    random_state=40,
                                                                                                    shuffle=True)
    X_train_sv = pd.concat([X_train_sv,X_train_rest], ignore_index=True)
    X_test_sv = pd.concat([X_test_sv,X_test_rest], ignore_index=True)
    y_train_sv = pd.concat([y_train_sv,y_train_rest], ignore_index=True)
    y_test_sv = pd.concat([y_test_sv,y_test_rest], ignore_index=True)
    
    print(X_train_sv.columns)

    # Creating Multi Layer Perceptron Classifier neural network/ Tworzenie sieci neuronowej Multi Layer Perceptron Classifier
    hidden_layers = round((len(dt) + len(Individuals)) / 2) # the optimal size of the hidden layer is usually between the size
    # of the input and size of the output layers' ~ Jeff Heaton 
    # (i) the number of hidden layers equals one.
    # (ii) the number of neurons in that layer is the mean of the neurons in the input and output layers. 
    MLPC = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(hidden_layers, ),
                                                max_iter=700,
                                                solver="adam") # lbfgs is more efficient for less amount of data /
    # lbfgs jest wydajniejsze dla małych danych
    MLPC.fit(X_train, y_train) # learning/ nauka modelu
    y_pred = MLPC.predict(X_test) # predictind persons id by neural network/ przewidywanie id osób przez sieć neuronową
    score = MLPC.score(X_test, y_test)

    MLPC_sv = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(round(1.99 * len(dt)), ),
                                                   max_iter=1500,
                                                   solver="adam") # lbfgs is more efficient for less amount of data /
    # lbfgs jest wydajniejsze dla małych danych
    MLPC_sv.fit(X_train_sv, y_train_sv)
    y_pred_sv = MLPC_sv.predict(X_test_sv) # predictind persons id by neural network/ przewidywanie id osób przez sieć neuronową
    score_sv = MLPC_sv.score(X_test_sv, y_test_sv)
    
    print(pd.DataFrame(y_pred, y_test))
    print(pd.DataFrame(y_pred_sv, y_test_sv).head(20))
    print(f"Wynik sieci neuronowej dla losowo dobranych danych uczących i testowych {str(round(score * 100, 2))} %")
    print((f"Wynik sieci neuronowej dla danych uczących i testowych dobranych pod warunki użytkowania Inteligentnej Kierownicy "
           f"{str(round(score_sv * 100, 2))} %"))
    print(f"Ilość wierszy: {len(dt)}")

if __name__ == "__main__":
    get_files(catalog_dir, saving_data_dir)
    Machine_Learning_part(saving_data_dir)

# clue: supervised splitting is giving better score than random splitting. These conditions are made for sytem, which is 
# measuring some drivers in the same car. First time the neural network is training during the first ride, after it's 
# training more or predicting, who is sitting in driver's seat.
# wniosek: kontrolowany podział danych dał możliwość uzyskania wyższego wyniku oceny modelu. Taki podział jest spowodowany
# warunkami w pojeździe podczas badania różnych kierowców za kierownicą, gdzie za pierwszym razem sieć neuronowa uczy się 
# sygnału zmierzonego na danej osobie, a następnie przy kolejnych podróżach może także się uczyć lub już przewidywać, kto 
# znajduje się w pojeździe na miejscu kierowcy
