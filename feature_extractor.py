import antropy as ant
import neurokit2 as nk
import numpy as np
import pandas as pd
import pathlib, os
from numpy import absolute, mean, median, std, var
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import gmean, hmean, iqr, kurtosis, mode, moment, skew, trim_mean

# Función para calcular la energía de Teager media
def EO_5(data):
  ave = (sum([i**2 for i in data[1:-1]])-sum([i*j for i,j in zip(data[:-2],data[2:])]))/len(data)
  num = (sum([(j**2-i*k-ave)**4 for i,j,k in zip(data[:-2],data[1:-1],data[2:])]))*(len(data)-1)
  den = (sum([(j**2-i*k-ave)**2 for i,j,k in zip(data[:-2],data[1:-1],data[2:])]))**2
  return num/den

DATA_DIR_PATH = "./Datos/"

HERTZ_FREQ = 60                             # Hertz frequency (how many rows there are in one second)
EPOCH_SIZE = 30                             # 30 seconds per epoch
WINDOW_SIZE = HERTZ_FREQ * EPOCH_SIZE       # 60*30 = 1800 samples per single window

sleep_stage_dict = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4
}

filenames = os.listdir(DATA_DIR_PATH)

# Se almacenan todas las características importantes por latido
results_arr = []

for patient_id, filename in enumerate(filenames):
    
    df = pd.read_csv(DATA_DIR_PATH + filename)

    print(len(df))

    df = df[df["Sleep_Stage"] != "P"]

    print(len(df))

    total_windows = len(df)/WINDOW_SIZE

    for i in range(0, len(df), WINDOW_SIZE):
        
        cur_window_df = df.iloc[i : i + WINDOW_SIZE]
        
        cur_window_results = []
        
        if len(cur_window_df) < WINDOW_SIZE:            # Si no hay una ventana completa, esta será ignorada
            break
        
        bvp = cur_window_df["BVP"].values
        
        filtered_bvp = savgol_filter(bvp, 9, 3)
        
        # Cálculo de características basadas en la morfología de la onda
        # --------------------------------------------------------------
        
        # -> A: la amplitud mide la diferencia entre el valor más alto en una sub-serie y el más bajo. 
        # Utilizando la función peak_detectors de scipy, se divide por latidos. Suponiendo que, en reposo,
        # en promedio se tienen unos entre 60-90 lpm, se divide en subsets de 30 muestras 
        # (1 latido -> 0.5 segundos -> 30 muestras)

        # -> Picos sistólico/diastólico: en el estudio de señales cardíacas, existe un punto llamado
        # "nodo dicrótico" que indica cuándo termina la fase de sístole y empieza la de diástole.
        # Este se basa en el concepto de la segunda derivada, que indica los puntos de concavidad
        # y convexidad de una gráfica, además de la aceleración (esta se utiliza porque en la sístole
        # la sangre sale disparada del corazón, y en la diástole se relaja, implicando valores evidentes
        # de la aceleración, ideal para detectar lo que se busca). Esto también influye en el cálculo
        # del área total del latido y del tiempo total del latido, ya que ambas son la suma de sus
        # características homónimas en sístole y diástole.
        
        # -> Inflection point time/area: se refiere a la relación entre las fases de sístole y
        # diástole, tanto en tiempo como en la suma de valores media en sí (el área bajo las curvas -> la integral)
        
        # -> Total beat time: tiempo total del latido (suma de tiempo en sístole + diástole)
        
        # Cálculo de características basadas en el ritmo y la respiración
        # ---------------------------------------------------------------
        
        # -> PPI: diferencia entre picos consecutivos, ideal para evaluar la variabilidad de los latidos
        
        # -> Características basadas en la respiración: la "respiration-induced amplitud modulation" (RIAM)
        # se calcula, en base a algunos papers, como la diferencia entre las amplitudes de los picos y su media 
        # (y, equivalentemente, teniendo en cuenta las diferencias entre instantes es como se calcula la 
        # "respiration-induced frequency modulation", o RIFM).
        
        second_deriv = np.gradient(np.gradient(filtered_bvp))
        
        positive_peaks, _ = find_peaks(filtered_bvp, distance=30, prominence=std(filtered_bvp)*0.2)      # Índice de los valores más altos (máximos locales) cada
                                                                                                            # 30 muestras (0.5s), prominence para evitar desviaciones
                                                                                                            # pequeñas
                                                                                                            
        negative_peaks, _ = find_peaks(-filtered_bvp, distance=30, prominence=std(filtered_bvp)*0.2)     # Índice de los valores más bajos (gráfica invetida ->
                                                                                                            # mínimos locales) cada 30 muestras
        
        # Array auxiliar, para aquellas medidas que no requerirán un cálculo posterior
        # de media o desviación estándar
        ppi_results = []
        
        # Cálculo del intervalo entre pulsos
        ppi = [(positive_peaks[j] - positive_peaks[j-1]) / HERTZ_FREQ for j in range(1, len(positive_peaks))]
        
        ppi_std = std(ppi)
        ppi_mean = mean(ppi)
        
        ppi_successive = [ppi[j] - ppi[j-1] for j in range(1, len(ppi))]
        ppi_rmssd = (np.sum([ppi_successive[j]**2 for j in range(0, len(ppi_successive))]) / len(ppi_successive)) ** 0.5
        
        # Cálculo de las diferencias en amplitudes e instantes de tiempo en los picos (RIAM y RIFM)
        amplitudes = [filtered_bvp[j] for j in positive_peaks]
        riam = std(amplitudes)/mean(amplitudes)

        rifm = std(ppi)/mean(ppi)
        
        ppi_results.append({
            "ppi_std": ppi_std,
            "ppi_mean": ppi_mean,
            "ppi_rmssd": ppi_rmssd,
            "riam": riam,
            "rifm": rifm
        })
        
        for peak in positive_peaks:
            
            peak_value = filtered_bvp[peak]
            
            # Inicio del último pico negativo anterior (mínimo local)
            last_negative_peak = negative_peaks[negative_peaks<peak]
            
            if len(last_negative_peak)==0:
                continue
            
            last_negative_peak_value = filtered_bvp[last_negative_peak[-1]]
            
            # Cálculo el valor de la amplitud
            amplitude = peak_value-last_negative_peak_value
            
            # Inicio del primer pico negativo posterior (si lo hay)
            next_negative_peaks = negative_peaks[negative_peaks>peak]
            
            if len(next_negative_peaks)==0:
                continue
            
            first_next_negative_peak = next_negative_peaks[0]
            
            # Área donde se encuentra el nodo diacrótico (entre el pico y el siguiente mínimo) -> segunda derivada
            # Si la segunda derivada devuelve "10", realmente indica que está 10 posiciones después del índice del
            # pico actual; de ahí que se sumen
            notch_index = peak + np.argmax(second_deriv[peak:first_next_negative_peak])
            
            # Cálculo del área sistólica, restando el mínimo inicial para normalizar los valores (si no,
            # se estaría sumando partiendo de ese valor, en lugar del 0, por lo que la diferencia no sería válida)
            systolic_area = np.sum(filtered_bvp[last_negative_peak[-1]:notch_index] - last_negative_peak_value)
            
            # Cálculo del área diastólica (relajación del corazón, de nuevo centrado en y=0)
            diastolic_area = np.sum(filtered_bvp[notch_index:first_next_negative_peak] - last_negative_peak_value)
            
            # Cálculo del tiempo total en fase de sístole, pasado a segundos.
            systolic_time = (notch_index-last_negative_peak[-1])/HERTZ_FREQ
        
            # Cálculo del tiempo total en fase de deiástole, pasado a segundos.
            diastolic_time = (first_next_negative_peak-notch_index)/HERTZ_FREQ
            
            # Cálculo de las métricas de inflexión + tiempo total del latido
            inflection_point_time = diastolic_time/systolic_time if systolic_time > 0 else 0
            inflection_point_area = diastolic_area/systolic_area if systolic_area > 0 else 0
            
            total_heartbeat_time = systolic_time + diastolic_time
            total_heartbeat_area = diastolic_area + systolic_area
            
            cur_window_results.append({
                "systolic_area": systolic_area,
                "diastolic_area": diastolic_area,
                "systolic_time": systolic_time,
                "diastolic_time": diastolic_time,
                "amplitude": amplitude,
                "inflection_point_area": inflection_point_area,
                "inflection_point_time": inflection_point_time,
                "total_heartbeat_time": total_heartbeat_time,
                "total_heartbeat_area": total_heartbeat_area
            })
        
        # Cálculo de las métricas basadas en la forma de la subfunción (las 1800 muestras/30s)
        # ------------------------------------------------------------------------------------
        
        shape_function_results = []
        
        # -> Media de la desviación en valor absoluto: indica cuánto se desvía de la media la función en términos absolutos
        avgADPPG = mean(absolute(filtered_bvp - mean(filtered_bvp, None)), None)
        
        # -> Desviación estándar de la desviación en valor absoluto: similar a la métrica anterior, pero utilizando la desviación estándar
        stdADPPG = std(absolute(filtered_bvp - mean(filtered_bvp, None)), None)
        
        # -> Desviación estándar de la mediana en valor absoluto: similar a las métricas anteriores, pero utilizando la mediana
        MADPPG = median(absolute(filtered_bvp - mean(filtered_bvp, None)), None)
        
        # -> Rango intercuartiles: devuelve la diferencia entre los percentiles 75 y 25 de los datos (otra forma de calcular desviaciones en los datos)
        IQRPPG = iqr(filtered_bvp)
        
        # -> Momento enésimo central: calcula la media de la diferencia de los valores respecto a un determinado centro. Es una medida sobre
        # la forma de los datos, como puedan ser la curtosis o el sesgo
        nCMPPG = moment(filtered_bvp, order=2, axis=None)
        
        # -> Energía media: toma la media de las magnitudes de la señal al cuadrado, con el objetivo de evaluar la intensidad
        avgEPPG = mean(filtered_bvp**2)
        
        # -> Promedio de la forma: indica cómo de diferente es la media de los valores más apuntados
        SFPPG = np.sqrt(mean(np.square(filtered_bvp))) / mean(np.abs(filtered_bvp))
        
        # -> Longitud media de la curva: mide la diferencia entre muestras consecutivas para evaluar variaciones bruscas
        avgCLPPG = mean([abs(filtered_bvp[j] - filtered_bvp[j-1]) for j in range(1, len(filtered_bvp))])
        
        # -> Energía de Teager media: medida de curtosis sobre la distribución original transformada para medir la
        # energía instantánea según el operador enegético de Teager:
        avgTEPPG = mean(filtered_bvp[1:-1]**2 - (filtered_bvp[:-2] * filtered_bvp[2:]))
        
        # -> Media geométrica: resultado de aplicar al producto de los valores de la distribución la raíz enésima,
        # siendo n el número de operandos que intervienen en ese producto. Esto solo funciona con valores positivos;
        # de ahí la resta del valor mínimo.
        GmPPG = gmean(filtered_bvp - np.min(filtered_bvp) + 0.01, axis=None)
        
        # -> Media armónica: es el recíproco de la media de los recíprocos (es decir, la inversa de la media de los
        # valores invertidos, 1/n, para los n valores de una distribución). Esto solo funciona con valores positivos;
        # de ahí la resta del valor mínimo.
        HmPPG = hmean(filtered_bvp - np.min(filtered_bvp) + 0.01, axis=None)
        
        # -> Media cortada al 25 y al 50%: elimina un porcentaje especificado de valores, tanto por arriba como por abajo
        TM25PPG = trim_mean(filtered_bvp, .125)
        TM50PPG = trim_mean(filtered_bvp, .25)
        
        # -> Sesgo y curtosis: medidas de la forma de una distribución (skew mide la asimetríe de la función respecto a la media,
        # y kurtosis mide cómo de "pesadas" son las colas de la distribución)
        SkewPPG = skew(filtered_bvp)
        KurtPPG = kurtosis(filtered_bvp)
        
        shape_function_results.append({
            "avgADPPG": avgADPPG,
            "stdADPPG": stdADPPG,
            "MADPPG": MADPPG,
            "IQRPPG": IQRPPG,
            "nCMPPG": nCMPPG,
            "avgEPPG": avgEPPG,
            "SFPPG": SFPPG,
            "avgCLPPG": avgCLPPG,
            "avgTEPPG": avgTEPPG,
            "GmPPG": GmPPG,
            "HmPPG": HmPPG,
            "TM25PPG": TM25PPG,
            "TM50PPG": TM50PPG,
            "SkewPPG": SkewPPG,
            "KurtPPG": KurtPPG
        })
        
        
        # Cálculo de las métricas basadas en la distribución de los puntos
        # ----------------------------------------------------------------
        
        point_distribution_results = []
        
        # -> Medidas de Poincaré (SD1 y SD2): se ponen los puntos de la distribución en una gráfica, de modo que
        # en el eje X se pone el valor actual, y en el eje Y el valor siguiente. La distribución va formando una
        # especie de óvalo, y SD1 y SD2 se encargan de medir cómo de ancha y larga es.
        
        x = filtered_bvp[:-1]
        y = filtered_bvp[1:]

        SD1PPG = np.std(x - y) / np.sqrt(2)
        SD2PPG = np.std(x + y) / np.sqrt(2)
        RSD1SD2PPG = SD1PPG/SD2PPG

        # -> Medida de correlación compleja (CCM): siguiendo con el gráfico de Poincaré, se van uniendo los puntos por orden,
        # de manera que, si no hay muchos saltos, la CCM será baja. Mide la predictibilidad de una observación dada la anterior.
        diffs = [abs(filtered_bvp[j]-filtered_bvp[j-1]) for j in range(1, len(filtered_bvp))]
        CCMPPG = mean(diffs)/(SD1PPG * SD2PPG)
        
        point_distribution_results.append({
            "SD1PPG": SD1PPG,
            "SD2PPG": SD2PPG,
            "RSD1SD2PPG": RSD1SD2PPG,
            "CCMPPG": CCMPPG
        })
        
        
        # Cálculo de las métricas no lineales
        # -----------------------------------
        
        # -> Parámetros de Hjord: actividad, movilidad y complejidad. Inicialmente pensados para encefalogramas, también se aplica
        # en otro tipo de pruebas, como los electrocardiogramas. Básicamente, la actividad es la varianza de las amplitudes, la 
        # movilidad es la proporción de la variación temporal (derivada) de la actividad respecto de la media de la actividad, y la
        # complejidad el cociente de la variación temporal de la movilidad respecto a la media de la movilidad, sirviendo esta última
        # para identificar la estabilidad de una señal por comparación directa con una función sinusoide
        
        non_lineal_results = []
        
        HAPPG = var(filtered_bvp)
        
        time_diff = [filtered_bvp[j]-filtered_bvp[j-1] for j in range(1, len(filtered_bvp))]
        HMPPG = np.sqrt(var(time_diff)/HAPPG)
        
        time_diff_2 = [time_diff[j]-time_diff[j-1] for j in range(1, len(time_diff))]
        mob_diff = np.sqrt(np.var(time_diff_2)/np.var(time_diff))
        HCPPG = mob_diff/HMPPG
        
        # -> Fractales de Higuchi y Katz:
        HFDPPG = ant.higuchi_fd(filtered_bvp)
        KFDPPG = ant.katz_fd(filtered_bvp)
        LCPPG = ant.detrended_fluctuation(filtered_bvp)
        
        # Se añaden las métricas no lineales
        non_lineal_results.append({
            "HAPPG": HAPPG,
            "HMPPG": HMPPG,
            "HCPPG": HCPPG,
            "HFDPPG": HFDPPG,
            "KFDPPG": KFDPPG,
            "LCPPG": LCPPG
        })
        
        if cur_window_results:
        
            # Cálculo de la media y desviaciones estándar de la ventana actual (las features
            # que realmente se utilizan para la predicción)
            cur_window_results_df = pd.DataFrame(cur_window_results)
            
            results_to_append = {"cur_window": int(i/WINDOW_SIZE)}
            
            for feature in cur_window_results_df.columns:
                results_to_append[f"mean_{feature}"] = cur_window_results_df[feature].mean()
                results_to_append[f"sd_{feature}"] = cur_window_results_df[feature].std()
            
            # Se añaden las métricas basadas en los intervalos entre pulsos (ppi)
            for name, value in ppi_results[0].items():
                results_to_append[name] = value
                
            # Se añaden todas las métricas relacionadas con la forma de la distribución
            for name, value in shape_function_results[0].items():
                results_to_append[name] = value
                
            # Se añaden las métricas basadas en la distribución de los puntos
            for name, value in point_distribution_results[0].items():
                results_to_append[name] = value
            
            for name, value in non_lineal_results[0].items():
                results_to_append[name] = value
            
            results_to_append["sleep_stage"] = mode(np.array(cur_window_df["Sleep_Stage"].apply(lambda x: sleep_stage_dict[x])))[0]
            
            results_to_append["patient_id"] = patient_id
            
            results_arr.append(results_to_append)

features_arr = pd.DataFrame(results_arr)
features_arr.to_csv("extracted_features.csv")

print(features_arr)
