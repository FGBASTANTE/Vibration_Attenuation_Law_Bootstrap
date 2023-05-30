# -*- coding: utf-8 -*-
"""
Aplicación del método de bootstrap a la determinación de la recta(curva) de 
seguridad de la ley de propagación de vibraciones en el terreno definido un 
nivel de confianza (nc) dado (naturalmente, se puede extender su uso a otras
aplicaciones).

Se espera que se haya definido previamente el modelo de distancia escalada
(s_d: Distancia/Carga^beta). Como ejemplo, y típicamente para cargas alargadas:
beta = 1/2, y para cargas esféricas: beta = 1/3.

En el fichero de entrada los valores x son los logaritmos decimales de
las distancias escaladas (log10(s_d)); los valores y son, consecuentemente,
los log10(ppv):
x	y
1.76779	0.2001
0.69139	1.96096
1.55308	1.06786
..............

También está implementado el modelo lognormal para comparar resultados.

Utilidad con fines docentes
@author: Fernando García Bastante
Universidad de Vigo
"""

# se importan los módulos requeridos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# definición de parámetros globales: fichero de datos, número de replicaciones
# en el bootstrap, nivel de confianza y número de puntos en el que se 
# discretizará el rango de x (log(sd)) para obtener los resultados.
# el fichero "data_simul.txt" contiene datos simulados de un modelo lognormal
# para comparar los resultados obtenidos con ámbos métodos

filename = "data_simul.txt" # fichero de datos x y -> log(sd) log(ppv)
n_rep=10000 # número de replicaciones del bootstrap
nc = 0.95   # nivel de confianza deseado
n_points = 10 # número de puntos en el rango de log(sd)

def normal_regression(x, y, x_grid, nc=nc):
    # antes de comenzar con el boostrap se aplica el modelo log-normal a los 
    # datos de partida (sd, ppv) para comparar posteriormente los resultados de
    # aplicar ambos métodos.
    # x e y representan los log10 de sd y de ppv, respectivamente, por lo que
    # se utiliza regresión lineal
   
    # regresión lineal, la predicción y los residuos correspondientes a los 
    # datos de partida
    slope, intercept = np.polyfit(x, y, deg=1)
    y_predict = intercept + slope*x
    rss_y = y - y_predict
    n = len(rss_y)
    mse = np.sqrt(np.square(rss_y).sum()/(n-2))

    # cálculos intermedios para obtener el error total en cada punto
    x_mean = np.mean(x)
    x_gap = x-x_mean
    ss = np.square(x_gap).sum()
    
    # cálculo del error total (en los puntos x_grid)
    se_x_grid = np.sqrt(1 + 1/n + np.square((x_grid - x_mean))/ss)* \
                mse*st.t.ppf(nc, df = n-2)
    
    # cálculo de predicción en x_grid con nivel de confianza nc
    y_pred_nc = intercept + slope*x_grid + se_x_grid

    return slope, intercept, rss_y, y_pred_nc

def bootstrap_regression(x, y, n_rep=n_rep):
    """realiza el bootstrap de los datos utilizando n_rep repeticiones y la 
    regresion lineal de cada bootstrap. Los datos de entrada (x, y) están 
    en escala logarítmica:(log10(sd), log10(ppv)).
    
    Esta función es llamada por la función: generar_bootstrap()
    """
    # se crean los índices para la selección aleatoria
    indx = np.arange(len(x))

    # tuplas donde se guardarán los resultados de las regresiones
    bs_slope = np.empty(n_rep)
    bs_intercept = np.empty(n_rep)

    # generación de muestras aleatorias y su regresión lineal
    for i in range(n_rep):
        bs_indx = np.random.choice(indx, size=len(indx))
        bs_x, bs_y = x[bs_indx], y[bs_indx]
        bs_slope[i], bs_intercept[i] = np.polyfit(bs_x, bs_y, deg=1)

    return bs_slope, bs_intercept
    
def generar_bootstrap(filename, n_rep=n_rep, n_points=n_points, draw_plot=1):
    '''
    Lee los datos de entrada y llama a la función bootstrap_regression()
    
    Parameters
    ----------
    filename : fichero de texto
        DESCRIPTION.
        Fichero de texto con encabezado: x y
        x es el log10(sd)
        y es el log10(ppv)
    n_rep : int
        DESCRIPTION.
        número de repeticiones del bootstrap
    n_points : int
        DESCRIPTION.
        Número de puntos en los que se va discretizar el rango del eje x
    draw_plot: int
        DESCRIPTION.
        Poner valor igual a 1 para dibujar resultados

    Returns
    -------
    bs_slope : Array of float64
        DESCRIPTION.
        Pendiente de la regresión de cada muestra
    bs_intercept : Array of float64
        DESCRIPTION.
        Ordenada en el origen de la regresión de cada muestra
    e_bs : Dataframe
        DESCRIPTION.
        Residuos de cada regresión (n_rep) en cada punto (x_grid)    
    slope, intercept, rss_y
        DESCRIPTION.
        Los valores equivalentes obtenidos con la muestra original
    '''
    # lectura de datos
    df = pd.read_csv(filename, sep="\s+")
    x = df['x']
    y = df['y']
    
    # se genera el x_grid
    x_grid = np.linspace(np.min(x), np.max(x), n_points)
    
    # # cálculo de predicción en x_grid con nivel de confianza nc  
    slope, intercept, rss_y, y_pred_nc = normal_regression(x, y, x_grid, nc=nc)
    
    # genera el bootstrap y las regresiones correspondientes
    bs_slope, bs_intercept = bootstrap_regression(x, y, n_rep)
    
    if draw_plot == 1:
        plot_bootstrap(x, y, bs_slope, bs_intercept)
    
    # calcula los valores medios del bs y las desviaciones de cada regresión 
    # con respecto a ellos
    bs_slope_mean = np.mean(bs_slope)
    bs_intercept_mean = np.mean(bs_intercept)
    
    bs_slope_diff = bs_slope_mean - bs_slope
    bs_intercept_diff = bs_intercept_mean - bs_intercept
    
    # desviación de la media del bs con respecto a los datos originales
    bias_intercept_diff = np.array(intercept - bs_intercept_mean)
    bias_slope_diff = np.array(slope - bs_slope_mean)
    
    # calcula los residuos de cada regresión con respecto a la media obtenida 
    # de los bootstrap en los puntos x_grid
    e_bs = pd.DataFrame()
    e_bs_bias = pd.DataFrame()
    
    for i in range(len(x_grid)):
        e_bs[i] = bs_intercept_diff + bs_slope_diff * x_grid[i]
    
    # comentar para no ajustar el sesgo del bootstrap
    # sesgo de la media del boostrap con respecto a la regresión
    e_bs_bias = (bias_intercept_diff + bias_slope_diff * x_grid)[np.newaxis]
    e_bs += e_bs_bias
    
    # se calcula la convolución de los errores mediante MonteCarlo
    se_x_grid_bs = generar_simul_convol(rss_y.values, e_bs.values, nc=nc)
    
    # se obtiene el valor predicho para el nc deseado
    y_predict = intercept + slope*x_grid
    y_pred_nc_bs = y_predict + se_x_grid_bs
    
    if draw_plot == 1:
        plot_results(x, y, x_grid, y_predict, y_pred_nc, y_pred_nc_bs)
    
    return y_pred_nc, y_pred_nc_bs

def generar_simul_convol(rss_y, e_bs, nc=nc, n_conv=200000):
    '''
    genera la convolución de los errores estimados (ruido:rss_y y modelo:e_bs)
    aplicando la simulación de montecarlo.
    a continuación se obtiene el error correspondiente al nivel de confianza
    deseado
    '''
    # generación de indices aleatorios, los errores corresponndientes y su suma
    idx_ryy = np.random.choice(len(rss_y), size=n_conv)
    idx_bs= np.random.choice(len(e_bs), size=n_conv)
    sim_err = rss_y[idx_ryy]
    sim_err = np.transpose(sim_err[np.newaxis])
    sim_bs = e_bs[idx_bs]
    error =  sim_bs + sim_err
    # determinación del error corresondiente al percentil/nc deseado
    se_x_grid_bs = np.percentile(error, nc*100, axis=0)
    
    return se_x_grid_bs

def plot_bootstrap(x, y, bs_slope, bs_intercept):
    '''
    dibujo de las rectas obtenidas del bootstrap a partir de los puntos (x, y)
    '''
    # límites del eje x
    min_x, max_x = np.min(x), np.max(x)
    
    # datos de entrada
    plt.figure(0)
    plt.plot(x, y, marker='.', linestyle='none')

    # etiquetas de los ejes
    plt.xlabel('sd')
    plt.ylabel('ppv')
    plt.margins(0.05)
    
    # se añade el intervalo del eje x
    x = np.array([(min_x), (max_x)])
        # se dibujan las rectas hasta un máximo de un millar
    
    for i in np.arange(len(bs_slope)):
        plt.plot(x,
                   bs_slope[i] * x + bs_intercept[i],
                   linewidth=0.5, alpha=0.25, color='blue')

def plot_results(x, y, x_grid, y_predict, y_pred_nc, y_pred_nc_bs):
    '''
    y_predict: valores predichos (mediana)
    y_pred_nc: valores predichos con nc en el modelo lognormal
    y_pred_nc_bs: valores predichos con nc empleando bootstrap
    '''
    # gráfico de resultados
    plt.figure(10)
    _ = plt.plot(x, y, marker='.', linestyle='none', label='data')
    plt.plot(x_grid, y_predict, linestyle='solid', label='regression')
    plt.plot(x_grid, y_pred_nc, linestyle='dashed', label='nc_lognormal')
    plt.plot(x_grid, y_pred_nc_bs, linestyle='solid', label='nc_bootstrap')
    plt.legend()
    # Label axes
    _ = plt.xlabel('log(sd)')
    _ = plt.ylabel('log(ppv)')
    plt.legend()
    plt.margins(0.05)
    
# log_ppv_normal,log_ppv_bootstrap = generar_bootstrap(filename)
log_ppv_normal,log_ppv_bootstrap = generar_bootstrap(
                                                     filename,
                                                     n_points = n_points,
                                                     draw_plot=1
                                                     )
            
# ratio entre las ppv´s calculadas por ambos modelos 
print(' el ratio entre las ppv´s (normal/bootstraap) en x_grid es:')
print (10**log_ppv_normal/10**log_ppv_bootstrap)
