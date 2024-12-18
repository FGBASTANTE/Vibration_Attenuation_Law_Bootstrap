"""
Aplicación del método de bootstrap a la determinación de las curvas de 
seguridad (intervalo de predicción e intervalo de tolerancia) de la ley de 
propagación de vibraciones en el terreno (PPV vs SD) definido un nivel de 
confianza (nc) y una cobertura.

Se espera que se haya establecido previamente el modelo de distancia escalada
(sd: Distancia/Carga^beta). Como ejemplo, y típicamente para cargas alargadas:
beta = 1/2, y para cargas esféricas: beta = 1/3.

En el fichero de entrada los valores x son los logaritmos decimales de
las distancias escaladas (log10(SD)) y los valores y son los log10(PPV):
x    y
1.76779    0.2001
0.69139    1.96096
1.55308    1.06786
..............
..............

También está implementado el modelo lognormal para comparar resultados.

Utilidad con fines docentes
@author: Fernando García Bastante
Universidad de Vigo
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.special as sp


def get_values(filename, number=1000):
    """
    Reads data from a CSV file and returns the first 'number' values for 
    columns 'x' and 'y'.
    
    Parameters
    ----------
    filename : str
        The path to the CSV file containing the data.
    number : int, optional
        The number of data points to return (default is 1000: all).
    
    Returns
    -------
    x : pandas.Series
        The first 'number' values of the 'x' colum: log10(SD)
    y : pandas.Series
        The first 'number' values of the 'y' column: log10(PPV)
    """
    df = pd.read_csv(filename, sep=r"\s+")
    if (len(df.index) < number) or (number == 1000):
        number = len(df.index)
    x = df['x'][:number]
    y = df['y'][:number]
    
    return x, y

def normal_regression(filename, n_points, nc=0.90, cobertura=0.95, plot_=0):
    """
    Performs normal regression on data from a file and calculates prediction
    intervals.
    
    Parameters
    ----------
    filename : str
        Path to the input data file.
    n_points : int
        The number of points to discretize the range of the x-axis for 
        predictions.
    nc : float, optional
        Confidence level for prediction intervals (default is 0.90).
    cobertura : float, optional
        Coverage for tolerance intervals (default is 0.95).
    plot_ : int, optional
        Flag to indicate whether to plot the results (default is 0, which means
                                                      do not plot).
    
    Returns
    -------
    x : pandas.Series
        The 'x' values used in the regression.
    y : pandas.Series
        The 'y' values used in the regression.
    slope : float
        The slope of the regression line.
    intercept : float
        The intercept of the regression line.
    rss_y : pandas.Series
        The residuals of the regression.
    df_intervals : pandas.DataFrame
        DataFrame containing the predicted regression values and 
        confidence/tolerance intervals.
    """

    x, y = get_values(filename)

    # Perform linear regression
    slope, intercept = np.polyfit(x, y, deg=1)
    y_predict = intercept + slope*x
    rss_y = y - y_predict
    n = len(rss_y)
    mse = np.sqrt(np.square(rss_y).sum()/(n-2))

    # Variables for confidence intervals
    x_mean = np.mean(x)
    x_gap = x-x_mean
    ss = np.square(x_gap).sum()
    
    # Generate the x_grid for predictions
    x_grid = np.linspace(np.min(x), np.max(x), n_points)
    
    # Calculate standard error for the confidence intervals
    se_x_grid = np.sqrt(1 + 1/n + np.square((x_grid - x_mean))/ss) * \
        mse*st.t.ppf(nc, df=n-2)

    # Predict the regression values for the x_grid and add confidence intervals
    y_pred_x_grid = intercept + slope*x_grid
    y_pred_x_grid_nc = y_pred_x_grid + se_x_grid

    # Calculate tolerance intervals for the x_grid
    _x_tol_grid = np.sqrt(1 / n + np.square((x_grid - x_mean)) / ss)
    zp_d_grid = st.norm.ppf(cobertura) / _x_tol_grid
    se_x_tol_grid = sp.nctdtrit(
        n - 2, zp_d_grid, nc, out=None) * mse * _x_tol_grid
    y_tol_x_grid_nc = intercept + slope*x_grid + se_x_tol_grid
    
    # Store the results in a DataFrame
    df_intervals = pd.DataFrame()
    df_intervals['x_grid'] = x_grid
    df_intervals['y_pred_regr'] = y_pred_x_grid
    df_intervals['y_pred_regr_nc'] = y_pred_x_grid_nc
    df_intervals['y_tol_regr_c_nc'] = y_tol_x_grid_nc
    
    # Optionally plot the results
    if plot_==1:
        plot_interv(x, y, df_intervals)

    return x, y, slope, intercept, rss_y, df_intervals


def bootstrap_regression(filename, n_rep=1000, draw_plot=1):
    """
    Performs bootstrap sampling to generate multiple regression lines and 
    calculates residuals for each sample.

    Parameters:
    ----------
    filename : str
        The path to the input data file containing 'x' and 'y' columns.
    n_rep : int, optional
        Number of bootstrap repetitions (default is 1000).
    draw_plot : int, optional
        Set to 1 to plot the bootstrap results, 0 otherwise (default is 0).

    Returns:
    -------
    bs_slope : np.ndarray
        The slopes of the regression lines for each bootstrap sample.
    bs_intercept : np.ndarray
        The intercepts of the regression lines for each bootstrap sample.
    bs_residuals : np.ndarray
        The residuals for each bootstrap sample.
    """
    x, y = get_values(filename)
    n = int(len(x))
    
    # Generate bootstrap samples and calculate regression parameters
    indx = np.arange(n)
    bs_slope = np.empty(n_rep)
    bs_intercept = np.empty(n_rep)
    bs_x = np.empty((n, n_rep))
    bs_y = np.empty((n, n_rep))
    bs_rss_y = np.empty((n, n_rep))

    for i in range(n_rep):
        bs_indx = np.random.choice(indx, size=len(indx), replace=True)
        bs_x[:, i], bs_y[:, i] = x[bs_indx], y[bs_indx]
        bs_slope[i], bs_intercept[i] = np.polyfit(
            bs_x[:, i], bs_y[:, i], deg=1)
        bs_rss_y[:, i] = y - bs_slope[i]*x - bs_intercept[i]
    
    # Plot results if specified
    if draw_plot == 1:
        plot_bootstrap(x, y, bs_slope, bs_intercept)

    return bs_slope, bs_intercept, bs_rss_y, bs_x


def deviat_regr_bs(x, slope, intercept, bs_slope, bs_intercept):
    """
    Calculates the deviations of the bootstrap regression lines from the 
    original regression.
    
    Parameters:
    ----------
    x : np.ndarray
        The x values (log10 of sd).
    slope : float
        The slope of the original regression line.
    intercept : float
        The intercept of the original regression line.
    bs_slope : np.ndarray
        The slopes of the regression lines for each bootstrap sample.
    bs_intercept : np.ndarray
        The intercepts of the regression lines for each bootstrap sample.
    
    Returns:
    -------
    np.ndarray
        Deviations of the bootstrap regression lines from the original 
        regression.
    """
    # mean regression parameters
    bs_slope_mean = np.mean(bs_slope)
    bs_intercept_mean = np.mean(bs_intercept)
    bs_slope_diff = bs_slope_mean - bs_slope
    bs_intercept_diff = bs_intercept_mean - bs_intercept

    # Calculate differences from the original regression parameters
    bias_intercept_diff = np.array(intercept - bs_intercept_mean)
    bias_slope_diff = np.array(slope - bs_slope_mean)                          
    e_bs_x = bs_intercept_diff[:,np.newaxis] + bs_slope_diff[:,np.newaxis] * x
    
    # Calculate bias from the original regression
    e_bs_bias = (bias_intercept_diff + bias_slope_diff * x)[np.newaxis]
    
    e_bs_x += e_bs_bias
    
    return e_bs_x.T


def convol(rss_y, bs_e, nc=0.90, cobertura=0.95):
    """
    Generates the convolution of the errors (residuals and bootstrap deviation) 
    to calculate prediction and tolerance intervals.

    Parameters:
    ----------
    rss_y : np.ndarray
        The residuals of the regression.
    bs_e : np.ndarray
        The bootstrap regression deviations.
    nc : float, optional
        Confidence level for the prediction interval (default is 0.90).
    cobertura : float, optional
        Coverage level for the tolerance interval (default is 0.95).

    Returns:
    -------
    se_x_bs_tol : np.ndarray
        The standard error for the tolerance interval.
    se_x_bs_pred : np.ndarray
        The standard error for the prediction interval.
    """
    # Convolution residuals + deviations
    desv = []
    [desv.append(np.add.outer(bs_e[i], rss_y)) for i in range(len(bs_e))]
    
    # nc deviation for prediction interval
    se_x_bs_pred = [np.percentile(desv[i], nc*100) for i in range(len(desv))]
    
    # c_nc deviation for tolerance interval
    desv_c=[]
    for item in desv:
        flat_item=[x for xs in item for x in xs]
        desv_c.append(np.percentile(np.random.choice(flat_item, 
                                                     size=(100, 100)),
                                                     q=cobertura*100,
                                                     axis=0))
        se_x_bs_tol = [np.percentile(desv_c[i], nc*100) for i in range(len(desv_c))]
        
    return se_x_bs_tol, se_x_bs_pred


def plot_bootstrap(x, y, bs_slope, bs_intercept):
    """
    Plots the results of the bootstrap regression lines.

    Parameters:
    ----------
    x : np.ndarray
        The x values (log10 of sd).
    y : np.ndarray
        The y values (log10 of ppv).
    bs_slope : np.ndarray
        The slopes of the regression lines for each bootstrap sample.
    bs_intercept : np.ndarray
        The intercepts of the regression lines for each bootstrap sample.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='.', linestyle='none')
    plt.xlabel('log10(sd)')
    plt.ylabel('log10(ppv)')
    plt.margins(0.05)
    x = np.array([np.min(x), np.max(x)])
    
    # se dibujan las rectas hasta un máximo de un millar
    for i in np.arange(len(bs_slope)):
        plt.plot(x,
                 bs_slope[i] * x + bs_intercept[i],
                 linewidth=0.5, alpha=0.25, color='blue')
    plt.show()


def plot_interv(x, y, df_interv):
    """
   Plots the data points along with the prediction and tolerance intervals.

   Parameters:
   ----------
   x : np.ndarray
       The x values (log10 of sd).
   y : np.ndarray
       The y values (log10 of ppv).
   df_interv : pd.DataFrame
       DataFrame containing the prediction and tolerance intervals.
   """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='.', linestyle='none', label='data')
    # Iterar por las columnas 'y' (todas las columnas excepto 'x')
    for column in df_interv.columns:
        if column != 'x_grid':
            plt.plot(df_interv['x_grid'], df_interv[column], label=column)

    plt.xlabel('log10(sd)')
    plt.ylabel('log10(PPV)')
    plt.title('Intervalos obtenidos con regresión (log)normal')
    plt.legend(title="Leyenda")

def generar_bootstrap(filename, n_rep=1000, n_points=20,
                      nc=0.90, cobertura=0.95, draw_plot=1):
    """
    Generates bootstrap samples, performs regression, and calculates prediction 
    and tolerance intervals for a given dataset.

    Parameters:
    ----------
    filename : str
        The path to the input data file containing 'x' and 'y' columns.
    n_rep : int, optional
        Number of bootstrap repetitions (default is 1000).
    n_points : int, optional
        Number of points for discretization (default is 20).
    nc : float, optional
        Confidence level for prediction intervals (default is 0.90).
    cobertura : float, optional
        Coverage level for tolerance intervals (default is 0.95).
    draw_plot : int, optional
        Set to 1 to plot the results, 0 otherwise (default is 1).

    Returns:
    -------
    df_interv : pd.DataFrame
        DataFrame containing the prediction and tolerance intervals.
    """

    x, y, slope, intercept, rss_y, df_interv = normal_regression(filename,
                                                                 n_points,
                                                                 nc,
                                                                 cobertura)

    bs_slope, bs_intercept, bs_rss_y, bs_x = bootstrap_regression(filename,
                                                                  n_rep)

    x_grid = np.linspace(np.min(x), np.max(x), n_points)

    e_bs_x = deviat_regr_bs(x_grid, slope, intercept,
                                 bs_slope, bs_intercept)
    se_x_bs_tol, se_x_bs_pred = convol(rss_y.values, e_bs_x, nc=nc,
                                       cobertura=cobertura)
    
    y_predict_x = intercept + slope*x_grid
    y_pred_nc_bs = y_predict_x + se_x_bs_pred
    y_tol_c_nc_bs = y_predict_x + se_x_bs_tol
   
    df_interv['y_pred_nc_bs'] = y_pred_nc_bs
    df_interv['y_tol_c_nc_bs'] = y_tol_c_nc_bs
    
    if draw_plot==1:
        plot_interv(x, y, df_interv)
        
    return df_interv, rss_y, e_bs_x


if __name__ == "__main__":

    filename = "data_simul.txt"
    n_rep = 1000
    nc = 0.90
    cobertura = 0.95
    n_points = 20
    draw_plot = 1
    
    df_results, rss_y, e_bs_x = generar_bootstrap(
        filename,
        n_rep=n_rep,
        n_points=n_points,
        nc=nc,
        cobertura=cobertura,
        draw_plot=1
    )