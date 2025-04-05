import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy.special as sspec

#### spline interpolation

def interp_1d_truth(x):
    return sspec.airy(x - 6.0)[0] + 0.6

def interp_1d_truth_der(x):
    return sspec.airy(x - 6.0)[1]

def interp_1d_truth_int(a, b):
    return sspec.itairy(b - 6.0)[0] - sspec.itairy(a - 6.0)[0] + 0.6 * (b - a)

np.random.seed(1001)
interp_1d_samp_x = np.arange(0.5, 10) + np.random.uniform(-0.2, 0.2, size=10)
interp_1d_samp_y = interp_1d_truth(interp_1d_samp_x)

#### spline fit

def fit_1d_truth(x):
    return np.sqrt(x + 1) + 0.5 * np.sin(0.6 * np.pi * x) * np.exp(-0.01*x)

fit_1d_noise_sd = 0.2

np.random.seed(101)
fit_1d_samp_x = np.arange(0, 10.1, 0.25)
fit_1d_samp_y = fit_1d_truth(fit_1d_samp_x) + np.random.normal(0, fit_1d_noise_sd, fit_1d_samp_x.shape)

np.random.seed(1001)
fit_1d_w_samp_x = np.arange(0, 10.1, 0.25)
fit_1d_weights = np.random.choice([0.05, 0.15, 0.3], size=fit_1d_w_samp_x.shape)
fit_1d_w_samp_y = 2 + 2 * np.sqrt(10 + fit_1d_w_samp_x) + np.random.normal(0, fit_1d_weights)

#### theory fit

theory_1d_noise_sd = 0.2

np.random.seed(8888)
theory_1d_params_truth = (
    np.random.uniform(1, 2),
    np.random.uniform(1, 3),
    1 / np.random.gamma(3, 2)
)

def theory_1d_truth(x):
    return theory_1d_params_truth[0] + theory_1d_params_truth[1] * np.exp(-theory_1d_params_truth[2] * x)

np.random.seed(9999)
theory_1d_samp_x = np.arange(0, 10.1, 0.25)
theory_1d_samp_y = (theory_1d_truth(theory_1d_samp_x) + 
                    np.random.normal(0, theory_1d_noise_sd, theory_1d_samp_x.shape))

#### dataset with "implicit" missing values

np.random.seed(8888)
df_implicit = pd.DataFrame({
    "year": np.repeat(np.arange(2022, 2025), 12),
    "month": np.tile(np.arange(1, 13), 3),
    "value": 1.4 - np.cos(np.arange(0, 36) * np.pi / 6) + np.random.uniform(-0.5, 0.5, size=36)
})
present = np.random.choice(np.arange(36), size=30, replace=False)
present.sort()

df_implicit = df_implicit.iloc[present, :].reset_index(drop=True)

#### 2D grid of temperature in Celcius

def temperature_2d_truth(x, y):
    return (15 + 3 * np.exp(-0.25 * ((x - 2.5)**2 + 1 * (y - 1.5)**2)) 
            - 0.6 / ((x - 1.5)**2 + (y - 0.5)**2 + 0.3)
            - 0.3 / ((x - 0.5)**2 + (y - 1.0)**2 + 0.15))

x_temp_loc = np.arange(0, 3.1, 0.2)
y_temp_loc = np.arange(0, 2.1, 0.2)
x_temp_grid, y_temp_grid = np.meshgrid(x_temp_loc, y_temp_loc)
temp_grid = temperature_2d_truth(x_temp_grid, y_temp_grid)

#### path to transverse

t_path = np.arange(0, 60, 1)
x_path = 1.5 + 1.2 * np.cos(np.pi * t_path / 30)
y_path = 1.0 + 0.8 * np.sin(np.pi * t_path / 30)

#### clean 2D function

def clean_2d(x, y):
    return (6 * np.exp(-0.3 * (x**2 + 3 * y**2)) 
            - 5 / (2 * (x - 2.5)**2 + (y - 1.5)**2 + 1)
            - 5 / (2 * (x + 2.5)**2 + (y + 1.5)**2 + 1))

X_clean = np.linspace(-3, 3, 61)
Y_clean = np.linspace(-2, 2, 41)
X_grid, Y_grid = np.meshgrid(X_clean, Y_clean)

Z_clean = clean_2d(X_grid, Y_grid)

X_noisy = X_clean.copy()
Y_noisy = Y_clean.copy()

np.random.seed(7777)
Z_noisy = Z_clean + np.random.uniform(-0.5, 0.5, Z_clean.size).reshape(Z_clean.shape)

#### 2D grid of NO3 concentration

def NO3_2d_truth(x, y):
    return (6 * np.exp(-0.5 * ((0.1 * x - 0.5)**2 + 2 * (0.1 *y - 1.5)**2)) 
            + 0.5 / ((0.1 *x - 1.5)**2 + (0.1 *y - 0.5)**2 + 0.1)
            + 0.8 / ((0.1 *x - 2.5)**2 + (0.1 *y - 1.0)**2 + 0.2))

x_stations, y_stations = np.meshgrid(np.arange(2.5, 31, 5), np.arange(2.5, 21, 5))
x_stations = x_stations.flatten()
y_stations = y_stations.flatten()

np.random.seed(1001)
x_stations = x_stations + np.random.uniform(-2, 2, 24)
y_stations = y_stations + np.random.uniform(-2, 2, 24)
missing = np.random.choice(range(24), size=6, replace=False)

x_stations = x_stations[[x not in missing for x in range(24)]]
y_stations = y_stations[[x not in missing for x in range(24)]]

NO3_stations = NO3_2d_truth(x_stations, y_stations)


#### probe path in (lon, lat, depth) space

probe_hour = np.arange(24)
probe_lat = np.linspace(43, 35, 24)
probe_lon = np.linspace(-135, -140, 24)
probe_depth = np.concatenate([np.arange(10, 101, 10), np.full(4, 100), np.arange(100, 5, -10)])

#### probe path in (t, x, y) space

t0_array = np.linspace(0, 10, 50, endpoint=False)
missing = [1, 3, 5, 7, 9]

probe_x = np.linspace(3, 10, t0_array.size) * np.cos(3 * t0_array)
probe_y = np.linspace(3, 10, t0_array.size) * np.sin(3 * t0_array)
probe_x = probe_x[[x not in missing for x in range(t0_array.size)]]
probe_y = probe_y[[x not in missing for x in range(t0_array.size)]]
probe_t = 10 * np.arange(probe_x.size)

### synthetic salinity data

def sal_2d_truth(x, y):
    return (3.5 + 0.5 * np.exp(-0.2 * ((0.1 * x - 1.1)**2 + 2 * (0.1 *y - 1.1)**2)) 
            - 0.5 * np.exp(-0.3 * (2 * (0.1 * x + 1.1)**2 + (0.1 *y + 1.1)**2)) )

np.random.seed(8888)
probe_sal = sal_2d_truth(probe_x, probe_y)
probe_sal = probe_sal + np.random.normal(0, 0.1, probe_sal.size)

#### creating clipping path for plot

def probe_clip(ax):
    path = mpl.path.Path(
        np.vstack([
            np.concatenate([probe_x[-11:], probe_x[-11:-10]]), 
            np.concatenate([probe_y[-11:], probe_y[-11:-10]])
        ]).transpose(), 
        codes = [mpl.path.Path.MOVETO] + [mpl.path.Path.LINETO] * 10 + [mpl.path.Path.CLOSEPOLY]
    )
    return (path, ax.transData)
