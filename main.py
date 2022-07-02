# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math, random
import config
import numpy as np
import pandas as pd


def generate_output_prognostics_model(num_units, max_life_unit, min_life_unit):
    all_times, all_ruls, all_units ,all_predictions = [], [], [], []
    num_units = int(num_units)
    max_life_unit = int(max_life_unit)
    min_life_unit = int(min_life_unit)
    for unit in range(num_units):
        life_unit = random.randrange(min_life_unit, max_life_unit)
        unit_times = np.arange(life_unit)
        unit_ruls = life_unit - unit_times
        unit_predictions = [np.random.normal(loc=rul, scale=rul * random.random()*0.3, size=None) for rul in unit_ruls]
        unit_list = [unit] * life_unit
        all_times.extend(unit_times)
        all_ruls.extend(unit_ruls)
        all_units.extend(unit_list)
        all_predictions.extend(unit_predictions)
    df = pd.DataFrame({'unit': all_units, 'RUL': all_ruls, 'time': all_times, 'prediction': all_predictions})
    return df


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


"""
Calculates the average scale independent error

Weighs exponentially the errors in RUL predictions and averages over several units under test (UUT)
Here, D_zero is a normalizing constant whose value depends on the magnitudes in the application
"""


def average_scale_independent_error(errors, D_zero):
    L = len(errors)
    asie = np.exp(-np.abs(errors) / D_zero)
    asie = np.sum(asie) / L
    return asie


"""
Calculates the average bias or mean error

Averages the errors in predictions made at all subsequent times after prediction
starts for the unit. This metric can be extended to average biases over all
UUTs to establish overall bias
"""


def average_bias(errors):
    avgbias = np.mean(errors)
    return avgbias


"""
Calculates the mean absolute error

Averages the *absolute* errors in predictions made at all times for the unit. 
This metric can be extended to average absolute errors over all
UUTs to establish the MAE
"""


def mean_absolute_error(errors):
    mae = np.mean(np.absolute(errors))
    return mae


"""
Calculates the root mean squared error

Averages the squared prediction error for multiple UUTs at the same prediction
horizon.
A derivative of MSE is Root Mean Squared Error (RMSE)
"""


def mean_squared_error(errors):
    mse = np.mean(np.power(errors, 2))
    rmse = math.sqrt(mse)
    return mse, rmse



"""
Calculates the mean absolute percentage error

Averages the *absolute percentage* errors in predictions made at all times for the unit. 
This metric can be extended to average absolute % errors over all
UUTs to establish the MAPE
"""


def mean_absolute_percentage_error(errors, actual_ruls):
    mape = np.mean(np.divide(np.absolute(errors), actual_ruls))
    return mape*100


"""
Calculates the average score (timeliness)

Exponentially weighs RUL prediction errors through an asymmetric weighting
function. Penalizes the late predictions more than early predictions.
    a_1 < a_2 < 0
"""


def average_score(errors, a_1, a_2):
    phi = []
    for error in errors:
        if error > 0: # early prediction
            score = math.exp(math.fabs(error) / a_1) - 1
            phi.append(score)
        else: # late prediction
            score = math.exp(math.fabs(error) / a_2) - 1
            phi.append(score)
    all_score = np.mean(phi)
    return all_score


"""
Calculates the number and percentage of false positives (FP)

FP assesses unacceptable early predictions at specific time instances
Users must set acceptable ranges T_FP. 
Early predictions result in excessive lead time, which may
lead to unnecessary correction
"""


def false_positives(errors, t_fp):
    num_false_pos = 0
    early_prd = 0
    for error in errors: # error > 0
        if error > 0: # early prediction
            if error > t_fp: #unacceptable early prediction
                num_false_pos += 1
            early_prd += 1
    return num_false_pos, num_false_pos/early_prd


"""
Calculates the number and percentage of false negatives (FN)

FP assesses unacceptable late predictions at specific time instances
Users must set acceptable ranges T_FN. 
s. Also note that, a prediction that is late more than
a critical threshold time units is equivalent to not making any prediction and
having the failure occurring.
T_FN = = user defined acceptable late prediction
"""


def false_negatives(errors, t_fn):
    num_false_neg = 0
    late_prd = 0
    for error in errors: # error > 0
        if error <= 0: # late prediction
            if -error > t_fn:
                num_false_neg += 1
            late_prd += 1
    return num_false_neg, num_false_neg/late_prd


"""
Calculates the anomaly correlation coefficient (ACC)

Measures correspondence or phase difference between prediction and
subtracting out the historical mean at each point. The anomaly
correlation is frequently used to verify output from numerical weather prediction
(NWP) models. ACC is not sensitive to error or bias, so a good anomaly
correlation does not guarantee accurate predictions. In the PHM context, ACC
computed over a few time-steps after tp can be used to modify long term
predictions. However, the method requires computing a baseline from history
data which may be difficult to come by. 

The anomaly correlation coefficient is equivalent to the Pearson correlation coefficient, 
except that both the forecasts and observations are first adjusted according to a climatology value. 
The anomaly is the difference between the individual forecast or observation and the typical situation,
as measured by a climatology (c) of some variety. 
It measures the strength of linear association between the forecast anomalies and observed anomalies. 
https://met.readthedocs.io/en/latest/Users_Guide/appendixC.html

Anomaly correlation can range between -1 and 1; 
A value of 1 indicates perfect correlation and a value of -1 indicates perfect negative correlation. 
A value of 0 indicates that the forecast and observed anomalies are not correlated.
"""


def anomaly_correlation_coefficient(times, predictions, observations):
    predictions_TTF = times + predictions # predicted time to failure (TTF)
    mean_predictions_TTF = []

    for i in range(len(predictions_TTF)):
        mean_predictions_TTF.append(np.mean(predictions_TTF[:i+1])) # calculate the mean of the past predictions (TTF)

    mean_predictions_TTF = mean_predictions_TTF - times # back to rul

    numerator = 0
    for f_i, o_i, i in zip(predictions, observations, range(len(observations))):
        c_reference = mean_predictions_TTF[i]
        numerator += (f_i - c_reference) * (o_i - c_reference)

    denominator = math.sqrt(np.sum(np.power(predictions - mean_predictions_TTF, 2)) * np.sum(np.power(observations - mean_predictions_TTF, 2)))
    print(predictions - mean_predictions_TTF)
    return numerator / denominator


"""
Calculates the symmetric mean absolute percentage error (sMAPE)

Averages the absolute percentage errors in the predictions of multiple UUTs at
the same prediction horizon. The percentage is computed based on the mean
value of the prediction and ground truth. This prevents the percentage error from
being too large for the cases where the ground truth is close to zero.
"""


def symmetric_mean_absolute_percentage_error(errors, actual_ruls, predictions):
    smape = np.mean(np.divide(np.absolute(errors), (actual_ruls + predictions) / 2))
    return smape*100


"""
Calculates the sample standard deviation (S)

Sample standard deviation measures the dispersion/spread of the error with
respect to the sample mean of the error. This metric is restricted to the
assumption of normal distribution of the error. It is, therefore, recommended to
carry out a visual inspection of the error plot.

"""


def sample_standard_deviation(errors):
    sad = math.sqrt(np.sum( np.power(np.absolute(errors) - np.mean(np.absolute(errors)), 2) )) / (len(errors) - 1)
    return sad


"""
Calculates the mean/median absolute deviation from the sample median (S)

This is a resistant estimator of the dispersion/spread of the prediction error. It is
intended to be used where there is a small number of UUTs and when the error
plots do not resemble those of a normal distribution.
"""


def mean_median_absolute_deviation_sample_median(errors):
    M = np.median(errors)
    MAD = np.mean(np.absolute(errors - M))
    MdAD = np.median(np.absolute(errors - M))
    return MAD, MdAD


"""
Calculates the prognostics horizon

Prognostic Horizon is the difference between the current time index i and EOP utilizing data
accumulated up to the time index i, provided the prediction meets desired specifications
"""


def prognostics_horizon(errors, actual_ruls, units, bound_late, bound_early):
    horizon_units, horizon_percentage_units = [], []
    for unit in np.unique(units):
        ruls_unit = actual_ruls[units == unit]
        ttf = ruls_unit[0]
        errors_unit = errors[units == unit]
        found_horizon = False
        for actual_rul, error in zip(ruls_unit, errors_unit):
            if error < 0 and abs(error) <= bound_late: # late prediction
                horizon_units.append(actual_rul)
                horizon_percentage_units.append(actual_rul/ttf)
                found_horizon = True
                break
            elif error >= 0 and abs(error) <= bound_early: # early prediction
                horizon_units.append(actual_rul)
                horizon_percentage_units.append(actual_rul / ttf)
                found_horizon = True
                break
        if not found_horizon:
            horizon_units.append(0)
            horizon_percentage_units.append(0)

    print(len(np.unique(units)), len(horizon_units))
    return horizon_units, horizon_percentage_units


"""
Calculates the prediction spread

This quantifies the variance of prediction over time for any UUT t. It can be computed over
any accuracy or precision based metric M
"""


def prediction_spread(predicted_ruls, times, units):
    prediction_spread_per_unit = []
    for unit in np.unique(units):
        times_unit = times[units == unit]
        predicted_ruls_unit = predicted_ruls[units == unit]
        predicted_ttfs = predicted_ruls_unit + times_unit
        prediction_spread_per_unit.append(np.std(predicted_ttfs))
    return prediction_spread_per_unit


"""
Calculates the alpha-lambda accuracy performance

Prediction accuracy at specific time instances; e.g., demand accuracy of prediction to be
within a* 1000/0 after fault detection some defined relative distance
A to actual failure. For
example, 200/0 accuracy (i.e., a=0.2) halfway to failure after fault detection (i.e.,
A=0.5).
"""

def alpha_lambda_accuracy(actual_ruls, predicted_ruls, units, errors, alpha_early_predictions, alpha_late_predictions):
    alpha_lambda_accuracy = []
    for unit in np.unique(units):
        ruls_unit = actual_ruls[units == unit]
        predicted_ruls_unit = predicted_ruls[units == unit]
        ttf_unit = ruls_unit[0]
        errors_unit = errors[units == unit]
        positives_unit = [0] * 10
        num_predictions_unit = [0] * 10
        for actual_rul, predicted_rul, error in zip(ruls_unit, predicted_ruls_unit, errors_unit):
            lambda_time = math.floor(actual_rul/ttf_unit * 10 -0.01) # 10 (9) is beginning of life, 0 is close to end of life
            if error < 0:
                if abs(error) <= actual_rul * alpha_late_predictions: # late prediction
                    positives_unit[lambda_time] += 1
            elif error >= 0:
                if abs(error) <= actual_rul * alpha_early_predictions: # early prediction
                    positives_unit[lambda_time] += 1
            num_predictions_unit[lambda_time] += 1
        alpha_lambda_accuracy_unit = np.divide(positives_unit, num_predictions_unit)
        alpha_lambda_accuracy.append(alpha_lambda_accuracy_unit)

    alpha_lambda_accuracy_avg_over_all_units = []
    for i in range(10):
        alpha_lambda_accuracy_at_lambda_i = []
        for alpha_lambda_accuracy_unit in alpha_lambda_accuracy:
            alpha_lambda_accuracy_at_lambda_i.append(alpha_lambda_accuracy_unit[i])
        alpha_lambda_accuracy_avg_over_all_units.append(np.mean(alpha_lambda_accuracy_at_lambda_i))

    return alpha_lambda_accuracy, alpha_lambda_accuracy_avg_over_all_units


"""
Calculates the alpha-lambda precision performance

Prediction accuracy at specific time instances; e.g., demand accuracy of prediction to be
within a* 1000/0 after fault detection some defined relative distance
A to actual failure. For
example, 200/0 accuracy (i.e., a=0.2) halfway to failure after fault detection (i.e.,
A=0.5).
"""


def alpha_lambda_precision(actual_ruls, predicted_ruls, times, units, errors):
    alpha_lambda_precision_all_units = []

    for unit in np.unique(units):
        ruls_unit = actual_ruls[units == unit]
        times_unit = times[units == unit]
        ttf_unit = ruls_unit[0]
        predicted_ruls_unit = predicted_ruls[units == unit]
        predicted_ttfs_unit = predicted_ruls_unit + times_unit
        errors_unit = errors[units == unit]
        ttf_predictions_unit = [[], [], [], [], [], [], [], [], [], []]
        alpha_lambda_precision_unit = [0] * 10
        for actual_rul, predicted_rul, predicted_ttf, error in zip(ruls_unit, predicted_ruls_unit, predicted_ttfs_unit, errors_unit):
            lambda_time = math.floor(actual_rul/ttf_unit * 10 -0.01) # 10 (9) is beginning of life, 0 is close to end of life
            ttf_predictions_unit[lambda_time].append(predicted_ttf)
        for i in range(10):
            alpha_lambda_precision_unit[i] = np.std(ttf_predictions_unit[i])
        alpha_lambda_precision_all_units.append(alpha_lambda_precision_unit)

    alpha_lambda_precision_avg_over_all_units = []
    for i in range(10):
        alpha_lambda_precision_at_lambda_i = []
        for alpha_lambda_precision_at_lambda_of_unit in alpha_lambda_precision_all_units: # for all units
            alpha_lambda_precision_at_lambda_i.append(alpha_lambda_precision_at_lambda_of_unit[i])
        alpha_lambda_precision_avg_over_all_units.append(np.mean(alpha_lambda_precision_at_lambda_i))

    return alpha_lambda_precision_all_units, alpha_lambda_precision_avg_over_all_units



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = generate_output_prognostics_model(config.num_units, config.max_life_unit, config.min_life_unit)
    errors = df.RUL.values - df.prediction.values # actual RUL - prediction
    actual_ruls = df.RUL.values
    times = df.time.values
    units = df.unit.values
    predicted_ruls = df.prediction.values
    asie = average_scale_independent_error(errors=errors, D_zero=config.max_life_unit)
    print('average scale independent error=', asie)
    avg_bias = average_bias(errors)
    print('Average bias=', avg_bias)
    mae = mean_absolute_error(errors)
    print('Mean absolute error=', mae)
    penalty_a1 = config.penalty_a1
    penalty_a2 = config.penalty_a2
    avg_score = average_score(errors, penalty_a1, penalty_a2)
    print('Average score=', avg_score)
    t_fp = config.t_fp
    t_fn = config.t_fn
    num_false_pos, perc_false_pos = false_positives(errors, t_fp=t_fp)
    print('False positives (unacceptable early predictions) =', num_false_pos, perc_false_pos)
    num_false_neg, perc_false_neg = false_negatives(errors, t_fn=t_fn)
    print('False negatives (unacceptable late predictions) =', num_false_neg, perc_false_neg)
    mape = mean_absolute_percentage_error(errors, actual_ruls)
    print('Mean absolute percentage error=', mape)

    acc = anomaly_correlation_coefficient(times, predicted_ruls, actual_ruls)
    print('Anomaly correlation coefficient=', acc)

    smape = symmetric_mean_absolute_percentage_error(errors, actual_ruls, predicted_ruls)
    print('Symmetric mean absolute percentage error=', smape)
    mse, rmse = mean_squared_error(errors)
    print('Mean squared error=', mse, "Root mean squared error=", rmse)

    sad = sample_standard_deviation(errors)
    print('Sample standard deviation =', sad)

    MAD, MdAD = mean_median_absolute_deviation_sample_median(errors)
    print('Mean/Median absolute deviation from the sample median =', MAD, MdAD)

    horizon_units, horizon_percentage_units = prognostics_horizon(errors, actual_ruls, units, bound_late=5, bound_early=10)
    print(horizon_units)
    print(horizon_percentage_units)

    prediction_spread_per_unit = prediction_spread(predicted_ruls, times, units)
    print(len(prediction_spread_per_unit), prediction_spread_per_unit)

    alpha_early_predictions = config.alpha_early_predictions
    alpha_late_predictions = config.alpha_late_predictions

    alpha_lambda_accuracy_all_units, alpha_lambda_accuracy_avg_over_all_units = alpha_lambda_accuracy(actual_ruls, predicted_ruls, units, errors, alpha_early_predictions=alpha_early_predictions, alpha_late_predictions=alpha_late_predictions)
    print(alpha_lambda_accuracy_all_units)
    print('Alpha lambda summary from EoL to beginning', alpha_lambda_accuracy_avg_over_all_units)

    alpha_lambda_precision_for_all_units, alpha_lambda_precision_avg_over_all_units = alpha_lambda_precision(actual_ruls, predicted_ruls, times, units, errors)
    print('alpha lambda precision avg over all units', alpha_lambda_precision_avg_over_all_units)
    print('alpha lambda precision for all units', alpha_lambda_precision_for_all_units)

    print(df.head())





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
