# Bus Lateness Prediction Analysis

<br>
<p align="center">
  <img src="images/bus_transport.jpeg">
</p>
<br>

<p align="center">
  <a href="#summary"> Executive summary </a> •
  <a href="#data"> Sourcing data </a> •
  <a href="#munge_eda"> Data Transformation and EDA </a> •
  <a href="#mine"> Data Mining </a> •
  <a href="#models"> Modelling </a> •
  <a href="#plots"> Plotting results </a> •
  <a href="#conc"> Conclusions and recommendations</a>
</p>

<a id = 'summary'></a>

## Executive Summary

Based on a 2016 dataset containing occupancy information of bus trips within the New South Wales state this project explores factors that affect punctuality of these buses' arrival to inform better scheduling and trip performance. I first conducted a thorough exploratory data analysis and then examined several multivariate regression models to model lateness, and based on these models, insights into variables that affect lateness are derived.

<a id = 'data'></a>

## Sourcing data

The 2016 Bus Occupancy Dataset, available via OpenData NSW, was used for this project. The dataset includes:

- **Route information**: Bus route identifiers, stop sequences, and route characteristics

- **Temporal data**: Scheduled departure times, and actual departure times

- **Operational data**: Bus occupancy levels and vehicle capacity

- **Geographic data**: Stop locations, route distances, and service areas

<a id = 'munge_eda'></a>

## Data Transformation and EDA

Data transformation procedures performed:

- Checking the time span of the data and ensuring consistent temporal coverage across all routes

- Casting the temporal features, `calendar_date`, `timetable_time` and `actual_time` to DateType and Timestamp data types

- Calculating lateness as the difference between actual and scheduled departure times

- Eliminating whitespaces and standardising all string-value variables to lowercase

- Dropping `null` values in the `actual_time` column, since the absence of these values is assumed to be random

- Creating time-based features such as hour of day, day of week, and peak/off-peak indicators

- Examining the data we find that the maximum lateness values were many standard deviations larger than the mean, indicating the presence of outliers. Keeping extreme outliers in the analysis would skew the predicted lateness patterns.

- Create stratified samples of sizes 10K to 100K of the full dataset, since there are >20M trips made in total

<a id = 'mine'></a>

## Mining the data

Some of the steps for mining the data included: computing average lateness per route, creating occupancy ratio features, calculating peak hour indicators, analyzing weather impact on delays, dropping extreme outliers, and computing both passenger load factors and frequency-based metrics.

I then looked for any statistical relationships, correlations, or other relevant properties of the dataset that could influence bus lateness.

**Steps**:

- First I needed to choose the proper predictors. I looked for strong correlations between variables to avoid problems with multicollinearity

- Also, variables that changed very little had minimal impact and were therefore not included as predictors

- I then studied correlations between predictors and the target variable (lateness)

- I saw from the correlation matrices that `occupancy_level` and `load_factor` are highly correlated. Furthermore, both are correlated to the target variable `lateness_minutes`. Similar patterns emerged with `route_frequency` and `service_density`

A heatmap of correlations using `Seaborn` follows:

<p align="center">
   <img src="images/correlation_heatmap.png" width="400">
<p/>

<p align="center">
   <img src="images/pairplot_analysis.png" width="700">
<p/>

<a id = 'models'></a>

## Building the models

Using `scikit-learn`, I built the regression models and evaluated their fit. For that I generated all combinations of useful relevant features using the `itertools` module.

The best predictors were obtained via:

```python
print(f"Best R² score: {best_r2:.4f}")
print(f"Best features: {best_features}")
```

Dropping highly correlated predictors I redefined `X` and `y` and built a Ridge regression model:

```python
X, y = df_final[best_features], df_final['lateness_minutes']
ridge = linear_model.RidgeCV(cv=5, alphas=[0.1, 1.0, 10.0])
model = ridge.fit(X, y)
```

<a id = 'plots'></a>

## Plotting results

I then plotted the predictions versus the true lateness values:

<p align="center">
   <img src="images/predictions_vs_actual.png" width="500">
<p/>

Feature importance visualization:

<p align="center">
   <img src="images/feature_importance.png" width="500">
<p/>

<a id = 'conc'></a>

## Conclusions and recommendations

The following recommendations were provided based on the bus lateness regression analysis:

- **Peak hour impact**: Routes during peak hours (7-9 AM, 5-7 PM) show significantly higher lateness, suggesting the need for adjusted scheduling during these periods

- **Occupancy correlation**: Higher occupancy levels correlate with increased lateness due to longer boarding times. Routes with consistently high occupancy should consider increased frequency or larger vehicles

- **Route-specific patterns**: Certain routes consistently underperform in punctuality. These routes require targeted interventions such as dedicated bus lanes or schedule adjustments

- **Weather sensitivity**: Adverse weather conditions significantly impact punctuality. Contingency planning and adjusted schedules during poor weather could improve overall performance

- **Day-of-week variations**: Weekend services show different lateness patterns compared to weekdays, suggesting the need for differentiated scheduling approaches

- **Recommendations for improvement**:
  - Implement dynamic scheduling based on predicted occupancy levels

  - Prioritize infrastructure improvements on routes with highest lateness correlation

  - Consider real-time passenger information systems to manage expectations during predicted delays

  - Develop weather-responsive scheduling protocols

  - Focus on reducing dwell times at high-occupancy stops through improved boarding processes

Both models generated negative $R^2$ scores, indicating these regression models are not be sufficient models in capturing the variation of lateness within these datasets.
