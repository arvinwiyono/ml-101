# Multiple / Multivariate Linear Regression

Linear Regression must follow these assumptions:

* Linearity
* Homoscedasticity
* Multivariate normality
* Independence of errors
* Lack of multicollinearity

## 50_Startups.csv
 > Profit = b0 + b1 * R&D Spend + b2 * Admin + b3 * Marketing + b4 * State(???)
 
 What should we place for the State column (string)?
 In this case, State is a categorical variable.
 
## Dummy Variables or One-Hot Encoding
 
 We need to use __Dummy Variables__
 
 We have two categories:
 
 - New York
 - California
 
 So add two new columns, New_York and California, which each accepts a value of either 1 or 0.

 So the new equation will be:
 > Profit = b0 + b1 * R&D Spend + b2 * Admin + b3 * Marketing + b4 * New_York
 
 Beware of dummy variable trap!
 Always omit one dummy variable.
 E.g. Have 8 dummy variables, must use 7 dummy variables
 
 ## Building a Model
 
 ### Backward Elimination
 
 Steps:
 
 1. Select a significance level  or confidence interval to stay in the model (e.g. SL = 0.05)
 2. Fit the model with all possible predictors
 3. Consider the predictor with the highest P-value, such that P-value > SL
 4. Remove the predictor because it is insignificant
 5. Re-fit the model without this variable
 6. Rinse and repeat until the highest P-val is less than the SL (go back to step 3)
 7. Model is ready


### Forward Selection

Steps:

1. Select a significane level to enter the model (e.g. SL = 0.05 or 5%)
2. Fit all simple regression models, select the one with the lowest P-val
3. Keep this variable and fit all possible models with one extra predictor addded to the one(s) we keep
4. If the new variable's P-val is less than SL, then go to step 3.
5. Stop when the remaining unselected variables have P-val > SL
6. Select the previous model instead of the current one.

### Bi-directional Elimination

Steps:

1. Set SLenter and SLstay
2. Perform a step of forward selection (P-val < SLenter)
3. Perform backward elimination (P-val < SLstay)
4. Stop when there is no new vars can enter and no old vars can leave

### All Possible Models (most resource consuming)

Overall, we want to compare the score of all possible models and select the best.

Steps:

1. Select a criterion of goodness of fit
2. Construct all possible regression models: 2^N-1 total combinations
3. Select the one with the best criterion