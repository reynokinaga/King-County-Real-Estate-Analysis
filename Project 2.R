#------------------------------------------------------------------------------
# Libraries
library(caret)
library(ROSE)
library(randomForest)
library(rpart)
library(rpart.plot)
library(forecast)
library(treeClust)
library(car)
library(dplyr)
library(tidyr)
library(lm.beta)
library(lmtest)
library(gvlma)

# Load Data
house <- read.csv("/Users/reyn/Documents/SU SR Year/Fall Quarter/Data Mining/Projects/house_27.csv", header = TRUE)
house_test <- read.csv("/Users/reyn/Documents/SU SR Year/Fall Quarter/Data Mining/Projects/house_test_27.csv", header = TRUE)

str(house)
str(house_test)

table(house$floors)
table(house_test$floors)

#------------------------------------------------------------------------------
#Training/Validation Set
set.seed(666)

training_index <- sample(1:nrow(house), 0.6 * nrow(house))
validation_index <- setdiff(1:nrow(house), training_index)

training_df <- house[training_index, ]
validation_df <- house[validation_index, ]


# Variables Chosen:
#  bedrooms: Number of Bedrooms/House
#  bathrooms: Number of bathrooms/House
#  sqft_living: square footage of the home
#  floors: Total floors (levels) in house
#  Condition: How good the condition is (Overall). Sale of 1:5/ 5 is very good.
#  Grade: overall grade given to the housing unit, based on King County 
# grading system. The higher the better
#  yr_built: Year built

#------------------------------------------------------------------------------
# Regression Tree
regress_tree <- rpart(price ~ bedrooms + bathrooms + sqft_living + 
                        floors + condition + grade + yr_built,
                      data = training_df, method = "anova", maxdepth = 20)
rpart.plot(regress_tree, type = 4)

predict_training <- predict(regress_tree, training_df)
accuracy(predict_training, training_df$price)

predict_validation <- predict(regress_tree, validation_df)
accuracy(predict_validation, validation_df$price)

which_node_train <- rpart.predict.leaves(regress_tree, newdata = training_df,
                                         type = "where")
head(which_node_train)

sd_node = aggregate(training_df$price, list(which_node_train), FUN = sd)
names(sd_node) <- c("Node", "sd")
sd_node

min_node = aggregate(training_df$price, list(which_node_train), FUN = min)
names(min_node) <- c("Node", "min")
min_node

max_node = aggregate(training_df$price, list(which_node_train), FUN = max)
names(max_node) <- c("Node", "max")
max_node

mean_node = aggregate(training_df$price, list(which_node_train), FUN = mean)
names(mean_node) <- c("Node", "mean")
mean_node

regress_tree_pred <- predict(regress_tree, newdata = house_test)
regress_tree_pred

house_test_node <- rpart.predict.leaves(regress_tree, 
                                  newdata = house_test,
                                  type = "where")
house_test_node

house_test_reg_pred <- data.frame(Node = house_test_node,
                            Prediction = regress_tree_pred)
house_test_reg_pred

house_test_reg_pred_range <- house_test_reg_pred %>%
  inner_join(min_node, by = "Node") %>%
  inner_join(max_node, by = "Node") %>%
  inner_join(sd_node, by = "Node")

house_test_reg_pred_range
#------------------------------------------------------------------------------
# Multi-linear Regression
house_2 <- house[, c("bedrooms", "bathrooms", "sqft_living",
                           "floors", "condition", "grade", "yr_built", "price")]
head(house_2)

house_2 <- house_2 %>% mutate_if(is.character, as.numeric)
str(house_2)

price_model_st <- lm(price ~ bedrooms + bathrooms + sqft_living + floors
                    + condition + grade + yr_built,
                    data = house_2)
summary(price_model_st)
coef(price_model_st)
confint(price_model_st, level = 0.95)

price_model_st_std <- lm.beta::lm.beta(price_model_st)
coef(price_model_st_std)
confint(price_model_st_std)

price_model_st_residuals <- rstandard(price_model_st)
head(price_model_st_residuals)
house_st_comb_2 <- cbind(house_2, price_model_st_residuals)
head(house_st_comb_2)
vif(price_model_st)
durbinWatsonTest(price_model_st)
bptest(price_model_st)
gvlma(price_model_st)

price_model_st_2 <- lm(price ~ bedrooms + bathrooms + sqft_living + floors
                       + condition + grade + yr_built,
                       data = training_df)
summary(price_model_st_2)
# The only variable correlation that doesn't make sense is bedrooms, as it has
# a negative coeffecient. This means that the less number of bathrooms, the 
# higher the price. This breaks logical thinking as most people would think 
# that the more bathrooms, the more expensive a house is. One explanation
# for this is that a one-bedroom house in the busy city costs more than a 
# two-bedroom house in a rural and less populated area. However, since our
# model does not factor in location, it does not control for this issue. 
price_model_st_2_pred_train <- predict(price_model_st_2,
                                      training_df)

accuracy(price_model_st_2_pred_train, training_df$price)
price_model_st_2_pred_valid <- predict(price_model_st_2,
                                      validation_df)
accuracy(price_model_st_2_pred_valid, validation_df$price)

max(training_df$price) - min(training_df$price)
sd(training_df$price)
max(validation_df$price) - min(validation_df$price)
sd(validation_df$price)
vif(price_model_st_2)
# The VIF of all the values are less than 10, suggesting that there is no 
# multicolinearity. Multicolinearity is where there is a strong correlation
# between two independent variables, which can lead to skewed or misleading 
# data. 
durbinWatsonTest(price_model_st_2)
# The D-W value is 1.98, which is less than or equal to 2, which means that
# there is a positive autocorrelation. Autocorrelation measures the relationship
# between a variable's current value and its past values, making it an indicator
# of prediction models. However, this does not apply to this situation due to 
# not involving time series data. 
bptest(price_model_st_2)
# Since the p-value of the Breusch-Pagan test is less than 0.05, we can reject
# the null hypothesis and conclude that heteroscedasticity is present. 
# Heteroscedasticity is where the standard deviations of a predicted variable
# are non-constant over a time. 
price_model_st_2_pred_new <- predict(price_model_st_2,
                                    newdata = house_test, interval = 
                                      "confidence")
# Out of all our models, this is the best model and it predicted the price 
# of the new houses to be the following:
price_model_st_2_pred_new
# The model predicts the price value in the fit column in our final prediction.
# The lower column predicts the lower confidence limit of 95% in our final 
# prediction. The upper column predicts the upper confidence limit of 95% in 
# our final prediction. These limits serve us as a range of values that we can
# be 95% confident that the price value is between the two numbers of the lower
# and upper limit. A. multilinear regression is determined by a constant added
# together with the coefficients times the predictors added with the error
# (noise) which produces the outcome variable, in this case price. 

#------------------------------------------------------------------------------
## Writeup
# The dataset (house) is about homes in King County and the different factors 
# that attribute to the homes. Our objective for this project is to create a 
# model that can predict the prices of new homes in a new dataset (house_test) 
# based on the given dataset's (house) variables due to starting up a 
# real estate business. 

# The variables we chose that were most important to include in our model was: 
# bedrooms, bathrooms, sqft_living, floors, condition, grade, and 
# yr_built. We chose these because after further research of what determines 
# the prices of homes, we concluded that the size and usable space of the 
# house and the age and condition are major factors of determining the price 
# of a house. The number of bathrooms, floors, and 
# bedrooms, and the square footage of the home(sqft_living) all relate 
# to the size and usable space of the house. The condition, grade given 
#to the housing unit by the King County System (grade), and the year the 
# house was built all relate to the age and condition of the home.

# After, we transformed the data by removing all the unnecessary variables in
# our model. We did this because it is not important for our models. 

# The models we chose were a regression tree and a multilinear regression 
# because the outcome variable or dependent variable (price) is continuous and 
# numerical. 

# After validating the model, the multilinear regression  was our best 
# because of the lower RMSE values compared to the regression tree. The RMSE is
# the root mean squared error that is one measure of error of a model. Also,
# the p-value of the multilinear regression was also less than 0.05 and 
# therefore is significant. It also gives a range of predictions to show where
# it can lie between the upper and lower limits besides its predicted point, 
# while the regression tree gives a range of minimum, maximum, standard deviation,
# and average price, which can be skewed by outliers. The regression tree is
# computed by calculating the relative importance of predictors by summing
# up the overall reduction of optimization criteria. The data is divided into 
# branches, nodes, and leaves. The target attribute values can be predicted 
# from their mean values in the leaves. 

# A limit to these model is overfitting as adding more variables can cause 
# overfitting. This overfitting can lead to error and not an accurate 
# predictions of the data. 

# As a side note, location (zipcode, lat, and long) was not included in our 
# model due to the specific models we used and how difficult it would be to use
# for our specific models we tested. 
