
library(reticulate)
library(tensorflow)
library(tfprobability)
library(keras)
library(tidyverse)
library(rsample)
library(lubridate)
library(recipes)

use_condaenv('tensorflow2gpu', required = TRUE)


multi <- readRDS('stonks_wide_all.RDS')


## Train/Test Split

ticker_split_df <- multi %>%
    group_by(ticker) %>%
    arrange(desc(last_date)) %>%
    select(ticker, out1) %>%
    slice(1) %>%
    ungroup()

split <- initial_split(ticker_split_df, strata = out1)

score_ticker <- multi %>%
    inner_join(select(testing(split), -out1)) %>%
    filter(ticker %in% sample(ticker, 100)) %>%
    .$ticker %>%
    unique()

## Extract Time Series Features
x_train <- multi %>%
    filter(year(last_date)<=2018) %>%
    na.omit() %>%
    select(open20:open2, high20:high2, low20:low2, close20:close2, volume20:volume2) %>%
    as.matrix()

x_train <- array_reshape(x_train, dim = c(dim(x_train)[1], 5, dim(x_train)[2]/5))

x_test <- multi %>%
    inner_join(select(testing(split), -out1)) %>%
    filter(year(last_date) %in% c(2018)) %>%
    na.omit() %>%
    select(open20:open2, high20:high2, low20:low2, close20:close2, volume20:volume2) %>%
    as.matrix()

x_test <- array_reshape(x_test, dim = c(dim(x_test)[1], 5, dim(x_test)[2]/5))

x_score <- multi %>%
    filter(ticker %in% score_ticker,
           year(last_date)>=2019) %>%
    na.omit() %>%
    select(open20:open2, high20:high2, low20:low2, close20:close2, volume20:volume2) %>%
    as.matrix()

x_score <- array_reshape(x_score, dim = c(dim(x_score)[1], 5, dim(x_score)[2]/5))


## Extract Non-TS Features
x_train_supp <- multi %>%
  inner_join(select(training(split), -out1)) %>%
  filter(year(last_date)<=2018) %>%
  na.omit() %>%
  select(last_close:last_high)

x_test_supp <- multi %>%
    inner_join(select(testing(split), -out1)) %>%
    filter(year(last_date) %in% c(2018)) %>%
    na.omit() %>%
    select(last_close:last_high)

x_score_supp <- multi %>%
    filter(ticker %in% score_ticker,
           year(last_date)>=2019) %>%
    na.omit() %>%
    select(last_close:last_high)


rec <- recipe(~., data = x_train_supp) %>%
    step_log(all_predictors()) %>%
    step_scale(all_predictors()) %>%
    step_center(all_predictors()) %>%
    prep(training=x_train_supp, retain=FALSE)

saveRDS(rec, 'stonk_recipe.RDS')


x_train_supp <- bake(rec, new_data = x_train_supp, composition = 'matrix')
x_test_supp <- bake(rec, new_data = x_test_supp, composition = 'matrix')
x_score_supp <- bake(rec, new_data = x_score_supp, composition = 'matrix')


## Extract Outcomes
y_train <- multi %>%
    inner_join(select(training(split), -out1)) %>%
    filter(year(last_date)<=2018) %>%
    na.omit() %>%
    .$out1

y_test <- multi %>%
    inner_join(select(testing(split), -out1)) %>%
    filter(year(last_date) %in% c(2018)) %>%
    na.omit() %>%
    .$out1

y_score <- multi %>%
    filter(ticker %in% score_ticker,
           year(last_date)>=2019) %>%
    na.omit() %>%
    .$out1





## Model - With Supplementary features ###################
## Sinh Arcsinh Model

ts_input <- layer_input(c(dim(x_train)[2], dim(x_train)[3]), name = 'ts_in') 

ts_lstm <- ts_input %>%
    layer_cudnn_lstm(units=50, name = 'lstm_1', return_sequences = TRUE) %>%
    layer_cudnn_lstm(units = 50, name = 'lstm_2')

supp_input <- layer_input(shape = ncol(x_train_supp), name = 'supp_in')

concat <- layer_concatenate(list(ts_lstm, supp_input), name = 'concat')

out_layers <- concat %>%
    layer_dense(units=64, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units = 32, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units=16, activation = 'relu', regularizer_l1_l2()) %>%
    #layer_dense(units=1, activation = 'linear') 
    layer_dense(units = 4, activation = "linear") %>%
    layer_distribution_lambda(function(x) {
        tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                         scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                         skewness=x[, 3, drop=FALSE],
                         tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
    }
    )

model_supp <- keras_model(
    inputs = c(ts_input, supp_input),
    outputs = out_layers
)


negloglik <- function(y, model) - (model %>% tfd_log_prob(y))

learning_rate <- 0.001
model_supp %>% compile(optimizer = optimizer_adam(lr = learning_rate), loss = negloglik)

history <- model_supp %>% fit(x=list(x_train, x_train_supp), 
                         y=list(y_train),
                         shuffle=TRUE,
                         validation_data = list(list(x_test, x_test_supp), y_test),
                         epochs = 500, 
                         batch_size=5000, 
                         callbacks=list(callback_early_stopping(monitor='loss', patience = 20)),
)


save_model_weights_tf(model_supp, 'stonk_weights_supp.tf', overwrite = TRUE)



pred_dist <- model_supp(list(tf$constant(x_score), tf$constant(x_score_supp)))

loc <- pred_dist$loc %>% as.numeric()
quantile(loc, probs = seq(0, 1, .1))
sd <- pred_dist$scale %>% as.numeric()
quantile(sd, probs = seq(0, 1, .1))
skewness <- pred_dist$skewness %>% as.numeric()
quantile(skewness, probs = seq(0, 1, .1))
tailweight <- pred_dist$tailweight %>% as.numeric()
quantile(tailweight, probs = seq(0, 1, .1))

pred1 <- pred_dist$cdf(1) %>% as.numeric()
pred1_1 <- pred_dist$cdf(1.1) %>% as.numeric()
quant10 <- pred_dist$quantile(.1) %>% as.numeric()
quant25 <- pred_dist$quantile(.25) %>% as.numeric()
quant50 <- pred_dist$quantile(.5) %>% as.numeric()
quant75 <- pred_dist$quantile(.75) %>% as.numeric()
quant90 <- pred_dist$quantile(.9) %>% as.numeric()


pred_df <- data.frame(loc, sd, skewness, tailweight, pred1, pred1_1,
                      quant10, quant25, quant50, quant75, quant90, actual=y_score)

saveRDS(pred_df, 'pred_stocks_new.RDS')


pred_df$up_10 <- as.numeric(pred_df$actual>pred_df$quant10)

pred_df$up_25 <- as.numeric(pred_df$actual>pred_df$quant25)

pred_df$up_50 <- as.numeric(pred_df$actual>pred_df$quant50)

pred_df$up_75 <- as.numeric(pred_df$actual>pred_df$quant75)

pred_df$up_90 <- as.numeric(pred_df$actual>pred_df$quant90)

mean(pred_df$up_10)
mean(pred_df$up_25)
mean(pred_df$up_50)
mean(pred_df$up_75)
mean(pred_df$up_90)


pred_df %>%
    select(up_10:up_90) %>%
    summarize_each(funs = mean) %>%
    pivot_longer(up_10:up_90, names_to = 'over_quantile', values_to = 'pct') %>%
    ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
    geom_col() +
    geom_text(vjust=1, color='white') +
    theme_minimal() +
    scale_y_continuous(labels=scales::percent) +
    ggtitle('% Stocks Going Over Specified Quantile')


qplot(pred_df$quant50)

qplot(pred_df$quant50, pred_df$actual) +
    coord_cartesian(xlim = c(.9, 1.1), ylim = c(.9, 1.1))


pred_df %>%
    mutate(spread_50=quant75-quant25,
           spread_80=quant90-quant10) %>%
    ggplot(aes(x=spread_50, y=actual)) +
    geom_point(alpha=.2) +
    geom_smooth() +
    scale_x_log10() +
    scale_y_log10()


## LogProb Test

dist <- tfd_sinh_arcsinh(loc = pred_df$loc[3372], 
                 scale = pred_df$sd[3372],
                 skewness = pred_df$skewness[3372], 
                 tailweight = pred_df$tailweight[3372])

dist$prob(pred_df$actual[3372])

tfd_sample(distribution = dist, 1000) %>% as.numeric() %>% qplot()


#### Graveyard #######################################



## Point Model
model_point <- keras_model_sequential() %>%
    #layer_conv_1d(filters = 10, kernel_size = 5, input_shape = c(dim(x_train)[2], dim(x_train)[3])) %>%
    layer_gru(units=16, input_shape = c(dim(x_train)[2], dim(x_train)[3]), regularizer_l1_l2()) %>%
    #layer_dense(units=16, activation = 'relu') %>%
    #layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units=16, activation = 'relu', regularizer_l1_l2()) %>%
    #layer_dense(units=1, activation = 'linear') 
    layer_dense(units = 1, activation = "linear") 


learning_rate <- 0.001
model_point %>% compile(optimizer = optimizer_adam(lr = learning_rate), loss = 'mse')

history <- model_point %>% fit(x_train, y_train, 
                               validation_data = list(x_test, y_test),
                               epochs = 20, 
                               batch_size=1000, 
                               callbacks=list(callback_early_stopping(monitor='val_loss', patience = 20)),
)


## Model_Normal
model_normal <- keras_model_sequential() %>%
    #layer_conv_1d(filters = 10, kernel_size = 5, input_shape = c(dim(x_train)[2], dim(x_train)[3])) %>%
    layer_lstm(units=16, input_shape = c(dim(x_train)[2], dim(x_train)[3]), regularizer_l1_l2()) %>%
    #layer_dense(units=16, activation = 'relu') %>%
    #layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units=16, activation = 'relu', regularizer_l1_l2()) %>%
    #layer_dense(units=1, activation = 'linear') 
    layer_dense(units = 2, activation = "linear") %>%
    layer_distribution_lambda(function(x) {
        tfd_normal(loc = x[, 1, drop = FALSE],
                   scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]))
    }
    )

negloglik <- function(y, model) - (model %>% tfd_log_prob(y))

learning_rate <- 0.001
model_normal %>% compile(optimizer = optimizer_adam(lr = learning_rate), loss = negloglik)

history <- model_normal %>% fit(x_train, y_train, 
                                validation_data = list(x_test, y_test),
                                epochs = 500, 
                                batch_size=1000, 
                                callbacks=list(callback_early_stopping(monitor='val_loss', patience = 20)),
)




## Sinh Arcsinh Model - Time Series Only
model <- keras_model_sequential() %>%
    #layer_conv_1d(filters = 10, kernel_size = 5, input_shape = c(dim(x_train)[2], dim(x_train)[3])) %>%
    layer_cudnn_lstm(units=64, input_shape = c(dim(x_train)[2], dim(x_train)[3]), return_sequences = T) %>%
    layer_cudnn_lstm(units=64) %>%
    layer_dense(units=64, activation = 'relu') %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units=16, activation = 'relu') %>%
    #layer_dense(units=1, activation = 'linear') 
    layer_dense(units = 4, activation = "linear") %>%
    layer_distribution_lambda(function(x) {
        tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                         scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                         skewness=x[, 3, drop=FALSE],
                         tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
    }
    )

negloglik <- function(y, model) - (model %>% tfd_log_prob(y))

learning_rate <- 0.001
model %>% compile(optimizer = optimizer_adam(lr = learning_rate), loss = negloglik)

history <- model %>% fit(x_train, y_train, 
                         validation_data = list(x_test, y_test),
                         epochs = 500, 
                         batch_size=5000, 
                         callbacks=list(callback_early_stopping(monitor='val_loss', patience = 20)),
)

#save_model_hdf5(model, 'nasdaq_nyse_stock_pred.hdf5', overwrite = TRUE)
save_model_weights_tf(model, 'stonk_weights.tf', overwrite = TRUE)



model2 <- keras_model_sequential() %>%
    #layer_conv_1d(filters = 10, kernel_size = 5, input_shape = c(dim(x_train)[2], dim(x_train)[3])) %>%
    layer_cudnn_lstm(units=64, input_shape = c(dim(x_train)[2], dim(x_train)[3]), return_sequences = T) %>%
    layer_cudnn_lstm(units=64) %>%
    layer_dense(units=64, activation = 'relu') %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units=16, activation = 'relu') %>%
    #layer_dense(units=1, activation = 'linear') 
    layer_dense(units = 4, activation = "linear") %>%
    layer_distribution_lambda(function(x) {
        tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                         scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                         skewness=x[, 3, drop=FALSE],
                         tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
    }
    )

load_model_weights_tf(model2, filepath = 'stonk_weights.tf')

pred_dist <- model2(tf$constant(x_score))

loc <- pred_dist$loc %>% as.numeric()
quantile(loc, probs = seq(0, 1, .1))
sd <- pred_dist$scale %>% as.numeric()
quantile(sd, probs = seq(0, 1, .1))
skewness <- pred_dist$skewness %>% as.numeric()
quantile(skewness, probs = seq(0, 1, .1))
tailweight <- pred_dist$tailweight %>% as.numeric()
quantile(tailweight, probs = seq(0, 1, .1))

pred1 <- pred_dist$cdf(1) %>% as.numeric()
pred1_1 <- pred_dist$cdf(1.01) %>% as.numeric()
quant10 <- pred_dist$quantile(.1) %>% as.numeric()
quant25 <- pred_dist$quantile(.25) %>% as.numeric()
quant50 <- pred_dist$quantile(.5) %>% as.numeric()
quant75 <- pred_dist$quantile(.75) %>% as.numeric()
quant90 <- pred_dist$quantile(.9) %>% as.numeric()


pred_df <- data.frame(loc, sd, skewness, tailweight, pred1, pred1_1,
                      quant10, quant25, quant50, quant75, quant90, actual=y_score)

saveRDS(pred_df, 'pred_stocks_new.RDS')


mean(pred_df$up_10 <- as.numeric(pred_df$actual>pred_df$quant10))

mean(pred_df$up_25 <- as.numeric(pred_df$actual>pred_df$quant25))

mean(pred_df$up_50 <- as.numeric(pred_df$actual>pred_df$quant50))

mean(pred_df$up_75 <- as.numeric(pred_df$actual>pred_df$quant75))

mean(pred_df$up_90 <- as.numeric(pred_df$actual>pred_df$quant90))


pred_df %>%
    select(up_10:up_90) %>%
    summarize_each(funs = mean) %>%
    pivot_longer(up_10:up_90, names_to = 'over_quantile', values_to = 'pct') %>%
    ggplot(aes(x=over_quantile, y=pct, label=scales::percent(pct, accuracy = .01))) +
    geom_col() +
    geom_text(vjust=1, color='white') +
    theme_minimal() +
    scale_y_continuous(labels=scales::percent) +
    ggtitle('% Stocks Going Over Specified Quantile')


qplot(pred_df$quant50)

qplot(pred_df$quant50, pred_df$actual) +
    coord_cartesian(xlim = c(.9, 1.1), ylim = c(.9, 1.1)) +
    geom_point(alpha=.2)


pred_df %>%
    mutate(spread_50=quant75-quant25,
           spread_80=quant90-quant10) %>%
    ggplot(aes(x=spread_80, y=actual)) +
    geom_point(alpha=.2) +
    geom_smooth() +
    scale_x_log10() +
    scale_y_log10()


