###### Load Libraries/Setup

library(tidyverse)
library(rsample)
library(riingo)
library(furrr)
library(tictoc)
library(lubridate)
library(reticulate)
library(tensorflow)
library(keras)
library(tfprobability)
library(recipes)
library(AlpacaforR)
plan(multiprocess)

use_condaenv('tensorflow2gpu', required = TRUE)


###### Pull Alpaca Stock Info
acct <- get_account()

assets <- get_assets()

buying_power <- acct$buying_power


## Make sure Tickers are Alpaca Tradeable
supported <- supported_tickers() %>%
    filter(exchange %in% c('NYSE', 'NASDAQ'),
           endDate >= Sys.Date()-days(5),
           startDate <= Sys.Date()-days(60),
           !ticker %in% c('PKDC', 'RGSE', 'TGE'),
           ticker %in% assets$symbol)


## Pull Time Series functions
get_current_prices <- function(ticker){
    dat <- riingo_prices(ticker, start_date = Sys.Date()-45) %>%
        select(ticker:volume) %>%
        group_by(ticker) %>%
        mutate(idx=nrow(.):1,
               last_open=last(open),
               open=open/last(open),
               last_high=last(high),
               high=high/last(high),
               last_low=last(low),
               low=low/last(low),
               last_close=last(close),
               close=close/last(close),
               last_volume=last(volume),
               volume=volume/last(volume),
               n_obs=n())
    return(dat)
}

## Pull Data

df <- future_map_dfr(supported$ticker, get_current_prices, .progress = TRUE)


## Transform Data
df <- df %>%
    filter(n_obs>20) %>%
    top_n(20, date) %>%
    select(-date, -n_obs) %>%
    distinct() %>%
    pivot_wider(names_from = idx, values_from = close:volume, names_sep = "") %>%
    ungroup()


saveRDS(df, 'daily_stock_ts.RDS')

#df <- readRDS('daily_stock_ts.RDS')


###### Feature Engineering

x_ts <- df %>%
    ungroup() %>%
    na.omit() %>%
    select(open20:open2, high20:high2, low20:low2, close20:close2, volume20:volume2) %>%
    as.matrix()

x <- array_reshape(x_ts, dim = c(dim(x_ts)[1], 5, dim(x_ts)[2]/5))

x_supp <- df %>%
    ungroup() %>%
    na.omit() %>%
    select(last_close, last_volume, last_open, last_low, last_high)

rec <- readRDS('stonk_recipe.RDS')

x_supp <- bake(rec, new_data = x_supp, composition = 'matrix')



#### Specify Model Structure ####
ts_input <- layer_input(c(dim(x)[2], dim(x)[3]), name = 'ts_in') 

ts_lstm <- ts_input %>%
    layer_cudnn_lstm(units=50, name = 'lstm_1', return_sequences = TRUE) %>%
    layer_cudnn_lstm(units = 50, name = 'lstm_2')

supp_input <- layer_input(shape = ncol(x_supp), name = 'supp_in')

concat <- layer_concatenate(list(ts_lstm, supp_input), name = 'concat')

out_layers <- concat %>%
    layer_dense(units=64, activation = 'relu') %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units=16, activation = 'relu') %>%
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


### Load Weights
load_model_weights_tf(model_supp, 'stonk_weights_supp.tf')




##### Score Model ####
pred_dist <- model_supp(list(tf$constant(x), tf$constant(x_supp)))

loc <- pred_dist$loc %>% as.numeric()
quantile(loc, probs = seq(0, 1, .1))
sd <- pred_dist$scale %>% as.numeric()
quantile(sd, probs = seq(0, 1, .1))
skewness <- pred_dist$skewness %>% as.numeric()
quantile(skewness, probs = seq(0, 1, .1))
tailweight <- pred_dist$tailweight %>% as.numeric()
quantile(tailweight, probs = seq(0, 1, .1))


pred_df1 <- tibble(loc, sd, skewness, tailweight)

quant10 <- pred_dist$quantile(.1) %>% as.numeric()
quant25 <- pred_dist$quantile(.25) %>% as.numeric()
quant50 <- pred_dist$quantile(.5) %>% as.numeric()
quant75 <- pred_dist$quantile(.75) %>% as.numeric()
quant90 <- pred_dist$quantile(.9) %>% as.numeric()

quant10_log <- log(quant10, base = 10)
quant25_log <- log(quant25, base = 10)
quant50_log <- log(quant50, base = 10)
quant75_log <- log(quant75, base = 10)
quant90_log <- log(quant90, base = 10)

pred0_90 <- 1-(pred_dist$cdf(.90) %>% as.numeric())
pred0_95 <- 1-(pred_dist$cdf(.95) %>% as.numeric())
pred0_99 <- 1-(pred_dist$cdf(.99) %>% as.numeric())
pred1_00 <- 1-(pred_dist$cdf(1) %>% as.numeric())
pred1_01 <- 1-(pred_dist$cdf(1.01) %>% as.numeric())
pred1_05 <- 1-(pred_dist$cdf(1.05) %>% as.numeric())
pred1_10 <- 1-(pred_dist$cdf(1.1) %>% as.numeric())


pred_df <- df %>%
    ungroup() %>%
    na.omit() %>%
    select(ticker, last_close, last_volume) %>%
    bind_cols(data.frame(loc, sd, skewness, tailweight,
                         quant10, quant25, quant50, quant75, quant90,
                         quant10_log, quant25_log, quant50_log, quant75_log, quant90_log,
                         pred0_90, pred0_95, pred0_99, pred1_00,
                         pred1_01, pred1_05, pred1_10)) 


pred_df_filter <- pred_df %>% filter(last_close>=2, last_volume>=10000, !grepl('-', ticker))


## Save Scored Files
saveRDS(pred_df_filter, 'pred_df_daily.RDS')