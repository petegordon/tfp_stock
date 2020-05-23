#config file with API KEYS
source("config.R")

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

use_condaenv('r-reticulate', required = TRUE)

py_config()

tf_config()

#tf_gpu_configured()

#md <- keras_model_sequential %>% layer_dense(units=2, input_share=5, activation='relu')




###### Pull Alpaca Stock Info
acct <- get_account()

assets <- get_assets()

calendar <- get_calendar(from = Sys.Date()-days(70), to=Sys.Date()) %>%
    arrange(desc(date)) %>%
    slice(1:43) 

trading_day <- Sys.Date() + days(1)

#trading_day <- as.Date('2020-05-18')

## Make sure Tickers are Alpaca Tradeable
supported <- supported_tickers() %>%
    filter(exchange %in% c('NYSE', 'NASDAQ', 'NYSE ARCA'),
           endDate >= Sys.Date()-days(5),
           startDate <= min(calendar$date),
           ticker %in% assets$symbol
    )




get_current_prices <- function(ticker){
    dat <- riingo_prices(ticker, start_date = min(calendar$date)) %>%
        select(ticker:volume) %>%
        group_by(ticker) %>%
        mutate(idx=nrow(.):1,
               open_last=last(open),
               #open=open/last(open),
               high_last=last(high),
               #high=high/last(high),
               low_last=last(low),
               #low=low/last(low),
               close_last=last(close),
               #close=close/last(close),
               volume_last=last(volume),
               #volume=volume/last(volume),
               n_obs=n())
    return(dat)
}




## Pull Data

df <- future_map_dfr(supported$ticker, get_current_prices, .progress = TRUE)


saveRDS(df, 'daily_stock_ts.RDS')

df <- readRDS('daily_stock_ts.RDS')


df <- df %>%
    filter(n_obs>=40) %>%
    top_n(40, date) %>%
    select(-date, -n_obs) %>%
    mutate(volume=ifelse(volume==0, 1, volume)) %>%
    mutate_at(vars(close:volume), log) %>%
    distinct() %>%
    pivot_wider(names_from = idx, values_from = close:volume, names_sep = "") %>%
    ungroup()



x_ts <- df %>%
    ungroup() %>%
    na.omit() %>%
    select(open40:open1, high40:high1, low40:low1, close40:close1, volume40:volume1) %>%
    as.matrix()

x <- array_reshape(x_ts, dim = c(dim(x_ts)[1], 5, dim(x_ts)[2]/5))

x_supp <- df %>%
    ungroup() %>%
    na.omit() %>%
    left_join(supported) %>%
    mutate(day_of_week = as.character(wday(trading_day, label = TRUE)),
           yrs_since_start = as.numeric(year(trading_day)-year(startDate)),
           yrs_since_start_cat = case_when(
               yrs_since_start==0 ~ 'same_year',
               yrs_since_start>=1 & yrs_since_start<5 ~ 'five_or_less',
               yrs_since_start>=5 & yrs_since_start<10 ~ 'five_to_ten',
               yrs_since_start>=10 & yrs_since_start<20 ~ 'ten_to_twenty', 
               yrs_since_start>=20 ~ 'more_than_twenty'
           )) %>%
    select(day_of_week, assetType, yrs_since_start_cat)




transform_recipe <- readRDS('supp_transform_recipe.RDS')


x_supp <- bake(transform_recipe, new_data=x_supp, composition = 'matrix')
apply(x_supp, 2, function(x) sum(is.na(x)))

x_supp[is.na(x_supp)] <- 0



ts_input <- layer_input(c(dim(x)[2], dim(x)[3]), name = 'ts_in') 

ts_lstm <- ts_input %>%
    bidirectional(layer_lstm(units = 128, name = 'lstm_1', return_sequences=TRUE, recurrent_regularizer = regularizer_l2())) %>%
    layer_lstm(units=64, name = 'lstm_2', recurrent_regularizer = regularizer_l2())


supp_input <- layer_input(shape = ncol(x_supp), name = 'supp_in')

supp_layers <- supp_input %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(32, activation='relu')

concat <- layer_concatenate(list(ts_lstm, supp_layers), name = 'concat')

out_layers <- concat %>%
    layer_dense(units=128, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units = 64, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units = 128, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units = 64, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units=32, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units=32, activation = 'relu', regularizer_l1_l2()) %>%
    layer_dense(units = 4, activation = "linear") %>%
    layer_distribution_lambda(function(x) {
        tfd_sinh_arcsinh(loc = x[, 1, drop = FALSE],
                         scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE]),
                         skewness=x[, 3, drop=FALSE],
                         tailweight= 1e-3 + tf$math$softplus(x[, 4, drop = FALSE]))
    }
    )

model_supp <- keras_model(
    inputs = list(ts_in=ts_input, supp_in=supp_input),
    outputs = out_layers
)
load_model_weights_tf(model_supp, 'stonk_weights_v2.tf')



pred_dist <- model_supp(list(tf$constant(x), tf$constant(x_supp)))




out_df <- tibble(
    
    ticker = df$ticker,
    
    loc = pred_dist$loc %>% as.numeric(),
    scale = pred_dist$scale %>% as.numeric(),
    skewness = pred_dist$skewness %>% as.numeric(),
    tailweight = pred_dist$tailweight %>% as.numeric(),
    
    close_last = df$close_last,
    volume_last = df$volume_last,
    
    quant10_pct = pred_dist$quantile(.1) %>% as.numeric(), #%>% exp(),
    quant25_pct = pred_dist$quantile(.25) %>% as.numeric(),# %>% exp(),
    quant50_pct = pred_dist$quantile(.5) %>% as.numeric(),# %>% exp(),
    quant75_pct = pred_dist$quantile(.75) %>% as.numeric(),# %>% exp(),
    quant90_pct = pred_dist$quantile(.9) %>% as.numeric(),# %>% exp()
    
    quant10_dlr = quant10_pct*close_last, 
    quant25_dlr = quant25_pct*close_last, 
    quant50_dlr = quant50_pct*close_last, 
    quant75_dlr = quant75_pct*close_last, 
    quant90_dlr = quant90_pct*close_last, 
    
    pred_int_50_pct = quant75_pct-quant25_pct,
    pred_int_80_pct = quant90_pct-quant10_pct,
    
    pred_int_50_dlr = quant75_dlr-quant25_dlr,
    pred_int_80_dlr = quant90_dlr-quant10_dlr,
    
    prob_better_down10 = 1-pred_dist$cdf(.9) %>% as.numeric(),
    prob_better_down05 = 1-pred_dist$cdf(.95) %>% as.numeric(),
    prob_better_down01 = 1-pred_dist$cdf(.99) %>% as.numeric(),
    prob_better_flat = 1-pred_dist$cdf(1) %>% as.numeric(),
    prob_better_up01 = 1-pred_dist$cdf(1.01) %>% as.numeric(),
    prob_better_up05 = 1-pred_dist$cdf(1.05) %>% as.numeric(),
    prob_better_up10 = 1-pred_dist$cdf(1.1) %>% as.numeric()
    
) %>%
    filter(close_last>5 & volume_last>50000) 


saveRDS(out_df, '../pred_df_daily.RDS')
