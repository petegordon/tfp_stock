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

acct <- get_account()

pred_df_filter <- readRDS('pred_df_daily.RDS')

quotes <- future_map_dfr(pred_df_filter$ticker, riingo_iex_quote, .progress = TRUE)

quotes <- quotes %>%
    filter(as.Date(quoteTimestamp)==Sys.Date()) %>%
    select(ticker, last, bidPrice, askPrice)



intraday <- pred_df_filter %>%
    inner_join(quotes, by='ticker') %>%
    mutate(last_prop=last/last_close) 


get_intraday_cdf <- function(value, loc, scale, skewness, tailweight){
    tfd_cdf(distribution = tfd_sinh_arcsinh(loc = loc,
                                            scale = scale,
                                            skewness = skewness,
                                            tailweight = tailweight
    ),
    value) %>%
        as.numeric()
    
}


cdf_inp <- list(value=intraday$last_prop,
                loc=intraday$loc,
                scale=intraday$sd,
                skewness=intraday$skewness,
                tailweight=intraday$tailweight)


intraday$intra_cdf <- 1-future_pmap_dbl(.l=cdf_inp, .f=get_intraday_cdf, .progress = TRUE)


longs <- intraday %>%
    arrange(-intra_cdf) %>%
    slice(1:10) %>%
    select(ticker, last_close, last_prop, last, bidPrice, askPrice)


long_intraday_multiple <- function(ticker, last_price, desired_dollar, ...){
    submit_order(ticker = ticker,
                 qty = as.character(round(desired_dollar/last_price)),
                 side = 'buy',
                 type = 'limit', ## Figure Out IOC
                 time_in_force = 'day',
                 limit_price = last_price,
                 extended_hours = TRUE
    )
    
}

long_inp <- list(ticker = longs$ticker,
                 last_price = longs$last,
                 desired_dollar = (as.numeric(acct$daytrading_buying_power)/nrow(longs))*.48)

riingo_meta(long_inp$ticker) %>%
    select(ticker, name)




shorts <- intraday %>%
    arrange(intra_cdf) %>%
    slice(1:10) %>%
    select(ticker, last_close, last_prop, last, bidPrice, askPrice)


short_intraday_multiple <- function(ticker, last_price, desired_dollar, ...){
    submit_order(ticker = ticker,
                 qty = as.character(round(desired_dollar/last_price)),
                 side = 'sell',
                 type = 'limit', ## Figure Out IOC
                 time_in_force = 'day',
                 limit_price = last_price*1.01,
                 extended_hours = TRUE
    )
    
}

short_inp <- list(ticker = shorts$ticker,
                  last_price = shorts$last,
                  desired_dollar = (as.numeric(acct$daytrading_buying_power)/nrow(shorts))*.48)

riingo_meta(short_inp$ticker) %>%
    select(ticker, name)


long_intra_orders <- pmap(long_inp, long_intraday_multiple)

short_intra_orders <- pmap(short_inp, short_intraday_multiple)

