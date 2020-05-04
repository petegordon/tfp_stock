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

assets <- get_assets()


shortable <- filter(assets, shortable==TRUE)

use_condaenv('tensorflow2gpu', required = TRUE)

pred_df_filter <- readRDS('pred_df_daily.RDS')

## Long Positions #######################
high_upside <- pred_df_filter %>%
    filter(loc>1,
           skewness>0.09) %>%
    arrange(desc(pred1_01)) %>%
    slice(1:20)

riingo_meta(high_upside$ticker) %>%
    select(ticker, name)


## Short Positions #######################
low_upside <- pred_df_filter %>%
    filter(ticker %in% shortable$symbol,
           abs(quant75_log)<abs(quant25_log),
           abs(quant90_log)<abs(quant10_log),
           skewness<0,
           loc<1) %>%
    arrange(pred1_00) %>%
    slice(1:20)

riingo_meta(low_upside$ticker) %>%
    select(ticker, name)




## Enter Long Orders ##################################
tgt_price <- high_upside$last_close*high_upside$quant50

submit_order_multiple <- function(ticker, last_price, desired_dollar, ...){
    submit_order(ticker = ticker,
                 qty = as.character(round(desired_dollar/last_price)),
                 side = 'buy',
                 type = 'limit',
                 time_in_force = 'day',
                 limit_price = last_price*1.01,
                 extended_hours = TRUE
    )
    
}

ls_inp <- list(ticker = high_upside$ticker,
               last_price = tgt_price,
               desired_dollar = (as.numeric(buying_power)/nrow(high_upside))*.3)

orders <- pmap(ls_inp, submit_order_multiple)



## Enter SHort Orders #######################################

short_order_multiple <- function(ticker, last_price, desired_dollar, ...){
    submit_order(ticker = ticker,
                 qty = as.character(round(desired_dollar/last_price)),
                 side = 'sell',
                 type = 'limit',
                 time_in_force = 'day',
                 limit_price = last_price,
                 extended_hours = TRUE
    )
    
}

short_entry <- pred_df_filter %>%
    inner_join(enframe(shortable, value='ticker'), by='ticker') %>%
    mutate(entry=last_close*quant50)

tgt_short_price <- low_upside$last_close*low_upside$quant50



short_inp <- list(ticker = low_upside$ticker,
                  last_price = tgt_short_price,
                  desired_dollar = (as.numeric(buying_power)/nrow(low_upside))*.6)


orders <- pmap(short_inp, short_order_multiple)

