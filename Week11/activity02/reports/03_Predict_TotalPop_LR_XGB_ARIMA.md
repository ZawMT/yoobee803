# Forecast Report — Total Population

## Approach

| Item | Detail |
| --- | --- |
| Target demographic | Total population |
| Training years | 2014, 2016 |
| Evaluation year | 2018 (held-out test) |
| Forecast year | 2020 |
| Models | LR, XGB, ANN, LSTM, ARIMA |
| Metric | Mean Absolute Error (MAE) |

> **Note on LSTM / ANN:** With only 3 data points per series, neural networks
> have minimal training data. Results should be interpreted cautiously.
> ARIMA uses AR(1) — the simplest viable model for 2–3 observations.

---

## 1. Model Evaluation on 2018 (Predicted vs Actual)

| Measure | Category | Actual 2018 | LR Pred (MAE) | XGB Pred (MAE) | ANN Pred (MAE) | LSTM Pred (MAE) | ARIMA Pred (MAE) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ability to express identity | Easy | 33.5 | 38.1 (4.6) | 35.8991 (2.3991) | 37.8137 (4.3137) | 36.0777 (2.5777) | 33.7 (0.2) |
| Ability to express identity | Hard/very hard | 1.9 | 2.1 (0.2) | 1.9991 (0.0991) | 2.0411 (0.1411) | 1.9463 (0.0463) | 1.9 (0.0) |
| Ability to express identity | Sometimes easy, sometimes hard | 14.3 | 10.4 (3.9) | 11.1009 (3.1991) | 10.6975 (3.6025) | 11.0315 (3.2685) | 11.8 (2.5) |
| Ability to express identity | Very easy | 50.3 | 49.4 (0.9) | 51.0009 (0.7009) | 49.8302 (0.4698) | 50.7252 (0.4252) | 52.6 (2.3) |
| Adequacy of income to meet everyday needs | Enough money | 44.4 | 46.8 (2.4) | 46.1991 (1.7991) | 46.8067 (2.4067) | 45.5209 (1.1209) | 45.6 (1.2) |
| Adequacy of income to meet everyday needs | More than enough money | 18.4 | 19.2 (0.8) | 18.1991 (0.2009) | 18.9472 (0.5472) | 17.8126 (0.5874) | 17.2 (1.2) |
| Adequacy of income to meet everyday needs | Not enough money | 10.0 | 10.2 (0.2) | 11.2009 (1.2009) | 10.2546 (0.2546) | 10.6959 (0.6959) | 12.2 (2.2) |
| Adequacy of income to meet everyday needs | Only just enough money | 27.1 | 23.8 (3.3) | 24.4009 (2.6991) | 24.1279 (2.9721) | 24.347 (2.753) | 25.0 (2.1) |
| Delayed replacing/repairing broken/damaged appliances to keep costs down | A little/a lot | 36.5 | 25.5 (11.0) | 28.001 (8.499) | 25.5535 (10.9465) | 28.2783 (8.2217) | 30.5 (6.0) |
| Done without/cut back on trips to the shops/other local places to keep costs down | A little/a lot | 46.0 | 34.5 (11.5) | 38.501 (7.499) | 38.0545 (7.9455) | 38.4836 (7.5164) | 42.5001 (3.4999) |
| Experience of discrimination in last 12 months | Experienced discrimination | 17.4 | 16.9 (0.5) | 17.0009 (0.3991) | 16.9496 (0.4504) | 16.9928 (0.4072) | 17.1 (0.3) |
| Feeling of safety at home alone at night | Very safe/safe | 86.7 | 84.2 (2.5) | 85.301 (1.399) | 84.9265 (1.7735) | 85.3498 (1.3502) | 86.4 (0.3) |
| Feeling of safety making online transactions | Very safe/safe | 71.8 | 77.6 (5.8) | 74.5991 (2.7991) | 77.4995 (5.6995) | 73.9961 (2.1961) | 71.6 (0.2) |
| Feeling of safety using/waiting for public transport at night | Very safe/safe | 52.9 | 54.4 (1.5) | 52.299 (0.601) | 54.3046 (1.4046) | 51.7385 (1.1615) | 50.2 (2.7) |
| Feeling of safety walking alone in neighbourhood after dark | Very safe/safe | 61.9 | 60.5 (1.4) | 60.7009 (1.1991) | 60.5701 (1.3299) | 60.7139 (1.1861) | 60.9 (1.0) |
| Felt lonely in last four weeks | A little of the time | 22.4 | 23.6 (1.2) | 22.8991 (0.4991) | 23.2683 (0.8683) | 22.5435 (0.1435) | 22.2 (0.2) |
| Felt lonely in last four weeks | Most/all of the time | 3.5 | 9.0 (5.5) | 6.2991 (2.7991) | 4.5291 (1.0291) | 6.6498 (3.1498) | 3.6 (0.1) |
| Felt lonely in last four weeks | None of the time | 61.0 | 56.5 (4.5) | 60.2009 (0.7991) | 59.3904 (1.6096) | 59.8318 (1.1682) | 63.9 (2.9) |
| Felt lonely in last four weeks | Some of the time | 13.1 | 10.9 (2.2) | 10.599 (2.501) | 10.5156 (2.5844) | 10.5663 (2.5337) | 10.3 (2.8) |
| Financial limitation buying a $300 item | A little limited | 23.4 | 20.7 (2.7) | 22.501 (0.899) | 21.3076 (2.0924) | 22.5221 (0.8779) | 24.3 (0.9) |
| Financial limitation buying a $300 item | Couldn't buy it | 24.1 | 28.1 (4.0) | 25.7991 (1.6991) | 26.3919 (2.2919) | 24.1605 (0.0605) | 23.5 (0.6) |
| Financial limitation buying a $300 item | Not at all limited | 21.4 | 22.9 (1.5) | 23.1009 (1.7009) | 23.0997 (1.6997) | 22.6764 (1.2764) | 23.3 (1.9) |
| Financial limitation buying a $300 item | Quite limited | 15.4 | 14.0 (1.4) | 14.301 (1.099) | 14.0819 (1.3181) | 14.2907 (1.1093) | 14.6 (0.8) |
| Financial limitation buying a $300 item | Very limited | 15.8 | 14.5 (1.3) | 14.3991 (1.4009) | 14.5022 (1.2978) | 14.3941 (1.4059) | 14.3 (1.5) |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | A little limited | 40.9 | 40.4 (0.5) | 39.899 (1.001) | 40.2613 (0.6387) | 39.7619 (1.1381) | 39.4 (1.5) |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | Not at all limited | 24.4 | 27.8 (3.4) | 27.3991 (2.9991) | 28.2116 (3.8116) | 26.0354 (1.6354) | 27.0 (2.6) |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | Quite limited | 21.3 | 18.9 (2.4) | 19.5009 (1.7991) | 19.3502 (1.9498) | 19.5498 (1.7502) | 20.1 (1.2) |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | Very limited | 13.4 | 12.9 (0.5) | 13.201 (0.199) | 12.9 (0.5) | 13.1891 (0.2109) | 13.5 (0.1) |
| Gone without fresh fruit/vegetables to keep costs down | A little/a lot | 23.3 | 16.7 (6.6) | 18.201 (5.099) | 18.5664 (4.7336) | 18.2513 (5.0487) | 19.7 (3.6) |
| House or flat colder than would like in winter | No | 47.4 | 50.9 (3.5) | 51.5009 (4.1009) | 50.9812 (3.5812) | 50.4966 (3.0966) | 52.1 (4.7) |
| House or flat colder than would like in winter | Yes always or often | 22.1 | 20.8 (1.3) | 21.0009 (1.0991) | 20.8135 (1.2865) | 21.0193 (1.0807) | 21.2 (0.9) |
| House or flat colder than would like in winter | Yes-sometimes | 30.4 | 28.2 (2.2) | 27.4991 (2.9009) | 28.1127 (2.2873) | 27.4463 (2.9537) | 26.8 (3.6) |
| Life worthwhile | 0 to 6 | 14.0 | 12.3 (1.7) | 12.601 (1.399) | 12.607 (1.393) | 12.6507 (1.3493) | 12.9 (1.1) |
| Life worthwhile | 10 | 23.3 | 23.2 (0.1) | 24.301 (1.001) | 23.5437 (0.2437) | 24.0108 (0.7108) | 25.4 (2.1) |
| Life worthwhile | 7 | 16.7 | 15.3 (1.4) | 15.9009 (0.7991) | 15.531 (1.169) | 15.9075 (0.7925) | 16.5 (0.2) |
| Life worthwhile | 8 | 28.2 | 30.5 (2.3) | 29.5991 (1.3991) | 30.4146 (2.2146) | 29.634 (1.434) | 28.7 (0.5) |
| Life worthwhile | 9 | 17.7 | 18.5 (0.8) | 17.4991 (0.2009) | 18.3236 (0.6236) | 17.0831 (0.6169) | 16.5 (1.2) |
| Life worthwhile | Mean rating | 8.1 | 8.1 (0.0) | 8.1 (0.0) | 8.0787 (0.0213) | 8.1 (0.0) | 8.1 (0.0) |
| Neighbourhood problems in the last 12 months | Burglary/assault | 20.4 | 26.0 (5.6) | 22.499 (2.099) | 25.6045 (5.2045) | 20.8305 (0.4305) | 19.0 (1.4) |
| Neighbourhood problems in the last 12 months | Dangerous driving | 28.6 | 29.5 (0.9) | 27.2991 (1.3009) | 29.3894 (0.7894) | 27.0138 (1.5862) | 25.1 (3.5) |
| Neighbourhood problems in the last 12 months | Drug/alcohol abuse | 18.5 | 18.8 (0.3) | 17.999 (0.501) | 18.6466 (0.1466) | 17.8132 (0.6868) | 17.2 (1.3) |
| Neighbourhood problems in the last 12 months | No neighbourhood problems | 42.4 | 43.2 (0.8) | 45.2009 (2.8009) | 44.1753 (1.7753) | 44.4546 (2.0546) | 47.2 (4.8) |
| Neighbourhood problems in the last 12 months | Noise/vandalism | 35.4 | 32.4 (3.0) | 32.099 (3.301) | 31.9661 (3.4339) | 32.1038 (3.2962) | 31.8 (3.6) |
| Not paid bills on time due to money shortage | Once/more than once | 12.4 | 9.3 (3.1) | 10.9009 (1.4991) | 9.2775 (3.1225) | 10.723 (1.677) | 12.5 (0.1) |
| Overall life satisfaction | 0 to 6 | 18.9 | 17.2 (1.7) | 17.3009 (1.5991) | 17.2819 (1.6181) | 17.3043 (1.5957) | 17.4 (1.5) |
| Overall life satisfaction | 10 | 17.0 | 18.2 (1.2) | 17.9991 (0.9991) | 18.3285 (1.3285) | 17.6415 (0.6415) | 17.8 (0.8) |
| Overall life satisfaction | 7 | 19.0 | 16.7 (2.3) | 18.001 (0.999) | 16.6998 (2.3002) | 18.0365 (0.9635) | 19.3 (0.3) |
| Overall life satisfaction | 8 | 30.3 | 32.8 (2.5) | 30.8991 (0.5991) | 32.3186 (2.0186) | 29.9496 (0.3504) | 29.0 (1.3) |
| Overall life satisfaction | 9 | 14.9 | 15.1 (0.2) | 15.8009 (0.9009) | 15.6496 (0.7496) | 15.5761 (0.6761) | 16.5 (1.6) |
| Overall life satisfaction | Mean rating | 7.7 | 7.8 (0.1) | 7.8 (0.1) | 7.7982 (0.0982) | 7.7579 (0.0579) | 7.8 (0.1) |
| Postponed/put off visits to the doctor to keep costs down | A little/a lot | 25.0 | 19.3 (5.7) | 21.401 (3.599) | 20.3311 (4.6689) | 21.5 (3.5) | 23.5 (1.5) |
| Put up with feeling cold | A little/a lot | 28.1 | 17.1 (11.0) | 20.2009 (7.8991) | 17.0914 (11.0086) | 20.1945 (7.9055) | 23.3001 (4.7999) |
| Satisfaction with job in the last 4 weeks | Dissatisfied/very dissatisfied | 10.6 | 6.5 (4.1) | 6.5 (4.1) | 6.4378 (4.1622) | 6.5 (4.1) | 6.5 (4.1) |
| Satisfaction with job in the last 4 weeks | No feeling either way | 12.7 | 9.8 (2.9) | 9.6991 (3.0009) | 9.6789 (3.0211) | 9.6965 (3.0035) | 9.6 (3.1) |
| Satisfaction with job in the last 4 weeks | Satisfied | 49.8 | 50.9 (1.1) | 49.599 (0.201) | 50.1353 (0.3353) | 49.0726 (0.7274) | 48.3 (1.5) |
| Satisfaction with job in the last 4 weeks | Very satisfied | 27.0 | 32.8 (5.8) | 34.2009 (7.2009) | 33.7018 (6.7018) | 32.7956 (5.7956) | 35.6 (8.6) |
| Self-rated general health status | Excellent | 16.5 | 16.6 (0.1) | 19.101 (2.601) | 17.957 (1.457) | 18.3378 (1.8378) | 21.6 (5.1) |
| Self-rated general health status | Fair/poor | 14.7 | 15.6 (0.9) | 14.5991 (0.1009) | 15.1176 (0.4176) | 13.9607 (0.7393) | 13.6 (1.1) |
| Self-rated general health status | Good | 30.0 | 29.5 (0.5) | 27.5991 (2.4009) | 29.1481 (0.8519) | 27.6432 (2.3568) | 25.7 (4.3) |
| Self-rated general health status | Very good | 38.8 | 38.3 (0.5) | 38.7009 (0.0991) | 38.3714 (0.4286) | 38.6722 (0.1278) | 39.1 (0.3) |
| Spent less on hobbies or other special interests than liked to keep costs down | A little/a lot | 59.7 | 45.7 (14.0) | 49.4009 (10.2991) | 48.4447 (11.2553) | 49.6393 (10.0607) | 53.1 (6.6) |
| Trust held for courts | 0 to 4 | 12.5 | 10.6 (1.9) | 12.401 (0.099) | 11.72 (0.78) | 12.4843 (0.0157) | 14.2 (1.7) |
| Trust held for courts | 5 to 6 | 23.4 | 25.2 (1.8) | 24.5991 (1.1991) | 25.0385 (1.6385) | 24.3224 (0.9224) | 24.0 (0.6) |
| Trust held for courts | 7 to 8 | 40.2 | 42.4 (2.2) | 41.4991 (1.2991) | 42.2789 (2.0789) | 41.3344 (1.1344) | 40.6 (0.4) |
| Trust held for courts | 9 to 10 | 23.9 | 21.8 (2.1) | 21.499 (2.401) | 21.3882 (2.5118) | 21.5499 (2.3501) | 21.2 (2.7) |
| Trust held for courts | Mean rating | 6.9 | 6.9 (0.0) | 6.7991 (0.1009) | 6.8219 (0.0781) | 6.7914 (0.1086) | 6.7 (0.2) |
| Trust held for education system | 0 to 4 | 9.2 | 8.3 (0.9) | 9.101 (0.099) | 8.5986 (0.6014) | 9.0758 (0.1242) | 9.9 (0.7) |
| Trust held for education system | 5 to 6 | 24.8 | 23.8 (1.0) | 23.5991 (1.2009) | 23.4973 (1.3027) | 23.5861 (1.2139) | 23.4 (1.4) |
| Trust held for education system | 7 to 8 | 45.8 | 46.3 (0.5) | 46.3 (0.5) | 46.4331 (0.6331) | 46.109 (0.309) | 46.3 (0.5) |
| Trust held for education system | 9 to 10 | 20.2 | 21.6 (1.4) | 20.9991 (0.7991) | 21.5531 (1.3531) | 20.9632 (0.7632) | 20.4 (0.2) |
| Trust held for education system | Mean rating | 7.0 | 7.0 (0.0) | 7.0 (0.0) | 6.9238 (0.0762) | 7.0 (0.0) | 7.0 (0.0) |
| Trust held for health system | 0 to 4 | 12.7 | 11.7 (1.0) | 11.4991 (1.2009) | 11.5589 (1.1411) | 11.5074 (1.1926) | 11.3 (1.4) |
| Trust held for health system | 5 to 6 | 22.4 | 21.9 (0.5) | 20.9991 (1.4009) | 21.3184 (1.0816) | 20.9237 (1.4763) | 20.1 (2.3) |
| Trust held for health system | 7 to 8 | 42.0 | 42.2 (0.2) | 43.301 (1.301) | 42.7763 (0.7763) | 42.9786 (0.9786) | 44.4 (2.4) |
| Trust held for health system | 9 to 10 | 22.9 | 24.0 (1.1) | 24.1009 (1.2009) | 24.3441 (1.4441) | 23.4221 (0.5221) | 24.2 (1.3) |
| Trust held for health system | Mean rating | 6.9 | 7.0 (0.1) | 7.0 (0.1) | 7.0373 (0.1373) | 6.9786 (0.0786) | 7.0 (0.1) |
| Trust held for media | 0 to 4 | 39.2 | 41.6 (2.4) | 40.099 (0.899) | 41.4722 (2.2722) | 39.5583 (0.3583) | 38.6 (0.6) |
| Trust held for media | 5 to 6 | 35.4 | 35.8 (0.4) | 36.4009 (1.0009) | 35.996 (0.596) | 36.1478 (0.7478) | 37.0 (1.6) |
| Trust held for media | 7 to 8 | 20.6 | 18.5 (2.1) | 19.1009 (1.4991) | 18.5149 (2.0851) | 19.1175 (1.4825) | 19.7 (0.9) |
| Trust held for media | 9 to 10 | 4.8 | 4.4 (0.4) | 4.5009 (0.2991) | 4.4867 (0.3133) | 4.5038 (0.2962) | 4.6 (0.2) |
| Trust held for media | Mean rating | 4.9 | 4.7 (0.2) | 4.8009 (0.0991) | 4.744 (0.156) | 4.8037 (0.0963) | 4.9 (0.0) |
| Trust held for parliament | 0 to 4 | 24.9 | 26.8 (1.9) | 29.1009 (4.2009) | 26.7666 (1.8666) | 27.8845 (2.9845) | 31.4 (6.5) |
| Trust held for parliament | 5 to 6 | 33.8 | 33.3 (0.5) | 32.999 (0.801) | 33.2903 (0.5097) | 32.9277 (0.8723) | 32.7 (1.1) |
| Trust held for parliament | 7 to 8 | 31.7 | 30.4 (1.3) | 29.4991 (2.2009) | 30.3437 (1.3563) | 29.4251 (2.2749) | 28.6 (3.1) |
| Trust held for parliament | 9 to 10 | 9.6 | 9.6 (0.0) | 8.3991 (1.2009) | 9.3685 (0.2315) | 8.3112 (1.2888) | 7.2 (2.4) |
| Trust held for parliament | Mean rating | 5.7 | 5.5 (0.2) | 5.3991 (0.3009) | 5.493 (0.207) | 5.3898 (0.3102) | 5.3 (0.4) |
| Trust held for people in New Zealand | 0 to 4 | 9.7 | 8.5 (1.2) | 8.6009 (1.0991) | 8.4643 (1.2357) | 8.6078 (1.0922) | 8.7 (1.0) |
| Trust held for people in New Zealand | 5 to 6 | 24.4 | 24.7 (0.3) | 23.7991 (0.6009) | 24.1722 (0.2278) | 23.4821 (0.9179) | 22.9 (1.5) |
| Trust held for people in New Zealand | 7 to 8 | 50.8 | 50.8 (0.0) | 52.2009 (1.4009) | 51.0632 (0.2632) | 51.8159 (1.0159) | 53.6 (2.8) |
| Trust held for people in New Zealand | 9 to 10 | 15.1 | 16.0 (0.9) | 15.3991 (0.2991) | 15.9908 (0.8908) | 15.1205 (0.0205) | 14.8 (0.3) |
| Trust held for people in New Zealand | Mean rating | 6.8 | 6.9 (0.1) | 6.9 (0.1) | 6.9147 (0.1147) | 6.8628 (0.0628) | 6.9 (0.1) |
| Trust held for police | 0 to 4 | 6.5 | 6.6 (0.1) | 7.0009 (0.5009) | 6.6582 (0.1582) | 6.8359 (0.3359) | 7.4 (0.9) |
| Trust held for police | 5 to 6 | 12.2 | 13.4 (1.2) | 13.4 (1.2) | 13.5981 (1.3981) | 12.9762 (0.7762) | 13.4 (1.2) |
| Trust held for police | 7 to 8 | 36.1 | 36.2 (0.1) | 36.501 (0.401) | 36.2162 (0.1162) | 36.3739 (0.2739) | 36.8 (0.7) |
| Trust held for police | 9 to 10 | 45.2 | 43.8 (1.4) | 43.0991 (2.1009) | 43.7368 (1.4632) | 43.0994 (2.1006) | 42.4 (2.8) |
| Trust held for police | Mean rating | 7.9 | 7.9 (0.0) | 7.7991 (0.1009) | 7.8463 (0.0537) | 7.777 (0.123) | 7.7 (0.2) |

---

## 2. 2020 Forecast by Model

| Measure | Category | LR 2020 | XGB 2020 | ANN 2020 | LSTM 2020 | ARIMA 2020 |
| --- | --- | --- | --- | --- | --- | --- |
| Ability to express identity | Easy | 34.1667 | 33.5009 | 32.4584 | 34.7578 | 35.9931 |
| Ability to express identity | Hard/very hard | 1.9333 | 1.9009 | 1.8648 | 1.9498 | 2.0 |
| Ability to express identity | Sometimes easy, sometimes hard | 14.9 | 14.2989 | 12.107 | 11.0403 | 10.9548 |
| Ability to express identity | Very easy | 49.0 | 50.301 | 49.6317 | 50.5245 | 51.3816 |
| Adequacy of income to meet everyday needs | Enough money | 44.2 | 44.401 | 44.5163 | 44.963 | 46.3435 |
| Adequacy of income to meet everyday needs | More than enough money | 19.1333 | 18.3991 | 18.9927 | 18.2126 | 17.8526 |
| Adequacy of income to meet everyday needs | Not enough money | 8.9333 | 10.001 | 8.7079 | 10.2927 | 11.1396 |
| Adequacy of income to meet everyday needs | Only just enough money | 27.6 | 27.0989 | 29.2225 | 24.5894 | 24.2744 |
| Delayed replacing/repairing broken/damaged appliances to keep costs down | A little/a lot | 37.6667 | 36.4986 | 41.6556 | 27.3389 | 27.4152 |
| Done without/cut back on trips to the shops/other local places to keep costs down | A little/a lot | 45.8333 | 45.9989 | 48.9059 | 38.4535 | 37.6466 |
| Experience of discrimination in last 12 months | Experienced discrimination | 17.4667 | 17.3991 | 17.5305 | 16.934 | 16.9778 |
| Feeling of safety at home alone at night | Very safe/safe | 86.4333 | 86.699 | 86.7067 | 85.261 | 85.1801 |
| Feeling of safety making online transactions | Very safe/safe | 72.8667 | 71.8008 | 71.0063 | 73.1255 | 74.4951 |
| Feeling of safety using/waiting for public transport at night | Very safe/safe | 54.5 | 52.8991 | 54.4022 | 52.0936 | 51.6478 |
| Feeling of safety walking alone in neighbourhood after dark | Very safe/safe | 62.1667 | 61.8991 | 62.3629 | 60.9547 | 60.6664 |
| Felt lonely in last four weeks | A little of the time | 22.7 | 22.4005 | 22.5948 | 22.5161 | 22.782 |
| Felt lonely in last four weeks | Most/all of the time | 4.3667 | 3.501 | 1.523 | 4.9034 | 6.3486 |
| Felt lonely in last four weeks | None of the time | 58.8 | 60.9995 | 60.7115 | 60.8323 | 61.7219 |
| Felt lonely in last four weeks | Some of the time | 14.1333 | 13.0989 | 11.4085 | 12.9406 | 10.7595 |
| Financial limitation buying a $300 item | A little limited | 22.5 | 23.3996 | 22.5804 | 22.894 | 23.0445 |
| Financial limitation buying a $300 item | Couldn't buy it | 25.0667 | 24.1004 | 22.6406 | 24.3714 | 25.4493 |
| Financial limitation buying a $300 item | Not at all limited | 20.7 | 21.401 | 20.8235 | 21.8914 | 22.9926 |
| Financial limitation buying a $300 item | Quite limited | 15.5667 | 15.399 | 16.1076 | 14.3077 | 14.2313 |
| Financial limitation buying a $300 item | Very limited | 16.3333 | 15.7989 | 16.8802 | 15.8459 | 14.4705 |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | A little limited | 41.5667 | 40.899 | 41.8886 | 40.7365 | 40.0059 |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | Not at all limited | 23.6667 | 24.4011 | 21.8243 | 25.3002 | 27.4497 |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | Quite limited | 21.5 | 21.299 | 20.1266 | 19.3344 | 19.3565 |
| Financial limitation when buying/thinking of buying clothes/shoes for yourself | Very limited | 13.2667 | 13.3996 | 13.3596 | 13.2525 | 13.2598 |
| Gone without fresh fruit/vegetables to keep costs down | A little/a lot | 24.0 | 23.2989 | 24.4901 | 17.9283 | 17.8491 |
| House or flat colder than would like in winter | No | 45.6333 | 47.4011 | 43.7656 | 48.1229 | 51.2112 |
| House or flat colder than would like in winter | Yes always or often | 22.3333 | 22.099 | 22.5466 | 21.0151 | 20.9636 |
| House or flat colder than would like in winter | Yes-sometimes | 31.8333 | 30.3989 | 31.2243 | 29.7661 | 27.7616 |
| Life worthwhile | 0 to 6 | 14.2667 | 13.9989 | 14.51 | 12.5248 | 12.5385 |
| Life worthwhile | 10 | 22.2333 | 23.301 | 22.7807 | 22.9932 | 24.3346 |
| Life worthwhile | 7 | 16.5667 | 16.6991 | 16.7182 | 15.9174 | 15.8241 |
| Life worthwhile | 8 | 28.3333 | 28.2009 | 27.0924 | 28.6449 | 29.7571 |
| Life worthwhile | 9 | 18.4333 | 17.6991 | 18.0837 | 17.3785 | 17.1526 |
| Life worthwhile | Mean rating | 8.1 | 8.1 | 8.0998 | 8.1 | 8.1 |
| Neighbourhood problems in the last 12 months | Burglary/assault | 22.0333 | 20.4005 | 20.0001 | 20.6233 | 21.6519 |
| Neighbourhood problems in the last 12 months | Dangerous driving | 30.5 | 28.599 | 30.4638 | 28.4551 | 26.9372 |
| Neighbourhood problems in the last 12 months | Drug/alcohol abuse | 19.2 | 18.499 | 18.965 | 18.0811 | 17.8807 |
| Neighbourhood problems in the last 12 months | No neighbourhood problems | 40.1333 | 42.401 | 39.5958 | 43.2092 | 44.9808 |
| Neighbourhood problems in the last 12 months | Noise/vandalism | 36.7 | 35.3989 | 37.232 | 34.3322 | 32.2845 |
| Not paid bills on time due to money shortage | Once/more than once | 11.8333 | 12.3992 | 11.5108 | 11.098 | 10.9523 |
| Overall life satisfaction | 0 to 6 | 19.3667 | 18.8989 | 20.2294 | 16.9729 | 17.3123 |
| Overall life satisfaction | 10 | 16.8 | 17.0011 | 16.6184 | 17.2711 | 18.0392 |
| Overall life satisfaction | 7 | 18.4667 | 18.9994 | 18.8675 | 18.183 | 18.1731 |
| Overall life satisfaction | 8 | 31.3667 | 30.3005 | 30.9923 | 30.1377 | 30.1654 |
| Overall life satisfaction | 9 | 14.1333 | 14.901 | 14.3415 | 14.8728 | 15.7421 |
| Overall life satisfaction | Mean rating | 7.6667 | 7.701 | 7.6354 | 7.7274 | 7.7971 |
| Postponed/put off visits to the doctor to keep costs down | A little/a lot | 24.8 | 24.9989 | 24.8022 | 21.598 | 20.9857 |
| Put up with feeling cold | A little/a lot | 28.6667 | 28.0987 | 29.6915 | 20.4119 | 19.4579 |
| Satisfaction with job in the last 4 weeks | Dissatisfied/very dissatisfied | 11.9667 | 10.5988 | 12.0774 | 8.8049 | 6.6167 |
| Satisfaction with job in the last 4 weeks | No feeling either way | 13.7667 | 12.6988 | 15.5703 | 12.5057 | 9.8169 |
| Satisfaction with job in the last 4 weeks | Satisfied | 50.7333 | 49.7991 | 50.6141 | 49.5913 | 49.1254 |
| Satisfaction with job in the last 4 weeks | Very satisfied | 23.6667 | 27.0016 | 24.4508 | 28.7684 | 33.6147 |
| Self-rated general health status | Excellent | 13.9667 | 16.501 | 18.4066 | 17.0496 | 19.0673 |
| Self-rated general health status | Fair/poor | 15.4 | 14.6991 | 14.9481 | 14.2811 | 14.2167 |
| Self-rated general health status | Good | 32.0667 | 29.9989 | 27.4103 | 29.0923 | 27.7464 |
| Self-rated general health status | Very good | 38.5667 | 38.7994 | 38.7619 | 38.7737 | 38.8619 |
| Spent less on hobbies or other special interests than liked to keep costs down | A little/a lot | 60.6667 | 59.6983 | 62.5995 | 47.6681 | 48.5112 |
| Trust held for courts | 0 to 4 | 11.3333 | 12.4994 | 11.3333 | 12.4942 | 13.1491 |
| Trust held for courts | 5 to 6 | 23.4 | 23.401 | 22.9032 | 23.823 | 24.7334 |
| Trust held for courts | 7 to 8 | 40.3667 | 40.201 | 38.9698 | 40.8116 | 41.638 |
| Trust held for courts | 9 to 10 | 24.9 | 23.899 | 25.6206 | 23.831 | 21.6562 |
| Trust held for courts | Mean rating | 7.0 | 6.899 | 6.9853 | 6.8994 | 6.8 |
| Trust held for education system | 0 to 4 | 8.7 | 9.1995 | 9.4978 | 9.1876 | 9.4355 |
| Trust held for education system | 5 to 6 | 25.3333 | 24.799 | 26.0 | 24.4716 | 23.6903 |
| Trust held for education system | 7 to 8 | 45.6333 | 45.8009 | 45.6006 | 45.9145 | 46.2858 |
| Trust held for education system | 9 to 10 | 20.3333 | 20.2009 | 19.5809 | 20.5676 | 21.0759 |
| Trust held for education system | Mean rating | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
| Trust held for health system | 0 to 4 | 13.2333 | 12.699 | 13.845 | 12.6457 | 11.5903 |
| Trust held for health system | 5 to 6 | 23.4667 | 22.3991 | 23.4135 | 22.5202 | 21.1275 |
| Trust held for health system | 7 to 8 | 40.8333 | 42.0009 | 40.8717 | 42.3102 | 43.2391 |
| Trust held for health system | 9 to 10 | 22.4333 | 22.9011 | 21.8772 | 23.2422 | 24.0354 |
| Trust held for health system | Mean rating | 6.8667 | 6.901 | 6.9118 | 6.9178 | 6.9971 |
| Trust held for media | 0 to 4 | 39.9 | 39.2004 | 39.4467 | 39.295 | 39.7365 |
| Trust held for media | 5 to 6 | 34.6667 | 35.4011 | 34.8678 | 35.4737 | 36.3029 |
| Trust held for media | 7 to 8 | 20.7 | 20.5989 | 21.7618 | 19.3234 | 18.9567 |
| Trust held for media | 9 to 10 | 4.8333 | 4.799 | 4.8548 | 4.462 | 4.4761 |
| Trust held for media | Mean rating | 4.8667 | 4.8991 | 4.8763 | 4.8028 | 4.8 |
| Trust held for parliament | 0 to 4 | 21.9667 | 24.9012 | 20.8669 | 25.594 | 28.6688 |
| Trust held for parliament | 5 to 6 | 34.2667 | 33.7989 | 33.0796 | 33.8032 | 33.0844 |
| Trust held for parliament | 7 to 8 | 33.0333 | 31.699 | 29.8373 | 30.9836 | 29.7352 |
| Trust held for parliament | 9 to 10 | 10.8 | 9.5991 | 10.207 | 9.1112 | 8.4 |
| Trust held for parliament | Mean rating | 5.8667 | 5.699 | 5.8847 | 5.6517 | 5.4307 |
| Trust held for people in New Zealand | 0 to 4 | 10.0 | 9.699 | 9.6077 | 8.6595 | 8.5978 |
| Trust held for people in New Zealand | 5 to 6 | 25.2 | 24.399 | 24.9961 | 24.2785 | 23.6828 |
| Trust held for people in New Zealand | 7 to 8 | 49.4 | 50.8009 | 49.8785 | 51.3095 | 52.2 |
| Trust held for people in New Zealand | 9 to 10 | 15.4 | 15.1005 | 15.0191 | 15.1261 | 15.2185 |
| Trust held for people in New Zealand | Mean rating | 6.7667 | 6.801 | 6.7042 | 6.8248 | 6.8971 |
| Trust held for police | 0 to 4 | 6.0667 | 6.5009 | 6.2358 | 6.7222 | 6.9706 |
| Trust held for police | 5 to 6 | 11.8 | 12.201 | 11.1353 | 12.546 | 13.3659 |
| Trust held for police | 7 to 8 | 35.7667 | 36.101 | 35.6999 | 36.2334 | 36.4717 |
| Trust held for police | 9 to 10 | 46.3667 | 45.199 | 46.425 | 44.3274 | 43.3149 |
| Trust held for police | Mean rating | 8.0 | 7.899 | 7.775 | 7.8818 | 7.8 |

---

## 3. Model Comparison — Mean MAE (lower is better)

| Model | Mean MAE |
| --- | --- |
| LR | 2.0750 |
| XGB | 1.7031 |
| ANN | 1.8911 |
| LSTM | 1.5609 |  **Best**
| ARIMA | 1.7312 |

**Best performing model: LSTM** (mean MAE = 1.5609)

---

## 4. Series Count

| Item | Count |
| --- | --- |
| Total series processed | 96 |
| Models compared | 5 |
