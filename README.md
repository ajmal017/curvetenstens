# CurveTensTens

This project was an experiment of quantitative research on a three-legged bond futures spread trading strategy. It only generates trading signals from backtesting data, without any execution involved. 

Contributed by: Desmond Ho and Mark Ng.
Pls contact desmondhw@gmail.com for any questions.

**Data**:<br>

Data was sampled was from 25/02/2012 to 18/02/2015 inclusive. Approximately 2.6 years of hourly data from CQG Integrated Client.

CurveTensTens_Cleaned.csv is the data file.

**Backtest**:<br>

CurveTensTens.py runs the backtest. Make sure to download CurveTensTens_Cleaned.csv into the same directory as CurveTensTens.py.
Ignore any trend following related codes (line 1301 and 1412).<br>
plotCurveTensTens.py plots the results.

**Report:**<br>
Full report can be found in: curveTensTensPaper.pdf

**Requirements:** <br>

* Python 2.7 <br>

**Python Libraries Required:**<br>

* datetime<br>
* numpy<br>
* pandas<br>
* dateutil<br>
* matplotlib.pyplot<br>
* csv<br>
* math<br>
* time<br>

Python libraries above can be installed via `pip`. Note that we only tested with the versions above, newer versions might not work.
