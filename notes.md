# Notes IKT 110

## Robustness
- filter data to remove outliers (sliding window, exp moving avg)
- scale data (min max, gaus (find median, scale all relative to median and to amount of standard deviations it is of))
- running managment -> dashboard to get overview of the systemsinternals and some possible parameter changes

## week 42
- to generalize you needto test on a lot of different settings/data
- key metrics: find metrics that are important for your system
  - for our auction house game: gold per point, #BIDS, #WINS/#BIDS, how much money is everyone getting
- Dashboard/controller
  - show metrics
  - allow parameter changes
  - when comparing own metrics to enviromental metrics exclude your own effects (so if you are bidding a lot and see in the enviroment that there is a lot of bidding thats probably because of you)

# next auction house
- make it robust!!!!
- find key metrics to show and change parameters from dashboard

# handin2
- develop a dashboard for real estate
- find hidden insights in the data!!
- Tools: raw python, numpy, dash, plotly
- learning outcome: write your own SGD and practice feature engineering
- -Milestones:
  - understand dataset
  - create model
  - train model
  - use model to obtain insights
- Must haves:
  - a model 
  - dashboard with key metrics
  - insights from data
  - short pic/summary
- how to handle missing data?
- be able to explain why model says how pirces are determined
- 