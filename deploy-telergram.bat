git init
heroku git:remote -a stock-market-price-prediction
git add .
git commit -am "deployed"
git push heroku master
cmd /k
pause