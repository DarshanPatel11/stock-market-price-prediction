git init
heroku git:remote -a stock-market-price-flask
git add .
git commit -am "deployed"
git push heroku master
cmd /k
pause