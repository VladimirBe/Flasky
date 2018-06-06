import os
from flask_bootstrap import Bootstrap
from flask import Flask,render_template, request, session, redirect, url_for,flash , send_from_directory, Markup
from flask_wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, MigrateCommand
from flask_script import Shell
from config import Config
from flask_mail import Mail, Message
import random
from bs4 import BeautifulSoup
import requests
import csv
import json
import pandas as pd
from key import secret_session_key, yelp_api_auth,google_api_key,yelp_new_key
import time
#from models import Result,Role,User
#import dash
#from werkzeug.wsgi import DispatcherMiddleware
#import dash_core_components as dcc
#import dash_html_components as html
#from dash.dependencies import Input, Output
import pickle
import datetime
from sqlalchemy import desc

from numpy import pi
from iexfinance import Stock as Stk
from iexfinance import get_historical_data

import ast
from roulette_multiplier import multiplier


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

mail = Mail(app)
#from app import routes, models

"""
dapp = dash.Dash(__name__,server=app,url_base_pathname='/dplot', csrf_protect=False)

dapp.layout = html.Div([
    # represents the URL bar, doesn't render anything
    dcc.Location(id='url', refresh=False),

    dcc.Link('Navigate to "/"', href='/dplot'),
    html.Br(),
    dcc.Link('Navigate to "/page-2"', href='/page-2'),

    # content will be rendered in this element
    html.Div(id='page-content')
])


@dapp.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    return html.Div([
        html.H3('You are on page {}'.format(pathname))
    ])


dapp.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})"""

#----------------

"""dapp.layout = html.Div(children=[
    html.H1(children='Are you a cat person?'),
    html.Label('Your name: '),
    dcc.Input(id='input-div'),
    html.Div(id='output-div', children=[])
])
@dapp.callback(
        Output(component_id='output-div', component_property='children'),
        [Input(component_id='input-div', component_property='value')]
)
def update_output(input_value):
    if input_value is None or not input_value:
        return ['You have not typed your name yet.']
    if input_value == 'Heisenberg':
        return ['You are a cat person.']
    else:
        return ['You are a dog person.']"""

#dapp2 = dash.Dash(__name__,server=app,url_base_pathname='/stocks', csrf_protect=False)
"""
mail = Mail(app)

app.config['SECRET_KEY'] = 'hard to guess string'
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True

db = SQLAlchemy(app)"""

"""class NameForm(Form):
    name = StringField('What is your name?', validators=[Required()])
    submit = SubmitField('Submit')

class Role(db.Model):
    __tablename__ = 'roles'
    extend_existing=True
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    users = db.relationship('User', backref='role')
    def __repr__(self):
        return '<Role %r>' % self.name

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    #gold=db.Column(db.Integer, default=100000)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    def __repr__(self):
        return '<User %r>' % self.username

class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String())
    result_all = db.Column(db.String())
    result_no_stop_words = db.Column(db.String())

    def __init__(self, url, result_all, result_no_stop_words):
        self.url = url
        self.result_all = result_all
        self.result_no_stop_words = result_no_stop_words

    def __repr__(self):
        return '<id {}>'.format(self.id)
"""
#app.config.from_object(os.environ['APP_SETTINGS'])


'''
app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')'''


def make_shell_context():
    return dict(app=app, db=db, User=User, Role=Role)
    manager.add_command("shell", Shell(make_context=make_shell_context))
    migrate = Migrate(app, db)
    manager.add_command

bootstrap = Bootstrap(app)
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = NameForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.name.data).first()
        if user is None:
            user = User(username=form.name.data)
            db.session.add(user)
            session['known'] = False
        else:
            session['known'] = True
        session['name'] = form.name.data
        form.name.data = ''
        return redirect(url_for('index'))
    return render_template('index.html',
        form=form, name=session.get('name'),
        known=session.get('known', False))

@app.route('/')
def home():
    loggedin_user = get_user()
    user, allplayers, leaders = get_leaderboard(loggedin_user)
    user_agent = request.headers.get('User-Agent')
    ip=request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    jsonip="http://freegeoip.net/json/" + ip
    jsonreq=requests.get(jsonip).json()
    location=jsonreq['country_name']
    return render_template("home.html",user_agent=user_agent,ip=ip,location=location,loggedin_user=loggedin_user,user=user,allplayers=allplayers,leaders=leaders)



"""@app.route('/about/')
def about():
    return render_template("about.html")"""

@app.route('/testp')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

@app.route('/test')
def test():
    return render_template('test.html')

"""@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)"""

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
#######################################################################################################################
@app.route('/guesn/',methods=['get','post'])
def count():
    message = None
    error = None
    session['guess_count'] = 0
    session['number']= random.randint(1,100)
    return render_template('guesn.html', message = message)

@app.route('/guess', methods=['post'])
def guess():
    try:
        val = int(request.form['guess'])
    except ValueError:
        error = "That's not an integer! Enter an integer between 1 and 100."
        return render_template('guesn.html', error = error)
    session['guess_count'] += 1
    if int(request.form['guess']) > session['number']:
        message = 'Too high. Enter a smaller number. Your guess count so far is '
        return render_template('guesn.html', message = message)
    elif int(request.form['guess']) < session['number']:
        message = 'Too low. Enter a larger number. Your guess count so far is '
        return render_template('guesn.html', message = message)
    elif int(request.form['guess']) == session['number']:
        message = 'Correct! Congratulations! Your total guesses was '
        return render_template('guesn.html', message = message)

@app.route('/reset', methods=['post'])
def reset():
    message = None
    error = None
    session['guess_count'] = 0
    session['number'] = random.randint(1,100)
    return render_template('guesn.html', message = message)
##########################################################################
@app.route('/scr', methods=['GET', 'POST'])
@app.route('/scraper/', methods=['GET', 'POST'])
def scraper():
    if request.method == 'GET':
        return render_template('links_main.html')
    else:
        links = []
        site = request.form['myUrl']
        r = requests.get("http://" + site)
        data = r.text
        soup = BeautifulSoup(data, "html.parser")
        for link in soup.find_all('a'):
            if 'href' in link.attrs:
                links.append(link)
                #with open("test.txt",'w') as fp:
                    #fp.write("link")
        return render_template('links.html', site=site, links=links)
##################################Yelp########################################
#https://www.yelp.com/developers/documentation/v3/authentication#where-is-my-client-secret-going
app.secret_key = "SECRET_KEY"

LATITUDE = 37.786882
LONGITUDE = -122.399972
YELP_ACCESS_TOKEN = "yelp_access_token"
EMPTY_RESPONSE = json.dumps('')



@app.route("/yelp")
def yelp():
    return render_template('yelp.html')


@app.route("/business_search")
def business_search():
    term = request.args.get("term", None)
    if term == None:
        print ("No term provided for business search, returning nothing")
        return EMPTY_RESPONSE

    response = requests.get('https://api.yelp.com/v3/businesses/search',
            params=get_search_params(term),
            headers=get_auth_dict())
    if response.status_code == 200:
        print ("Got 200 for business search")
        return json.dumps(response.json())
    else:
        print ("Received non-200 response({}) for business search, returning empty response".format(response.status_code))
        return EMPTY_RESPONSE


@app.route("/autocomplete")
def autocomplete():
    term = request.args.get("term", None)
    if term==None:
        print ("No term provided for autocomplete, returning nothing")
        return EMPTY_RESPONSE
    print ("autocompleting for: ", term)

    response = requests.get('https://api.yelp.com/v3/autocomplete', params=get_autocomplete_params(term), headers=get_auth_dict())
    if response.status_code == 200:
        # We return a list of businesses that autocomplete appended with a list of terms that autocomplete.
        return json.dumps([business['name'] for business in response.json()['businesses']]
        + [term['text'] for term in response.json()['terms']])
    else:
        print ("received non-200 response({}) for autocomplete, returning empty response".format(response.status_code))
        return EMPTY_RESPONSE


"""def get_yelp_access_token():
    # WARNING: Ideally we would also expire the token. An expiry is sent with the token which we ignore.NOT VALID ANYMORE
    if YELP_ACCESS_TOKEN in session:
        print ("access token found in session")
    else:
        print ("access token needs to be retrieved")
        response = requests.post('https://api.yelp.com/oauth2/token', data=yelp_api_auth)
        if response.status_code == 200:
            session[YELP_ACCESS_TOKEN] = response.json()['access_token']
            print ("stored access token in session:", session[YELP_ACCESS_TOKEN])
        else:
            raise RuntimeError("Unable to get token, received status code " + str(response.response))

    return session[YELP_ACCESS_TOKEN]"""


def get_search_params(term, latitude=LATITUDE, longitude=LONGITUDE):
    return {'term': term, 'latitude' : latitude, 'longitude' : longitude}


def get_autocomplete_params(term, latitude=LATITUDE, longitude=LONGITUDE):
    return {'text': term, 'latitude' : latitude, 'longitude' : longitude}


def get_auth_dict():
    return {'Authorization' : "Bearer " + yelp_new_key}

####################################Upload#######################################
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = basedir + '/files/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/uploader')
def upload_file():
   return render_template('upload.html')
@app.route('/uploaded', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            return redirect(url_for('upload_file',
                                    filename=filename))

#################################Google_PLaces##############################################
from flask import jsonify
@app.route("/places", methods=['GET', 'POST'])
def places():

    if request.method == 'GET':
        return render_template('places.html')
    else:
        """place = request.form['City']
        link = "https://www.google.com/maps/embed/v1/search?q=" + place +"AIzaSyCdDtCcQ32VC2bEUoQYGnGLShKz_1N1Cbc"
        return render_template('places.html',link=link)"""
nr_businesses=60
key=google_api_key
location="55.945905, -3.190425"
radius="50000"
keyword="vegan restaurant"
searchnearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
url_place="https://maps.googleapis.com/maps/api/place/details/json"
place_id=[]

def get_places(pagetoken):

    search_payload_nearby = {"key":key, "location":location,"radius":radius,"keyword":keyword,"pagetoken":pagetoken}
    search_req = requests.get(searchnearby_url, params=search_payload_nearby)
    search_json = search_req.json()
    for a in search_json['results']:
        place_id.append(a['place_id'])
    new_pagetoken=search_json['next_page_token']
    if len(place_id)<30:
        time.sleep(5)
        get_places(new_pagetoken)
    return place_id

def get_place_info(place_id):
    search_payload_places={"key":key, "place_id":place_id}
    search_req = requests.get(url_place, params=search_payload_places)
    search_json = search_req.json()
    data={"name":search_json['result']['name'],"phone":search_json['result']['formatted_phone_number'],"rating":search_json['result']['rating'],"website":search_json['result']['website']}
    return data


########################Plotly##########################

import plotly
import plotly.graph_objs as go


@app.route('/plot')
def plot():
    # Read in the Data via Pandas


    # Create the Plotly Data Structure
    graph = dict(
        data=[go.Scatter(
            x=[12,3,5,2,34],
            y=[5,33,21,2]
        )],
        layout=dict(
            title='Bar Plot',
            yaxis=dict(
                title="Count"
            ),
            xaxis=dict(
                title="Fruit"
            )
        )
    )

    # Convert the figures to JSON
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

    # Render the Template
    return render_template('plot.html', graphJSON=graphJSON)


######################Predict##################################

from sklearn.externals import joblib

#joblib.dump(lin_reg, "linear_regression_model.pkl")  "Created pkl localy and uploaded pkl to pew"
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            years_of_experience = float(data["yearsOfExperience"])

            lin_reg = joblib.load("linear_regression_model.pkl")
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(years_of_experience).tolist())


@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            training_set = joblib.load("training_data.pkl")
            training_labels = joblib.load("training_labels.pkl")

            df = pd.read_json(data)

            df_training_set = df.drop(["Salary"], axis=1)
            df_training_labels = df["Salary"]

            df_training_set = pd.concat([training_set, df_training_set])
            df_training_labels = pd.concat([training_labels, df_training_labels])

            new_lin_reg = LinearRegression()
            new_lin_reg.fit(df_training_set, df_training_labels)

            os.remove("linear_regression_model.pkl")
            os.remove("training_data.pkl")
            os.remove("training_labels.pkl")

            joblib.dump(new_lin_reg, "linear_regression_model.pkl")
            joblib.dump(df_training_set, "training_data.pkl")
            joblib.dump(df_training_labels, "training_labels.pkl")

            lin_reg = joblib.load("linear_regression_model.pkl")
        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))

        return jsonify("Retrained model successfully.")


@app.route("/currentDetails", methods=['GET'])
def current_details():
    if request.method == 'GET':
        try:
            lr = joblib.load("linear_regression_model.pkl")
            training_set = joblib.load("training_data.pkl")
            labels = joblib.load("training_labels.pkl")

            return jsonify({"score": lr.score(training_set, labels),
                            "coefficients": lr.coef_.tolist(), "intercepts": lr.intercept_})
        except (ValueError, TypeError) as e:
            return jsonify("Error when getting details - {}".format(e))

######################Predict with Dash##################################
"""import plotly.graph_objs as go

dapp2 = dash.Dash(__name__,server=app,url_base_pathname='/dpredict', csrf_protect=False)

training_data = joblib.load("./training_data.pkl")
training_labels = joblib.load("./training_labels.pkl")

dapp2.layout = html.Div(children=[
    html.H1(children='Simple Linear Regression', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Enter years of experience: '),
        dcc.Input(id='years-of-experience', placeholder='Years of experience', type='text'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),

    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=training_data['YearsExperience'],
                    y=training_labels,
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Years of Experience'},
                yaxis={'title': 'Salary'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )
])


@dapp2.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='years-of-experience', component_property='value')])
def update_years_of_experience_input(years_of_experience):
    if years_of_experience is not None and years_of_experience is not '':
        try:
            salary = model.predict(float(years_of_experience))[0]
            return 'With {} years of experience you should earn a salary of ${:,.2f}'.\
                format(years_of_experience, salary, 2)
        except ValueError:
            return 'Unable to give years of experience'
"""

################################Wordcount######################

import operator
import nltk
from stop_words import stops
from collections import Counter
import re
@app.route('/wordcount', methods=['GET', 'POST'])
def wordcount():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the person has entered
        try:
            url = request.form['url']
            r = requests.get(url)
        except:
            errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
            )
            return render_template('wordcount.html', errors=errors)
        if r:
            # text processing
            raw = BeautifulSoup(r.text, 'html.parser').get_text()
            nltk.data.path.append('./nltk_data/')  # set the path
            tokens = nltk.word_tokenize(raw)
            text = nltk.Text(tokens)
            # remove punctuation, count raw words
            nonPunct = re.compile('.*[A-Za-z].*')
            raw_words = [w for w in text if nonPunct.match(w)]
            raw_word_count = Counter(raw_words)
            # stop words
            no_stop_words = [w for w in raw_words if w.lower() not in stops]
            no_stop_words_count = Counter(no_stop_words)
            # save the results
            results = sorted(
                no_stop_words_count.items(),
                key=operator.itemgetter(1),
                reverse=True
            )

    return render_template('wordcount.html', errors=errors, results=results)

######################### Markets ###############
from models import User, Stock, Portfolio, Trade, Position
from forms import StockSearchForm, LoginForm, RegisterForm, PasswordReminderForm, PasswordResetForm, DeleteAccountForm, TradeForm, FullTradeForm, RulForm
app.secret_key = 'some secret key'
def get_datetime_today():
	now = datetime.datetime.now()
	today = datetime.date(now.year, now.month, now.day)
	return today

# Converts numbers to more readable financial formats
def pretty_numbers(value):
	return '${:,.2f}'.format(value)

def pretty_ints(value):
	return '{:,}'.format(value)

def pretty_percent(value):
	return '{:,.2f}%'.format(value)

def pretty_leaders(leaders):
	for l in leaders:
		l.prettyvalue = pretty_numbers(l.value)
	return leaders
def pretty_large(num):
    i=0
    while abs(num)>=1000:
        i+=1
        num=num/1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][i])

def set_color(change):
	if float(change) < 0.000000:
		return True
	else:
		return False


def get_leaderboard(user):
	allplayers = Portfolio.query.order_by(desc(Portfolio.value)).all()
	leaders = Portfolio.query.order_by(desc(Portfolio.value)).limit(5).all()
	leaders = pretty_leaders(leaders)
	allplayers = pretty_leaders(allplayers)

	if user != None:
		user = User.query.filter_by(name=session['username']).first()
		# finding player's position in leaderboard
		for idx, val in enumerate(allplayers):
			if user.portfolio == val:
				user.rank = idx+1
	else:
		loggedin_user = None
		user = None
	return user, allplayers, leaders

def get_user():
	if 'username' in session:
		loggedin_user = session['username']
		user = session['username']
	else:
		loggedin_user = None
		user = None
	return user

def get_account_details(portfolio, positions):
	value = portfolio.cash
	total_gain_loss = float(0.00)
	total_cost = float(0.00)
	portfolio.daily_gain = 0.000
	for p in positions:
		# stock_lookup_and_write(p.symbol) # unfactoring to use stock.stuff
		stock = set_stock_data(Stk(p.symbol))
		write_stock_to_db(stock)
		p.value = Stock.query.filter_by(symbol=p.symbol).first().price*p.sharecount
		p.prettyvalue = pretty_numbers(p.value)
		p.prettycost = pretty_numbers(p.cost)
		value += p.value
		p.gain_loss = p.value - p.cost
		p.gain_loss_percent = p.gain_loss/p.cost*100
		if p.gain_loss <= 0.0000:
			p.loss = True
		p.prettygain_loss = pretty_numbers(p.gain_loss)
		total_gain_loss = float(p.gain_loss) + total_gain_loss
		total_cost = float(p.cost) + total_cost
		p.prettygain_loss_percent = pretty_percent(p.gain_loss_percent)
		p.daily_gain = float(stock.change)*p.sharecount
		p.prettydaily_gain = pretty_numbers(p.daily_gain)
		if p.daily_gain <= 0.0000:
			p.daily_gain_loss = True
		portfolio.daily_gain += p.daily_gain
	portfolio.prettydaily_gain = pretty_numbers(portfolio.daily_gain)
	if portfolio.daily_gain <= 0.0000:
		portfolio.daily_gain_loss = True
	portfolio.total_cost = total_cost
	portfolio.prettytotal_cost = pretty_numbers(total_cost)
	portfolio.value = value
	portfolio.prettyvalue = pretty_numbers(portfolio.value)
	portfolio.prettycash = pretty_numbers(portfolio.cash)
	portfolio.total_stock_value = portfolio.value - portfolio.cash
	portfolio.prettytotal_stock_value = pretty_numbers(portfolio.total_stock_value)
	portfolio.total_gain_loss = total_gain_loss
	portfolio.prettytotal_gain_loss = pretty_numbers(portfolio.total_gain_loss)

	if portfolio.total_cost != 0.00:
		portfolio.total_gain_loss_percent = portfolio.total_gain_loss/portfolio.total_cost*100
		portfolio.prettytotal_gain_loss_percent = pretty_percent(portfolio.total_gain_loss_percent)
	else:
		portfolio.total_gain_loss_percent = 0
		portfolio.prettytotal_gain_loss_percent = "0%"
	if portfolio.total_gain_loss < 0.00:
		portfolio.loss = True

	db.session.commit() # not necessary?
	return portfolio, positions

# This is to take out punctuation and white spaces from the stock search string.
def clean_stock_search(symbol):
	punctuation = '''!()-[]{ };:'"\,<>./?@#$%^&*_~0123456789'''
	no_punct = ""
	for char in symbol:
		if char not in punctuation:
			no_punct = no_punct + char
	if len(no_punct) == 0:
		no_punct = 'RETRY'
	return no_punct

# bypass?
# @db_if_yahoo_fail
def get_Share(symbol):
	stock = Stk(clean_stock_search(symbol))
	return stock

# Puts various attributes into 'stock' via different Share methods.
def set_stock_data(stock):
	stock.name = stock.get_quote(filter_="companyName")['companyName']#
	stock.symbol = stock.get_quote(filter_="symbol")['symbol'].upper()#
	stock.exchange = stock.get_company(filter_="exchange")['exchange']#
	stock.price = stock.get_price()
	stock.prettyprice = pretty_numbers(stock.price)
	stock.change = stock.get_quote(filter_="change")['change']#
	stock.percent_change = stock.get_quote(filter_="changePercent")['changePercent']#
	#stock.afterhours = stock.data_set['AfterHoursChangeRealtime']
	stock.last_traded = stock.get_quote()['latestTime']#
	stock.prev_close = stock.get_quote(filter_="previousClose")['previousClose']#
	stock.open = stock.get_quote(filter_="open")['open']#
	stock.bid = stock.get_quote()['iexBidPrice']
	stock.ask = stock.get_quote()['iexAskPrice']
	#stock.yr_target = stock.data_set['OneyrTargetPrice']
	stock.volume = stock.get_volume()#
	stock.av_volume = stock.get_quote()['avgTotalVolume']#
	stock.day_low = stock.get_open_close()['low']#
	stock.day_high = stock.get_open_close()['high']#
	stock.day_range = str(stock.day_high)+" - "+str(stock.day_low)
	stock.year_high = stock.get_key_stats()['week52high'] #
	stock.year_low = stock.get_key_stats()['week52low'] #
	stock.year_range = str(stock.year_high)+" - "+str(stock.year_low)
	stock.market_cap = stock.get_key_stats()['marketcap']  #
	stock.prettymarket_cap=pretty_large(stock.market_cap)
	stock.peratio = stock.get_book()['quote']['peRatio'] #
	if stock.peratio != None:
		stock.prettyperatio = pretty_numbers(float(stock.peratio))
	else:
		stock.prettyperatio = None
	stock.div = stock.get_key_stats()['dividendYield']
	# not sure why this is causing problems, commenting for now
	# stock.div = float(stock.div)
	stock.prettyex_div = stock.get_key_stats()['exDividendDate']
	stock.ex_div = None #convert_yhoo_date(stock.get_key_stats()['exDividendDate'])
	stock.prettydiv_pay = stock.get_key_stats()['exDividendDate']
	stock.div_pay = None #convert_yhoo_date(stock.get_key_stats()['exDividendDate'] )
	stock.view_count = 1
	stock.loss = set_color(stock.change)
	return stock

def write_stock_to_db(stock):
	if Stock.query.filter_by(symbol=stock.symbol).first() == None:
		db.session.add(Stock(stock.symbol, stock.name, stock.exchange, stock.price, \
			stock.div, stock.ex_div, stock.div_pay, stock.market_cap, stock.view_count))
		db.session.commit()
	else:
		write_stock = Stock.query.filter_by(symbol=stock.symbol).first()
		write_stock.view_count += 1
		write_stock.price = stock.price
		write_stock.div_yield = stock.div
		write_stock.ex_div = stock.ex_div
		write_stock.div_pay = stock.div_pay
		write_stock.market_cap = stock.market_cap
		db.session.commit()

# Look up a stock based on a 'cleaned' input string
def stock_lookup_and_write(symbol):
	stock = set_stock_data(Stk(symbol))
	write_stock_to_db(stock)
	return stock

# not implemented
def search_company(symbol):
	symbol = "%"+symbol+"%"
	# results = Stock.query.filter(Stock.name.ilike(symbol)).first()
	results = Stock.query.filter(Stock.name.ilike(symbol)).all()

	return results


def convert_yhoo_date(yhoo_date):
	# argument yhoo_date should look like "8/6/2015" or None.
	if yhoo_date != None:
		# split and unpack month, day, year variables
		month, day, year = yhoo_date.split('/')
		# convert from strings to integers, for datetime.date function below
		month = int(month)
		day = int(day)
		year = int(year)
		# create datetime object
		return datetime.date(year, month, day)
	else:
		return None

def trade(stock, share_amount, buy_or_sell, user, portfolio, positions):
	stock = set_stock_data(stock)
	write_stock_to_db(stock)
	# get actual stock in db ##
	stock = Stock.query.filter_by(symbol=stock.symbol).first()
	# price and total_cost should be float
	price = (stock.price) #
	total_cost = float(share_amount*price)
	today = get_datetime_today()

	# 1 or -1 multiplier against share_amount
	if buy_or_sell == 'buy':
		# wants to buy
		bs_mult = 1
		total_cost = total_cost*bs_mult
		# check to see if user has enough cash available
		cash = float(portfolio.cash)

		if cash > total_cost:
			new_cash = cash - total_cost

			# for new positions in a given stock
			if portfolio.positions.filter_by(symbol=stock.symbol).first() == None:
				# create & write the new position
				position = Position(user.portfolio.id, stock.symbol, total_cost, total_cost, share_amount, None)
				db.session.add(position)
				db.session.commit()
				flash(" Opened position in " + stock.name + ".")
				# now create trade (need datetime object)
				trade = Trade(stock.symbol, position.id, user.portfolio.id, total_cost, share_amount, today, stock.div_yield, stock.ex_div, stock.div_pay)
				db.session.add(trade)
				# db.session.commit()
				flash("You bought " + str(share_amount) + " shares of " + stock.name + " at " + pretty_numbers(price) + " per share.")
				# adjusting user.portfolio.cash
				user.portfolio.cash = new_cash
				db.session.commit()
				flash("Cash adjusted: -" + pretty_numbers(total_cost))
			# for already existing positions
			elif user.portfolio.positions.filter_by(symbol=stock.symbol).all() != None:
				position = user.portfolio.positions.filter_by(symbol=stock.symbol).first()
				# found the position, now adjust share count.
				trade = Trade(stock.symbol, position.id, user.portfolio.id, total_cost, share_amount, today, stock.div_yield, stock.ex_div, stock.div_pay)
				db.session.add(trade)
				flash("You bought " + str(share_amount) + " shares of " + stock.name + " at " + pretty_numbers(price) + " per share.")
				user.portfolio.cash = new_cash
				position.cost = float(position.cost) + total_cost
				position.value = float(position.value) + total_cost
				position.sharecount += share_amount
				db.session.commit()
		else:
			deficit = total_cost - cash
			flash("Sorry, that costs "+ pretty_numbers(total_cost) + ", which is " + pretty_numbers(deficit) + " more than you have available. Try buying fewer shares.")
	else:
		# wants to sell
		bs_mult = -1
		total_cost = total_cost*bs_mult
		# check to see if there are enough stocks in the user's position
		position = user.portfolio.positions.filter_by(symbol=stock.symbol).first()
		if position != None:
			if position.sharecount >= share_amount:
				trade = Trade(stock.symbol, position.id, user.portfolio.id, total_cost, -1*share_amount, today, stock.div_yield, stock.ex_div, stock.div_pay)
				db.session.add(trade)
				flash("You sold " + str(share_amount) + " shares of " + stock.name + " at " + pretty_numbers(stock.price) + " per share. Adding " + pretty_numbers(total_cost*-1) + " to your cash balance.")
				# update position
				user.portfolio.cash = float(user.portfolio.cash) - total_cost
				position.cost = float(position.cost) + total_cost
				position.value = float(position.value) + total_cost
				position.sharecount = position.sharecount + share_amount*bs_mult
				# I'll remove this one if I can figure out the bug with Heroku's db.
				db.session.commit()
				# close position if no more shares
				if position.sharecount == 0:
					try:
						db.session.delete(position)
						db.session.commit()
						flash("Your position in " + stock.name + " has been closed.")
					except:
						flash("Your position in " + stock.name + " is now empty. I'm working on a way to remove it from the database.")
			else:
				flash("You only have " + str(position.sharecount) + " shares of " + str(stock.symbol) + ". Try selling fewer shares.")
		else:
			flash("You don't have any shares of " + stock.symbol + " to sell.")

def prepare_stock_graph(symbol, start, end):

	stock = Stk(symbol)
	df = get_historical_data(str(symbol), start=start, end=end, output_format='pandas')

	return df

def build_portfolio_pie(portfolio, positions):
	percent_base = 0.00
	percents = []
	for p in positions:
		p.position_value_percentage = float(p.value)/float(portfolio.value-portfolio.cash)
		percent_base = percent_base + float(p.position_value_percentage)
		percents.append(percent_base)
	# percents.append(float(portfolio.cash)/float(portfolio.value))
	stocknames = [p.symbol for p in positions]
	# stocknames = stocknames.append('Cash')

	starts = [float(p)*2*pi for p in percents[:-1]]
	ends = [float(p)*2*pi for p in percents[1:]]
	# ends.append(starts[:-1])
	color_palette = ['aqua', 'aquamarine', 'cadetblue', 'chartreuse', 'cornflowerblue','darkslateblue', 'darkslategray', 'deepskyblue','dodgerblue','lawngreen', 'lightblue', 'lightcyan', 'lightseagreen', 'lightsteelblue', 'mediumaquamarine','mediumblue','mediumseagreen', 'blue', 'green', 'navy','indigo','purple','cyan','darkblue','darkcyan','darkseagreen', 'darkturquoise', 'forestgreen','mediumturquoise']
	colors = [color_palette[n] for n in range(0,len(percents))]
	graph_values=[{
	    'labels':stocknames,
	    'values':percents,
	    'type':'pie',
	    'insidetextfont':{'color':'#FFFFFF','size':'14',
	        },


	    }
	    ]
	layout={'title':'Portfolio Pie'}
	return graph_values,layout


"""
	p = figure(x_range=(-1.1,1.85), y_range=(-1,1), title='Stock positions', toolbar_location='below', tools='', width=420, plot_height=320)

	for n in range(0,len(positions)):
		p.wedge(x=0, y=0, radius=1, start_angle=starts[n], end_angle=ends[n], color=colors[n], legend=stocknames[n])

	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None
	p.xaxis.major_tick_line_color = None
	p.xaxis.minor_tick_line_color = None
	p.yaxis.major_tick_line_color = None
	p.yaxis.minor_tick_line_color = None
	p.outline_line_color = None

	script, div = components(p)
	return script, div, colors"""

def build_stock_plot(df):
    trace = go.Scatter(x=df.index,
                       y=df.close
                       )
    data = [trace]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

# === decorator and email imports ======
from decorators import *
from emails import send_async_email, send_email, new_user_email, password_reminder_email, password_reset_email
# Importing email functions here since they use the above decorator.

#=== views ====================
@app.errorhandler(404)
def not_found(e):
	flash('Resource not found.')
	user = get_user()
	return render_template('/Markets/404.html', loggedin_user=user)

@app.route('/about')
@login_reminder
def about():
	title = 'About Oligarch'
	user = get_user()
	return render_template('/Markets/about.html', title=title, loggedin_user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
	title = 'Register a new account'
	form = RegisterForm(request.form)

	if request.method == 'POST' and form.validate():
		now = datetime.datetime.now()
		username = form.username.data.lower()
		email = form.email.data
		password = form.password.data
		if User.query.filter_by(name=username).first() == None:
			if User.query.filter_by(email=email).first() == None:
				user = User(username, email, password, now)
				db.session.add(user)
				db.session.commit()

				# create portfolio for the user at the same time
				port = Portfolio(user.id, 1000000, 1000000)
				db.session.add(port)
				db.session.commit()

				session['logged_in'] = True
				session['username'] = user.name
				flash('Thanks for registering!')
				flash('$1,000,000.00 was added to your account.')
				new_user_email(user)
				return redirect(url_for('user'))
			else:
				flash('That email is already registered with a user. Please log in or register another user.')
				return redirect(url_for('register'))
		else:
			flash('That user name already exists.')
			return redirect(url_for('register'))
	elif request.method == 'POST' and not form.validate():
		flash('Try again.')
	elif request.method == 'GET':
		return render_template('/Markets/register.html', title=title, form=form)
	return render_template('/Markets/register.html', title=title, form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
	error = None
	form = LoginForm(request.form)
	title = 'Login'

	if request.method == 'POST' and form.validate():
		user = User.query.filter_by(name=form.username.data).first()
		if user != None:
			userpw = user.password
			if userpw == form.password.data:
				session['logged_in'] = True
				# experiment
				session['username'] = request.form['username']
				flash('You were just logged in.')
				user.last_seen = datetime.datetime.now()
				db.session.commit()
				return redirect(url_for('user'))
			else:
				flash('Incorrect password for that user name, please try again.')
				return redirect(url_for('login'))
		else:
			# Allowing the user to sign in using email.
			user = User.query.filter_by(email=form.username.data).first()
			if user != None:

				userpw = user.password
				if userpw == form.password.data:
					session['logged_in'] = True
					session['username'] = user.name
					flash('You were just logged in.')
					user.last_seen = datetime.datetime.now()
					db.session.commit()
					return redirect(url_for('user'))
			else:
				flash('That user name does not exist in our system. Please try again or sign up for a new account.')
				return redirect(url_for('login'))
		return render_template('/Markets/login.html', form=form, error=error, title=title)
	elif request.method == 'POST' and not form.validate():
		flash('Invalid username or password. Try again or register a new account.')
		return redirect(url_for('login'))
	elif request.method == 'GET':
		return render_template('/Markets/login.html', form=form, error=error, title=title)

@app.route('/logout')
@login_required
def logout():
	session.pop('logged_in', None)
	session.pop('username', None)
	flash('You were just logged out.')
	return redirect(url_for('stocks'))

@app.route('/password_reminder', methods=['GET', 'POST'])
def password_reminder():
	error = None
	form = PasswordReminderForm(request.form)
	title = "Forgot your password?"

	if request.method == 'POST' and form.validate():
		user = User.query.filter_by(name=form.username.data).first()
		if user != None:
			password_reminder_email(user)
			flash("Sent reminder email to "+user.name+"'s email address. Please check your inbox and sign in. Check your spam folder if you don't see our email within a couple of minutes.")
			return redirect(url_for('login'))
		else:
			# Allowing the user to sign in using email.
			user = User.query.filter_by(email=form.username.data).first()
			if user != None:
				password_reminder_email(user)
				flash("Sent reminder email to "+user.email+". Please check your inbox and sign in. Check your spam folder if you don't see our email within a couple of minutes.")
				return redirect(url_for('login'))
			else:
				flash("We couldn't find any user with that username or email address. Please try a different name/address or register a new account.")

	elif request.method == 'POST' and not form.validate():
		flash('Invalid username or password. Try again or register a new account.')
		return redirect(url_for('password_reminder'))

	# elif request.method == 'GET':
	return render_template('/Markets/password_reminder.html', form=form, title=title, error=error)

@app.route('/db_view')
@login_reminder
# @cache.cached(timeout=40)
# unless I figure out a better way, I can't cache user pages. Two concurrent users are able to see the other's page if it's in cache!
def db_view():
	title = "Under the hood"
	user = get_user()
	stocks = Stock.query.all()
	users = User.query.all()
	trades = Trade.query.all()
	portfolios = Portfolio.query.all()
	positions = Position.query.all()
	return render_template("/Markets/db_view.html", title=title, stocks=stocks, users=users, trades=trades, positions=positions, portfolios=portfolios, loggedin_user=user)

@app.route('/about')
def aboutadam():
	return render_template('/Markets/about.html')

@app.route('/tos')
def tos():
	return render_template('/Markets/tos.html')

@app.route('/news')
@login_reminder
def news():
	title = 'Release log'
	user = get_user()
	return render_template('/Markets/news.html', title=title, loggedin_user=user)

@app.route('/leaderboard')
@login_reminder
def leaderboard():
	title = "Leaderboard"
	flash("This page is under development. It will look nicer soon!")
	loggedin_user = get_user()
	user, allplayers, leaders = get_leaderboard(loggedin_user)

	return render_template('/Markets/leaderboard.html', title=title, leaders=allplayers, loggedin_user=loggedin_user)
@app.route('/usertest')
def usertest():
    a = request.args.get('a', "0", type=str)
    flash(a)
    return jsonify(a=a)


@app.route('/user', methods=['GET', 'POST'])
@login_required
def user():
	today = get_datetime_today()
	form = FullTradeForm(request.form)
	loggedin_user = get_user()
	user = User.query.filter_by(name=session['username']).first()
	title = user.name+"'s account summary"
	portfolio = user.portfolio
	positions = portfolio.positions.all()
	for p in positions:
		p.prettysharecount = pretty_ints(p.sharecount)

	if request.method == 'GET':
		# refresh current stock prices and therefore account value
		portfolio, positions = get_account_details(portfolio, positions)


		graph_values, layout = build_portfolio_pie(portfolio, positions)

		return render_template('/Markets/account.html', title=title, user=user, portfolio=portfolio, form=form, loggedin_user=loggedin_user, positions=positions, graph_values=graph_values, layout=layout)
	elif request.method == 'POST' and form.validate():
		stock = get_Share(form.symbol.data)
		# stock = Share(clean_stock_search(form.symbol.data))
		share_amount = form.share_amount.data
		buy_or_sell = form.buy_or_sell.data
		if stock.get_price() == None:
			# If it's POST and valid, but there's no such stock
			flash("Couldn't find stock matching "+form.symbol.data.upper()+". Try another symbol.")
			return redirect(url_for('user'))
		else:
			# if it's POSTed, validated, and there actually is a real stock
			trade(stock, share_amount, buy_or_sell, user, portfolio, positions)
			return redirect(url_for('user'))
	elif request.method == 'POST' and not form.validate():
		flash('Invalid values. Please try again.')
		return redirect(url_for('user'))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
	loggedin_user = get_user()
	user, allplayers, leaders = get_leaderboard(loggedin_user)
	form = PasswordResetForm(request.form)
	deleteform = DeleteAccountForm(request.form)
	title = "{}'s account settings".format(user.name)

	if request.method == 'POST' and form.validate():
		if form.old_password.data == user.password:
			flash("Your password has been reset.")
			user.password = form.new_password.data
			db.session.commit()
			password_reset_email(user)
			return redirect(url_for('user'))
		else:
			flash("Your old password was incorrect. Please try again.")
			return redirect(url_for('settings'))

	elif request.method == 'POST' and not form.validate():
		flash("Something went wrong; please try again.")
		return redirect(url_for('settings'))

	else:
		return render_template('/Markets/settings.html', title=title, loggedin_user=loggedin_user, user=user, form=form, deleteform=deleteform)

@app.route('/delete_account', methods=['GET', 'POST'])
@login_required
def delete_account():
	deleteform = DeleteAccountForm(request.form)
	loggedin_user = get_user()
	user, allplayers, leaders = get_leaderboard(loggedin_user)

	if request.method == 'POST' and deleteform.validate():
		if deleteform.confirm.data.upper() == 'DELETE':
			db.session.delete(user)
			db.session.commit()
			flash("Your account has been deleted.")
			return redirect(url_for('logout'))
		else:
			flash('Type "DELETE" in the field below if you are sure you want to delete your account; this cannot be undone.')
			return redirect(url_for('settings'))
	elif request.method == 'POST' and not deleteform.validate():
		flash('Type "DELETE" in the field below if you are sure you want to delete your account; this cannot be undone.')
		return redirect(url_for('settings'))

@app.route('/stocks', methods=['GET', 'POST'])
@login_reminder
def stocks():
	title = 'Wall Street'
	stock = None
	loggedin_user = get_user()
	user, allplayers, leaders = get_leaderboard(loggedin_user)
	form = StockSearchForm(request.form)
	tradeform = TradeForm(request.form)
	stocks = Stock.query.order_by(desc(Stock.view_count)).limit(10).all()
	if request.method == 'POST':
		if form.validate():
			stock = get_Share(form.stocklookup.data)
			# stock = Share(clean_stock_search(form.stocklookup.data))
			if stock.get_quote(filter_="open")['open'] == None:
			# company lookup goes here
				company_results = search_company(form.stocklookup.data)
				stock = None
				if len(company_results) == 0:
					flash("Couldn't find symbol or company matching "+form.stocklookup.data.upper()+". Try searching for something else.")
				else:
					flash("Didn't find that symbol, but found " + str(len(company_results)) +" matching company names:")

				return render_template('/Markets/stocks.html', stock=stock, form=form, stocks=stocks, leaders=leaders, user=user, loggedin_user=loggedin_user, results=company_results)
			else:
				# There is a stock with this symbol, serve the dynamic page
				stock = set_stock_data(stock)
				# Some stocks appear to not have company names
				if stock.name != None:
					title = stock.symbol+" - "+stock.name
				else:
					title = stock.symbol+" - Unnamed company"
				write_stock_to_db(stock)
				return redirect(url_for('stock', symbol=stock.symbol))
		elif not form.validate():
			flash("Please enter a stock.")
			return redirect(url_for('stocks'))
		return render_template('/Markets/stocks.html', form=form, tradeform=tradeform, stock=stock, leaders=leaders, title=title, user=user, loggedin_user=loggedin_user)
	elif request.method == 'GET':
		for s in stocks:
			s.prettyprice = pretty_numbers(s.price)
			s.prettymarket_cap=pretty_large(int(s.market_cap))
		return render_template('/Markets/stocks.html', form=form, tradeform=tradeform, stock=stock, stocks=stocks, leaders=leaders, title=title, user=user, loggedin_user=loggedin_user)

@app.route('/<symbol>', methods=['GET', 'POST'])
def stock(symbol):
	stock = get_Share(symbol)
	if stock.get_quote(filter_="open")['open'] == None:
		# flash("Couldn't find that stock. Try another symbol.")
		stock = None
		return redirect(url_for('stocks'))
	else:
		# you wrote a function for these two lines, replace here!
		stock = set_stock_data(Stk(symbol))
		write_stock_to_db(stock)
		### ^^
	title = stock.name
	loggedin_user = get_user()
	user, allplayers, leaders = get_leaderboard(loggedin_user)

	form = StockSearchForm(request.form)
	tradeform = TradeForm(request.form)
	stocks = Stock.query.order_by(desc(Stock.view_count)).limit(10).all()

	if user != None:
		portfolio = user.portfolio
		portfolio.prettycash = pretty_numbers(portfolio.cash)
		# This is to show many shares much of that particular stock a user has in his/her position.
		positions = portfolio.positions
		position = portfolio.positions.filter_by(symbol=symbol).first()
		if position:
			position.prettysharecount = pretty_ints(position.sharecount)
	else:
		portfolio = None
		position = None
		positions = None

	if request.method == 'POST' and tradeform.validate():
		share_amount = tradeform.amount.data
		buy_or_sell = tradeform.buy_or_sell.data
		if stock.get_price() == None:
			# If it's POST and valid, but there's no such stock
			flash("Couldn't find stock matching "+symbol.upper()+". Try another symbol.")
			return redirect(url_for('stocks'))
		else:
			# if it's POSTed, validated, and there is a real stock
			trade(stock, share_amount, buy_or_sell, user, portfolio, positions)
			return redirect(url_for('user'))

	elif request.method == 'POST' and not tradeform.validate():
		flash("Invalid share amount; please try again.")
		return redirect(url_for('stocks'))

	if request.method == 'GET':
		start = datetime.date(2017, 2, 9)
		end = datetime.date(2018, 5, 22)
		df = prepare_stock_graph(symbol, start, end)
		graphJSON=build_stock_plot(df)

	return render_template('/Markets/stock.html', form=form, tradeform=tradeform, stock=stock, stocks=stocks, leaders=leaders, title=title, user=user, loggedin_user=loggedin_user, position=position, graphJSON=graphJSON)

@app.route('/roulette',methods=['GET', 'POST'])
@login_required
def roulette():
    loggedin_user = get_user()
    user, allplayers, leaders = get_leaderboard(loggedin_user)
    if user != None:
        portfolio = user.portfolio
        portfolio.prettycash = pretty_numbers(portfolio.cash)
    rulform=RulForm(request.form)
    return render_template("roulette.html",rulform=rulform,user=user,loggedin_user = get_user())

def check_ball(nr):
    win_list=[]
    if nr==0 or nr==-1:
        return win_list.append("ZERO")
    if nr%2==0:
        win_list.append("EVEN")
    else:
        win_list.append("ODD")
    red = [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]
    if nr in red:
        win_list.append("RED")
    else:
        win_list.append("BLACK")
    if nr in range(1,19):
        win_list.append("1 to 18")
    else:
        win_list.append("19 to 36")

    if nr in list(range(1,35,3)):
        win_list.append("1st column")
    elif nr in list(range(2,36,3)):
        win_list.append("2nd column")
    else:
        win_list.append("3rd column")

    if nr in range(1,13):
        win_list.append("1st dozen")
    elif nr in range(13,25):
        win_list.append("2nd dozen")
    else:
        win_list.append("3rd dozen")
    straight="Straight up "+str(nr)
    win_list.append(straight)

    return win_list



@app.route('/spin')
def spin():
    a = request.args.get('a', "0", type=str)
    ball_land_on=random.randint(-1,36)
    win_list=check_ball(ball_land_on)
    data=ast.literal_eval(a)
    d={}
    for i in data:
        if i["bet_on"] in d:
            d[i["bet_on"]]+=int(i["amount"])
        else:
            d[i["bet_on"]]=int(i["amount"])
    amount_won=0
    amount_bet=0
    loggedin_user = get_user()
    user, allplayers, leaders = get_leaderboard(loggedin_user)

    #for key in d:
    #    if key in multiplier:
    #        amount_won=d[key]*multiplier[key]

    for key in d:
        amount_bet+=d[key]
        if key in win_list:
            amount_won+=d[key]*multiplier[key]
    cash= user.portfolio.cash
    new_cash=pretty_numbers(cash-amount_bet+amount_won)
    #user.portfolio.cash=new_cash
    #db.session.commit()

    return jsonify(result=ball_land_on,new_cash=new_cash,amount_won=amount_won,amount_bet=amount_bet)
