from flask_wtf import FlaskForm
from wtforms import TextField, IntegerField, BooleanField, PasswordField, RadioField, validators
from wtforms.validators import Length, EqualTo, Email, NumberRange, Required

class StockSearchForm(FlaskForm):
    stocklookup=TextField('stocklookup',[validators.Length(min=1,max=18)])

class LoginForm(FlaskForm):
	username = TextField('username', [validators.Length(min=2, max=30)])
	password = PasswordField('password', [validators.Length(min=5, max=30)])

class PasswordReminderForm(FlaskForm):
	username = TextField('username', [validators.Length(min=2, max=30)])

class PasswordResetForm(FlaskForm):
	old_password = PasswordField('Old password', validators=[Length(min=6, max=30, message='Password must be between 6 and 30 characters.')])
	new_password = PasswordField('New password', validators=[Length(min=6, max=30, message='Password must be between 6 and 30 characters.')])
	confirm_new_password = PasswordField('Confirm new password', validators=[EqualTo('new_password', message='Passwords must match.')])

class DeleteAccountForm(FlaskForm):
	confirm = TextField('Type "DELETE" to delete account', validators=[Length(min=6, max=6, message='Please type "delete" to confirm; this cannot be undone!')])

class RegisterForm(FlaskForm):
	username = TextField('Username', validators=[Length(min=2, max=25, message='Username must be between 2 and 25 characters.')])
	email =  TextField('Email address', validators=[Length(min=6, max=50), Email(message='Please enter a valid email address.')])
	password = PasswordField('Password', validators=[Length(min=6, max=30, message='Password must be between 6 and 30 characters.')])
	confirm = PasswordField('Confirm password', validators=[EqualTo('password', message='Passwords must match.')])
	accept_tos = BooleanField('I accept the Terms of Service (required)', validators=[Required(message='You must accept the Terms of Service to register an account.')])

class TradeForm(FlaskForm):
	amount = IntegerField('Shares', [validators.NumberRange(min=1, max=9999999999999, message="Invalid share quantity. Please try again.")])
	buy_or_sell = RadioField('Buy or Sell', choices=[('buy','Buy'), ('sell','Sell')])

class FullTradeForm(FlaskForm):
	symbol = TextField('Stock symbol', [validators.Length(min=1, max=10)])
	share_amount = IntegerField('Shares', validators=[NumberRange(min=1, max=999999999999, message="Invalid share quantity. Please try again.")])
	buy_or_sell = RadioField('Buy or Sell', choices=[('buy','Buy'), ('sell','Sell')])

class RulForm(FlaskForm):

	jetons = RadioField('Jetons', choices=[('50','50$'),('100','100$'), ('200','200$'), ('500','500$'), ('1000','1000$')])