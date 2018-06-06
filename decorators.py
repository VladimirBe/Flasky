from functools import wraps
from threading import Thread
from flask_app import render_template, redirect, url_for, request, session, flash, Markup

# decorators

def login_required(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args, **kwargs)
		else:
			flash('You need to log in first.')
			return redirect(url_for('login'))
	return wrap

def login_reminder(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args, **kwargs)
		else:
			message = Markup("<a href='/login'>Sign in</a> or <a href='/register'>register</a> to play.")
			flash(message)
			return f(*args, **kwargs)
	return wrap

# This decorator is to perform asynchronous tasks (such as sending emails)
def async(f):
	def wrapper(*args, **kwargs):
		thr = Thread(target=f, args=args, kwargs=kwargs)
		thr.start()
	return wrapper
