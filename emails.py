from flask_app import async, app, Config, Message, mail

# emails
@async
def send_async_email(app, msg):
	with app.app_context():
		mail.send(msg)

def send_email(subject, sender, recipients, text_body, html_body):
	msg = Message(subject, sender=sender, recipients=recipients)
	msg.body = text_body
	msg.html = html_body
	send_async_email(app, msg)

def new_user_email(user):
	subject = "Welcome to Wall Street, {}!".format(user.name)
	sender=('Wall Street', Config.MAIL_USERNAME)
	recipients=[user.email]
	text_body = "Welcome to Wall Street. Log in and start trading!"
	html_body = "<h3>Hi %s,</h3><p>Thanks for registering an account with Wall Street. We've added $1,000,000 of play money to your account. <a href='http://vladimirb.pythonanywhere.com/login'>Sign in</a> and start trading!<br><br>Good luck!<br> -Admin</p>"%(user.name)
	send_email(subject, sender, recipients, text_body, html_body)

def password_reset_email(user):
	subject = "Password reset"
	sender=('Wall Street', Config.MAIL_USERNAME)
	recipients=[user.email]
	text_body = "Your Wall Street password has been reset."
	html_body = "<h3>Hi %s,</h3><p>Your password has been reset.</p><br>Happy trading!<br> - Admin</p>"%(user.name)
	send_email(subject, sender, recipients, text_body, html_body)

def password_reminder_email(user):
	subject = "Forgotten password"
	sender=('Wall Street', Config.MAIL_USERNAME)
	recipients=[user.email]
	text_body = user.password
	html_body = "<h3>Hi %s,</h3><p>Your password is: <b>%s</b> </p><p>We suggest you <a href='http://vladimirb.pythonanywhere.com/login'>sign in</a> and change your password on the user settings page.</p><br>Happy trading!<br> - Admin</p>"%(user.name, user.password)
	send_email(subject, sender, recipients, text_body, html_body)
