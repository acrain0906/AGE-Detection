

# =============================================================================
# End Of Program Notify 
# =============================================================================
import atexit
import sys

# test if credentials are loaded, else, set variable to prevent using missing variables 
login = True 
try:
	import credentials
except: 
	print('failed to load credentials')
	login = False
	
	#from testfixtures import LogCapture

# sys.stderr = 
log = open('log.txt', 'w')
# sys.stdout = notify.log

def exit_handler():
	# mail('Finished', 'Check It!')
	global log
	subject = 'model'
	for arg in sys.argv:
		subject += '-' + arg
	# subject += ' finished'
	
	log.close()
	with open('log.txt', 'r') as f:
		body = '\n'.join(f.readlines())
	if 'Traceback' in body:
		subject = 'Failed: ' + subject 
	else:
		subject = 'Finished: ' + subject 
	mail(subject, body)
	# text('hi sexy')


# =============================================================================
# Texting 
# =============================================================================

from twilio.rest import Client

def text (message):
	if login:
		account_sid = credentials.ACCOUNT_SID 
		auth_token = credentials.AUTH_TOKEN 
		client = Client(account_sid, auth_token)

		message = client.messages.create(
			body='Austin is waiting',
			from_= '+13344893316',
			to='+16105540906'
		)
		
# =============================================================================
# Email 
# =============================================================================

import smtplib

def mail (subject, body):
	if login:
		gmail_user = 'liuf0134@gmail.com'
		gmail_password = credentials.EPWD

		sent_from = gmail_user
		to = 'aoc5329@gmail.com'

		email_text = """\
		Subject: %s

		%s
		""" % (subject, body)
		
		email_text = 'Subject:{}\n\n{}'.format(subject, body)

		# # type your message: use two newlines (\n) to separate the subject from the message body, and use 'f' to  automatically insert variables in the text
		# message = f"""\
		# Subject: Hi Mailtrap
		# To: {sent_to}
		# From: {sent_from}

		# This is my first message with Python."""
		server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
		server.ehlo()
		server.login(gmail_user, gmail_password)
		server.sendmail(sent_from, to, email_text)
		server.close()
	
	
	
	
if __name__ == '__main__':
	mail('Finished Running', '')
else:
	atexit.register(exit_handler)