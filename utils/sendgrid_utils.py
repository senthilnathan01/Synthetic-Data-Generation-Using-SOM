import os
import ssl

from sendgrid import SendGridAPIClient

SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
FROM_EMAIL = os.getenv('FROM_EMAIL')
TO_EMAIL = os.getenv('TO_EMAIL')

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

sg = SendGridAPIClient(SENDGRID_API_KEY)