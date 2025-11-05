import base64
import os
from datetime import datetime, timedelta
import random

from flask import render_template
from sendgrid import Mail

from utils.sendgrid_utils import FROM_EMAIL, sg

training_data = {}
code_store = {}

SAVE_DIRECTORY = os.path.normpath(os.getenv('SAVE_DIRECTORY', '/'))


def generate_six_digit_code():
    return random.randint(100000, 999999)


def send_verification_code(email):
    code = generate_six_digit_code()
    expiration_time = datetime.now() + timedelta(minutes=10)
    code_store[email] = {'code': code, 'expires_at': expiration_time}
    return code


def remove_expired_codes():
    current_time = datetime.now()
    expired_keys = [email for email, details in code_store.items() if details['expires_at'] < current_time]
    for key in expired_keys:
        del code_store[key]


def get_email_content(name, verification_code):
    logo_image = get_image('nallai-brand.png')
    account_image = get_image('account-circle-line.png')
    with open('email_template.html', 'r') as file:
        email_template = file.read()
        email_template = email_template.replace('{{username}}', name)
        email_template = email_template.replace('{{verification_code}}', verification_code)
        email_template = email_template.replace('{{logo_image}}', logo_image)
        email_template = email_template.replace('{{account_image}}', account_image)
        return email_template


def get_image(file_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    parent_directory = os.path.dirname(current_directory)
    image_path = os.path.join(parent_directory, file_name)
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def send_email(email, verification_code):
    email_content = get_email_content(email, verification_code)
    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=email,
        subject='Your Verification Code for Login',
        html_content=email_content
    )

    try:
        sg.send(message)
        return True
    except Exception as e:
        print(e)
        return False


def serve_web_page(page_name):
    """
     Returns:
        str: HTML content of the home page.
    Raises:
        404 Not Found: If the 'home_page.html' template is not found.
    """
    try:
        return render_template(page_name)
    except Exception as e:
        # Handle the error gracefully and display a custom error message
        return render_template("error.html", error_message=f"'{page_name}' is not found"), 404


def get_user_visualisations_directory(current_user, request_id):
    file_path = get_user_directory(current_user)
    title = training_data[request_id]['title']
    file_path = os.path.join(file_path, title)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    return file_path


def get_user_directory(current_user):
    file_path = os.path.join(SAVE_DIRECTORY, current_user)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    return file_path


def remove_script_tags(script_content):
    # Remove <script> tag at the start
    script_content = script_content.replace('<script type="text/javascript">', '', 1)

    # Remove </script> tag at the end
    script_content = script_content.rsplit('</script>', 1)[0]

    return script_content
