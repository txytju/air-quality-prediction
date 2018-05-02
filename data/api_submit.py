
# coding: utf-8

import requests

files={'files': open('my_submission_0501_2.csv','rb')}

data = {
    "user_id": "txy_tju",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "6fd77b348c20220607f190dc8dd9cf2ef17aff5acdd803e014ee143e44bb1889", #your team_token.
    "description": '20180501_2',  #no more than 40 chars.
    "filename": "my_submission_0501_2", #your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)


