from bs4 import BeautifulSoup
import requests

html_text = requests.get('https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=python&txtLocation=').text
soup = BeautifulSoup(html_text,'lxml')
jobs = soup.find_all('li', class_ ='clearfix job-bx wth-shd-bx')
for job in jobs:
    published_date = job.find('span',class_='sim-posted').span.text
    if 'few' in published_date:
        company_name = job.find('h3', class_ = 'joblist-comp-name').text.replace(' ','')
        skills = job.find('span', class_ ='srp-skills').text.replace(' ','')


with open('home.html','r') as html_file:
    content = html_file.read()

    soup = BeautifulSoup(content,'lxml')
    tags = soup.find_all('h5')
    print(tags)


