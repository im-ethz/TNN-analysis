"""
Scrape Trainingpeaks
Select by: bike and cycling
"""
from config import username_TP, password_TP, username_LV, password_LV

import pyderman as dr
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

download_path = '/local/home/evanweenen/bicyclediabetes/Code/Data/'

url_login = 'https://home.trainingpeaks.com/login'

start_dates_list = ['1/1/2020', '1/2/2020', '1/3/2020', '1/4/2020', '1/5/2020', '1/6/2020', '1/7/2020', '1/8/2020', '1/9/2020', '1/10/2020']
end_dates_list = ['31/1/2020', '29/2/2020', '31/3/2020', '30/4/2020', '31/5/2020', '30/6/2020', '31/7/2020', '31/8/2020', '30/9/2020', '31/10/2020']

path = dr.install(browser=dr.chrome, file_directory='./lib/', verbose=True, chmod=True, overwrite=False, version='86.0.4240.22', filename=None, return_info=False)
options = webdriver.ChromeOptions()
options.add_argument("download.default_directory="+download_path)
driver = webdriver.Chrome(path, options=options)
driver.maximize_window()

driver.get(url_login)

# login
driver.find_element_by_name('Username').send_keys(username_TP)
driver.find_element_by_name('Password').send_keys(password_TP)
driver.find_element_by_name('submit').click()
driver.implicitly_wait(10)

# select athlete
driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[2]/div/div/button").click() # dropdown menu
driver.implicitly_wait(2)
athlete_elements = driver.find_elements_by_class_name('athleteOption') # list of athletes

for i in range(1, len(athlete_elements)):
	athlete_elements[i].click() # click on ith athlete
	driver.implicitly_wait(3)
	
	# go to list layout
	driver.find_element_by_class_name('searchButton').click()
	driver.implicitly_wait(3)

	driver.find_element_by_class_name('filter').click()
	driver.implicitly_wait(5)

	# select only bike training (only first time visiting website)
	if i == 1:
		driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[3]/div[3]/div/div[2]/div[5]/div[4]/div[2]/label[2]").click() # select bike
		driver.implicitly_wait(10)
		#driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[3]/div[3]/div/div[2]/div[5]/div[5]/div/div[2]/label[1]").click() # select cycling


	for d in range(len(start_dates_list)):
		driver.find_element_by_class_name('startDate').clear()
		driver.find_element_by_class_name('startDate').send_keys(start_dates_list[d]+'\n')

		driver.find_element_by_class_name('endDate').clear()
		driver.find_element_by_class_name('endDate').send_keys(end_dates_list[d]+'\n')
		driver.implicitly_wait(5)

		activities_elements = driver.find_elements_by_class_name("activity")
		for j in range(int(driver.find_element_by_class_name('totalHits').text.strip(' results'))):
			
			driver.find_elements_by_class_name("activity")[j].click()
			driver.implicitly_wait(4)

			driver.find_element_by_id('quickViewFileUploadDiv').click()
			driver.implicitly_wait(4)

			# download
			driver.find_element_by_class_name('download').click()
			driver.implicitly_wait(2)

			driver.find_element_by_id('closeIcon').click()
			driver.implicitly_wait(2)

	driver.find_element_by_class_name('closeIcon').click()

	driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[2]/div/div/button").click() # dropdown menu
	driver.implicitly_wait(1)
	athlete_elements = driver.find_elements_by_class_name('athleteOption') # list of athletes


"""
#import requests
#from bs4 import BeautifulSoup

# Start the session

headers = {
    'authority': 'home.trainingpeaks.com',
    'cache-control': 'max-age=0',
    'origin': 'null',
    'upgrade-insecure-requests': '1',
    'dnt': '1',
    'content-type': 'application/x-www-form-urlencoded',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-user': '?1',
    'sec-fetch-dest': 'document',
    'accept-language': 'nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7,de;q=0.6',
    'cookie': '__RequestVerificationToken=NOuL5JuSEjiYAX3XFfR0ys2-0jSsm7DZsSjJ0mN4D79inJcJ2HYENi6x92I5bQiLGzljXSrkDOr5DCsqc5Pck0QXzfo1',
}

data = {
  '__RequestVerificationToken': 'y3hEi8TL5IuVb1FF9psKNZd7_Tqbn7wTsKKA3Di03XMV4mx-6WL7U_aHF6AAVTBZXCPJzDN7kniyXlsJ19JW0S51OKY1',
  'Username': username_TP,
  'Password': password_TP
}


with requests.Session() as session:

	response = session.post(url_login, headers=headers, data=data)
	
	content = response.content

	asker_jeukendrup = 

	soup = BeautifulSoup(content, features="lxml")
	soup.prettify()

	wrapper = soup.find(id="wrapper")#.find(id="outerMain")#id="athleteLibrary")
	librarytabitem = soup.find_all(class_='LibraryTabItem')

#iframe
frame = driver.find_element_by_tag_name("iframe")
driver.switch_to.frame(frame)

# print element ids
ids = driver.find_elements_by_xpath('//*[@id]')
for i in ids:
	print(i.get_attribute('id'))


		# close
		#driver.switch_to.active_element
		#close_button = driver.find_element_by_id('closeIcon')
		#driver.execute_script("arguments[0].click();", close_button)
		#driver.implicitly_wait(1)
		
		#url = driver.find_element_by_class_name('download').get_attribute('fileurl')
		#wget.download(url, out=athlete_path)
		#TODO: process
"""