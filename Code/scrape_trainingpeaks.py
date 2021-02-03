"""
Scrape Trainingpeaks
Select by:
- workout type = bike
- date range = 01/01/2020 until 31/10/2020
"""
from config import username_TP, password_TP

import pyderman as dr
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
import time

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
driver.set_page_load_timeout(10)#driver.implicitly_wait(10)

# select athlete
driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[2]/div/div/button").click() # dropdown menu
driver.set_page_load_timeout(10)#driver.implicitly_wait(2)
athlete_elements = driver.find_elements_by_class_name('athleteOption') # list of athletes

for i in range(1, len(athlete_elements)):
	athlete_elements[i].click() # click on ith athlete
	time.sleep(5)#driver.set_page_load_timeout(10)#driver.implicitly_wait(20)
	
	# go to list layout
	driver.find_element_by_class_name('searchButton').click()
	time.sleep(2)#driver.set_page_load_timeout(10)#driver.implicitly_wait(3)

	# select only bike training (only first time visiting website)
	if i == 1:
		driver.find_element_by_class_name('filter').click()
		driver.set_page_load_timeout(10)#driver.implicitly_wait(20)

		driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[3]/div[3]/div/div[2]/div[5]/div[4]/div[2]/label[2]").click() # select bike
		driver.set_page_load_timeout(10)#driver.implicitly_wait(10)
		#driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[3]/div[3]/div/div[2]/div[5]/div[5]/div/div[2]/label[1]").click() # select cycling


	for d in range(len(start_dates_list)):
		driver.find_element_by_class_name('startDate').clear()
		driver.find_element_by_class_name('startDate').send_keys(start_dates_list[d]+'\n')
		time.sleep(1)

		driver.find_element_by_class_name('endDate').clear()
		driver.find_element_by_class_name('endDate').send_keys(end_dates_list[d]+'\n')
		time.sleep(5)#driver.set_page_load_timeout(10)#driver.implicitly_wait(10)

		activities_elements = driver.find_elements_by_class_name("activity")
		for j in range(int(driver.find_element_by_class_name('totalHits').text.strip(' results'))):
			
			driver.find_elements_by_class_name("activity")[j].click()
			time.sleep(1)#driver.set_page_load_timeout(10)driver.implicitly_wait(20)

			if driver.find_element_by_id('quickViewFileUploadDiv').text != 'Upload':
				driver.find_element_by_id('quickViewFileUploadDiv').click()
				driver.set_page_load_timeout(10)#driver.implicitly_wait(4)

				# download
				driver.find_element_by_class_name('download').click()
				driver.set_page_load_timeout(10)#driver.implicitly_wait(2)

			driver.find_element_by_id('closeIcon').click()
			driver.set_page_load_timeout(10)#driver.implicitly_wait(2)

	driver.find_element_by_class_name('closeIcon').click()
	driver.set_page_load_timeout(10)

	driver.find_element_by_xpath("//*[@id='main']/div[1]/div/div/div[2]/div/div/button").click() # dropdown menu
	driver.set_page_load_timeout(10)#driver.implicitly_wait(1)
	athlete_elements = driver.find_elements_by_class_name('athleteOption') # list of athletes