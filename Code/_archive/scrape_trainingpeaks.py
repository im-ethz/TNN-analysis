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