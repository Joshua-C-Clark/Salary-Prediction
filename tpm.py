from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--disable-notifications")

URL = 'https://www.levels.fyi/comp.html?track=Technical%20Program%20Manager'

driver = webdriver.Chrome(options=options)

Companies = []
Location = []
Date = []
Job_Title = []
Subspecialty = []
Years_Experience = []
Total_Comp = []

driver.get(URL)


page = 1

WebDriver.find_element_by_xpath(driver, "//span[contains(@class, 'btn-group dropup')]/button").click()
WebDriver.find_element_by_xpath(driver, "//span[contains(@class, 'btn-group dropup')]/ul/li[4]").click()

max_page = WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//html/body/div[2]/div/div[3]/div[3]/div/div[1]/div[1]/div[3]/div[2]/ul/li[8]/a")))[0].text
# max_page = int(max_page)

# Had to hard code the 6 for max pages. Re-investigate the crash
# error later

while page <= 6:
    rows= WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//*[@id='compTable']/tbody//tr")))
    # np = WebDriver.find_element_by_xpath(driver,"//li[contains(@class, 'page-next')]/a")

    for row in rows:
        # The Company, Date, Location, and Total_Comp scrapings work.
        Companies.append(row.find_element_by_xpath('.//td[2]/span[2]').text)
        Date.append(row.find_element_by_xpath('.//td[2]/span[4]').text[-8::])
        Location.append(row.find_element_by_xpath('.//td[2]/span[4]').text[:-10])
        Job_Title.append(row.find_element_by_xpath('.//td[3]/span[1]').text)
        Subspecialty.append(row.find_element_by_xpath('.//td[3]/span[2]').text)
        Years_Experience.append(row.find_element_by_xpath('.//td[4]').text[-2::])
        Total_Comp.append(row.find_element_by_xpath('.//td[5]/div[1]/div[1]/span[1]').text)

    try:   
        WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//li[contains(@class, 'page-next')]/a")))[0].click()
    except:
        WebDriver.find_element_by_xpath(driver, "//div[contains(@class, 'submit-compensation-remodal')]/button").click()
        WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.XPATH, "//li[contains(@class, 'page-next')]/a")))[0].click()
    print(f'Navigating to page {page + 1}')
    page += 1

driver.close()

df = pd.DataFrame(list(zip(Companies, Location, Date, Job_Title, Subspecialty, Years_Experience, Total_Comp)), columns=['Company', 'Location', 'Date', 'Job_Title', 'Subspecialty', 'Years_Experience', 'Total_Comp'])

df.to_csv('./data/technical_program_manager.csv')
