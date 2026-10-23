import datetime
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

###############################################################################
# notes
# pip install selenium
# get webdriver from https://selenium-python.readthedocs.io/installation.html

###############################################################################

item1 = "Abc"
url1 = "https://www.amazon.co.uk/Dartington-Crystal-White-Wine-Glasses/dp/B0759ZLRL1/?_encoding=UTF8&pd_rd_w=mpCcZ&content-id=amzn1.sym.cb5e8cad-675d-47f0-b759-3d7e2d67f357&pf_rd_p=cb5e8cad-675d-47f0-b759-3d7e2d67f357&pf_rd_r=GJP7C5YXX2N3RCY5WFR9&pd_rd_wg=skkTO&pd_rd_r=ebbab47c-3bd1-4ecf-ac92-c4beb6880ca7&ref_=pd_hp_d_r_atf_dealz_wd_des"

###############################################################################

start_time = time.time()
item_names = []
urls = []
prices = []
times = []

def web_query(item,url,ref):
    opts = webdriver.ChromeOptions()
    # The below line will make your browser run in background when uncommented
    opts.add_argument('--headless')

    service = Service(executable_path='C:\webdrivers\chromedriver.exe')
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=opts)
        
    #driver = webdriver.Chrome(options= opts, executable_path = 'C:\webdrivers\chromedriver.exe')

    driver.get(url)

    #prices = driver.find_elements_by_class_name('main-price')
    price = driver.find_element(By.CLASS_NAME, ref)
    pricer = price.text
    driver.quit()
    
    # get time
    timer = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # append to list
    item_names.append(item)
    urls.append(url)
    prices.append(pricer)
    times.append(timer)
    
###############################################################################

# run for e.g. as many instances

web_query(item1,url1, "a-price-whole")
web_query(item1,url1, "a-price-whole")

# print the details at the end
for entry in range(0,len(item_names)):
    print(f"The item {item_names[entry]} is now {prices[entry]} as at {times[entry]}")


print("runtime of processing: " +str(round(time.time()-start_time,3))+" seconds")
