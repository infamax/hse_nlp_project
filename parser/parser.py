import polars as pl
from selenium import webdriver
import chromedriver_binary
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict

def _extract_all_digits_from_text(text: str) -> int:
    return int("".join(filter(str.isdigit, text)))


@dataclass 
class CarInfo:
    is_available: bool = False
    gen: str = ""
    year: int = 0 
    mileage: int = 0 # km
    color: str = ""
    equipment: str = ""
    tax: int = 2
    transmission: str = ""
    drive: str = ""
    wheel_type: str = ""
    state: str = ""
    owners: str = ""
    price: int = 0 # rubles
    model_name: str = ""
    description: str = ""


    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_page(cls, driver: webdriver.Chrome, url: str):
        driver.get(url)
        car_info = driver.find_element(By.CLASS_NAME, "CardInfo-ateuv")
        soup = BeautifulSoup(car_info.get_attribute("innerHTML"), "html.parser")

        is_available_text = soup.find("li", class_="CardInfoRow CardInfoRow_availability").find_all("div", class_="CardInfoRow__cell")[1].text.replace(u"\xa0", u" ")
        is_available = (is_available_text == "В наличии")
        gen = soup.find("li", class_="CardInfoRow CardInfoRow_superGen").find_all("div", class_="CardInfoRow__cell")[1].text.replace(u"\xa0", u" ")
        year = _extract_all_digits_from_text(soup.find("li", class_="CardInfoRow CardInfoRow_year").find_all("div", class_="CardInfoRow__cell")[1].text)
        mileage = _extract_all_digits_from_text(soup.find("li", class_="CardInfoRow CardInfoRow_kmAge").find_all("div", class_="CardInfoRow__cell")[1].text.replace(u"\xa0", u" "))
        color = soup.find("li", class_="CardInfoRow CardInfoRow_color").find_all("div", class_="CardInfoRow__cell")[1].text.replace(u"\xa0", u" ")
        equipment = soup.find("li", class_="CardInfoRow CardInfoRow_complectationOrEquipmentCount").find_all("div", class_="CardInfoRow__cell")[1].text.replace(u"\xa0", u" ")
        tax = _extract_all_digits_from_text(soup.find("li", class_="CardInfoRow CardInfoRow_transportTax").find_all("div", class_="CardInfoRow__cell")[1].text)
        transmission = soup.find("li", class_="CardInfoRow CardInfoRow_transmission").find_all("div", class_="CardInfoRow__cell")[1].text
        drive = soup.find("li", class_="CardInfoRow CardInfoRow_drive").find_all("div", class_="CardInfoRow__cell")[1].text
        wheel_type = soup.find("li", class_="CardInfoRow CardInfoRow_wheel").find_all("div", class_="CardInfoRow__cell")[1].text
        state = soup.find("li", class_="CardInfoRow CardInfoRow_state").find_all("div", class_="CardInfoRow__cell")[1].text
        owners = soup.find("li", class_="CardInfoRow CardInfoRow_ownersCount").find_all("div", class_="CardInfoRow__cell")[1].text
        model_name = driver.find_element(By.CLASS_NAME, "CardHead__title").text
        price = _extract_all_digits_from_text(driver.find_element(By.CLASS_NAME, "OfferPriceCaption__price").text)
        desc = BeautifulSoup(driver.find_element(By.CLASS_NAME, "CardDescriptionHTML").get_attribute("innerHTML"), "html.parser")
        description = ""
        for span in desc.find_all("span"):
            description += span.text
        
        return cls(
            is_available=is_available,
            gen=gen,
            year=year,
            mileage=mileage,
            color=color,
            equipment=equipment,
            tax=tax,
            transmission=transmission,
            drive=drive,
            wheel_type=wheel_type,
            state=state,
            owners=owners,
            price=price,
            model_name=model_name,
            description=description,
        )
    
url = "https://auto.ru/moskva/cars/used/"

chrome_options=webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("window-size=1400,2100") 
chrome_options.add_argument('--disable-gpu')

driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(5)

for page_index in range(1, 100):
    page_url = url
    if page_index > 1:
        page_url = f"{url}?page={page_index}"

    try:
        driver.get(page_url)
        cars = driver.find_elements(By.CLASS_NAME, "Link.ListingItemTitle__link")
        car_links = [car.get_attribute("href") for car in cars]
    except: 
        continue

    car_infos = []
    for car_link in car_links:
        try:
            car_info = CarInfo.from_page(driver, car_link)
            car_infos.append(car_info)
        except:
            continue
    
    car_infos = [car_info.dict() for car_info in car_infos]
    df = pl.DataFrame(car_infos)
    print(f"df.shape: {df.shape}")
    df.write_parquet(f"cars_{page_index}.parquet")
    