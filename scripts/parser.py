import argparse
import os
from dataclasses import dataclass
import itertools

import polars as pl
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
from joblib import Parallel, delayed

CITY_NAMES = [
    "moskva",
    "sankt-peterburg",
    "ekaterinburg",
    "novosibirsk",
    "kaliningrad",
    "voronezh",
    "saratov",
    "samara",
]
AUTO_RU_URL = "https://auto.ru/{}/cars/used/"


def _extract_all_digits_from_text(text: str) -> int:
    return int("".join(filter(str.isdigit, text)))


def _get_car_page_value(
    soup: BeautifulSoup,
    li_class_name: str,
    div_class_name: str,
    default_value: str = "",
) -> str:
    try:
        return (
            soup.find("li", class_=li_class_name)
            .find_all("div", class_=div_class_name)[1]
            .text
        )
    except:
        return default_value


@dataclass
class CarInfo:
    is_available: bool | None = False
    gen: str | None = ""
    year: int | None = 0
    mileage: int | None = 0  # km
    color: str | None = ""
    equipment: str | None = ""
    tax: int | None = 2
    transmission: str | None = ""
    drive: str | None = ""
    wheel_type: str | None = ""
    state: str | None = ""
    owners: int | None = ""
    price: int | None = 0  # rubles
    model_name: str | None = ""
    description: str | None = ""

    def dict(self):
        return {
            "is_available": self.is_available,
            "gen": self.gen,
            "year": self.year,
            "mileage": self.mileage,
            "color": self.color,
            "equipment": self.equipment,
            "tax": self.tax,
            "transmission": self.transmission,
            "drive": self.drive,
            "wheel_type": self.wheel_type,
            "state": self.state,
            "owners": self.owners,
            "price": self.price,
            "model_name": self.model_name,
            "description": self.description,
        }

    @classmethod
    def from_page(cls, driver: webdriver.Chrome, url: str):
        driver.get(url)
        car_info = driver.find_element(By.CLASS_NAME, "CardInfo-ateuv")
        soup = BeautifulSoup(car_info.get_attribute("innerHTML"), "html.parser")

        is_available_text = _get_car_page_value(
            soup=soup,
            li_class_name="CardInfoRow CardInfoRow_availability",
            div_class_name="CardInfoRow__cell",
            default_value="",
        ).replace("\xa0", " ")

        is_available = is_available_text == "В наличии"

        gen = _get_car_page_value(
            soup=soup,
            li_class_name="CardInfoRow CardInfoRow_superGen",
            div_class_name="CardInfoRow__cell",
            default_value="",
        ).replace("\xa0", "")

        year = _extract_all_digits_from_text(
            soup.find("li", class_="CardInfoRow CardInfoRow_year")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        mileage = _extract_all_digits_from_text(
            soup.find("li", class_="CardInfoRow CardInfoRow_kmAge")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text.replace("\xa0", " ")
        )
        color = (
            soup.find("li", class_="CardInfoRow CardInfoRow_color")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text.replace("\xa0", " ")
        )
        equipment = (
            soup.find(
                "li", class_="CardInfoRow CardInfoRow_complectationOrEquipmentCount"
            )
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text.replace("\xa0", " ")
        )
        tax = _extract_all_digits_from_text(
            soup.find("li", class_="CardInfoRow CardInfoRow_transportTax")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        transmission = (
            soup.find("li", class_="CardInfoRow CardInfoRow_transmission")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        drive = (
            soup.find("li", class_="CardInfoRow CardInfoRow_drive")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        wheel_type = (
            soup.find("li", class_="CardInfoRow CardInfoRow_wheel")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        state = (
            soup.find("li", class_="CardInfoRow CardInfoRow_state")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        owners = _extract_all_digits_from_text(
            soup.find("li", class_="CardInfoRow CardInfoRow_ownersCount")
            .find_all("div", class_="CardInfoRow__cell")[1]
            .text
        )
        model_name = driver.find_element(By.CLASS_NAME, "CardHead__title").text
        price = _extract_all_digits_from_text(
            driver.find_element(By.CLASS_NAME, "OfferPriceCaption__price").text
        )
        desc = BeautifulSoup(
            driver.find_element(By.CLASS_NAME, "CardDescriptionHTML").get_attribute(
                "innerHTML"
            ),
            "html.parser",
        )
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


def _parse_auto_ru(url: str, city_name: str) -> list[CarInfo]:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument(
        "--disable-dev-shm-usage"
    )  # Overcome limited resource problems

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5)
    car_info_pages = []
    for page_index in range(1, 100):
        page_url = url
        if page_index > 1:
            page_url = f"{url}?page={page_index}"

        try:
            driver.get(page_url)
            cars = driver.find_elements(By.CLASS_NAME, "Link.ListingItemTitle__link")
            car_links = [car.get_attribute("href") for car in cars]
        except Exception as err:
            print(f"Cannot get cars by url: {page_url}. Error: {str(err)}")
            continue

        car_infos = []
        for car_link in car_links:
            try:
                car_info = CarInfo.from_page(driver, car_link)
                car_infos.append(car_info)
            except Exception as e:
                continue

        car_infos = [car_info.dict() for car_info in car_infos]
        df = pl.DataFrame(car_infos)
        print(
            f"Number parsed cars from page: {page_index} = {df.shape[0]} for city: {city_name}"
        )
        car_info_pages.append(df)

    return car_info_pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--destination",
        help="Folder name to be dataset collected",
        type=str,
        default="dataset",
    )
    args = parser.parse_args()
    car_dfs = []
    car_dfs = Parallel(n_jobs=len(CITY_NAMES))(
        delayed(_parse_auto_ru)(AUTO_RU_URL.format(city_name), city_name)
        for city_name in CITY_NAMES
    )
    car_dfs = list(itertools.chain.from_iterable(car_dfs))
    car_dfs = [car_df for car_df in car_dfs if car_df.shape[0] != 0]
    final_df = pl.concat(car_dfs)
    os.makedirs(args.destination, exist_ok=True)
    path_to_final_df = os.path.join(args.destination, "auto_ru_cars.parquet")
    final_df.write_parquet(path_to_final_df)
    return 0


if __name__ == "__main__":
    exit(main())
