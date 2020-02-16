from bs4 import BeautifulSoup
from selenium import webdriver

from datetime import datetime
import csv

urls = [
    f'https://sportsdatabase.com/nba/query?output=default&sdql=date+and+team+and+season%3D{season}+and+line&submit=++S+D+Q+L+%21++' for season in range(1996, 2019)]


def get_csv(csv_name):
    start = datetime.now()
    with open(csv_name, 'w', newline='') as out_file:
        csv_w = csv.writer(out_file)

        counter = 0

        for url in urls:
            curr_row = []

            driver = webdriver.Chrome()
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            chunks = soup.find('table', {'id': 'DT_Table', 'class': 'dataTable no-footer',
                                               'role': 'grid', 'aria-describedby': 'DT_Table_info'}).find('tbody').find_all('tr')

            for tr in chunks:
                txt = tr.a.text
                txt_list = txt.split(' and ')
                curr_row.append(txt_list[0][7:])
                curr_row.append(txt_list[1])
                curr_row.append(txt_list[3][7:])
                csv_w.writerow(curr_row)
                curr_row = []

            driver.close()
            counter += 1
            print(
                f'Finished URL #{counter}; Time elapsed: {datetime.now()-start}')


if __name__ == '__main__':
    csv_name = input('Name of the CSV file: ')
    get_csv(csv_name)

    input('Done...')
