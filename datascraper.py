from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
import random
import numpy as np
import pandas as pd
import concurrent.futures

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")

def get_data(urls):
    start = datetime.now()

    index = range(236236)
    columns = ['Date', 'Game', 'Team', 'Home/Away', 'Player', 'Min',
               'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA',
               'FT%', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 
               'PF', 'PTS', '+/-']

    data = pd.DataFrame(index=index, columns=columns)

    objs_list = ['Date', 'Team', 'Home/Away', 'Player', 'Min']

    for col in data.columns:
        if col not in objs_list:
            data[col] = pd.to_numeric(data[col])

    counter_ix = 0
    counter_col = 0

    old_tables = ['96', '97', '98', '99', '00', '01', '02', '03', '04', 
                  '05', '06']
    
    counter = 0
    
    for game in urls:
        print('Using: ', game)
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(game)

        page = driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(page, 'html.parser')
        table = soup.findAll('tbody')
    
        if game[-7:-5] in old_tables:  
            tables = [['Away', table[1]], ['Home', table[3]]]
        else:
            try:
                tables = [['Away', table[4]], ['Home', table[6]]]
            except IndexError:
                tables = [['Away', table[1]], ['Home', table[3]]]
    
        date = soup.find('div', {'class': 'game-summary__date'}).get_text()   
        game_id = game[-10:]    
    
        away_team = soup.title.get_text()[-10:-7]
        home_team = soup.title.get_text()[-3:]
    
        for tag, table in tables:
            for row in table.find_all('tr'):
        
                data.iloc[counter_ix, counter_col] = date
                counter_col += 1
        
                data.iloc[counter_ix, counter_col] = game_id
                counter_col += 1
        
                if tag == 'Away':
                    data.iloc[counter_ix, counter_col] = away_team
                    counter_col += 1
                elif tag == 'Home':
                    data.iloc[counter_ix, counter_col] = home_team
                    counter_col += 1
            
                data.iloc[counter_ix, counter_col] = tag
                counter_col += 1
                
                for instance in row.find_all('td'):
                    cell = instance.get_text().replace('\n', '').replace(' ', '')
                      
                    if counter_col in [4, 5]:
                        cell = cell
                    else:
                        try:
                            cell = pd.to_numeric(cell)
                        except:
                            data.iloc[counter_ix, 5] = None
                            cell = None
                
                    data.iloc[counter_ix, counter_col] = cell
                    counter_col += 1
            
                counter_ix += 1
                counter_col = 0
    
        driver.quit()
        print('Time Elapsed: ', datetime.now()-start)
        counter += 1
        print('Process ', counter, ' finished')
        
    return data.loc[~data['PTS'].isnull()]
    
url = 'https://stats.nba.com/game/002'
years = list(range(96,100))+list(range(0, 19))
num_games = 1231

all_games = []

for year in years:
    if year == 96 or year == 97 or year == 99 or year < 4:
        num_games = 29*41+1
    elif year == 98:
        num_games = 29*25+1
    elif year == 11:
        num_games = 30*33+1
    
    year = str(year).zfill(2)
    for game in range(1, num_games):
        game = str(game).zfill(5)
        all_games.append(url+year+game)
    num_games = 1231
    
#test_list = random.sample(all_games, 50)


if __name__ == '__main__':
    years_to_get = input('Two last digits of desired year, split with space: ')
    years_to_get = years_to_get.split()
    desired_urls = [url for url in all_games if url[-7:-5] in years_to_get]
    
    splits = np.array_split(desired_urls, 2)
    list1 = splits[0].tolist()
    list2 = splits[1].tolist()
    scrape_lists = [list1, list2]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(get_data, scrape_lists)
        print([verbose for verbose in results])
        
    input('ASDQWRsg')
        
    final_data = pd.concat([df for df in results])
    final_data.to_csv(path=r'C:\Users\vucin\Destop\Robbing the NBA')