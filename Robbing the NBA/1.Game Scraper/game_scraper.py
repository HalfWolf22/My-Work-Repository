import requests
from datetime import datetime
import csv

years = list(range(96, 100)) + list(range(0, 19))
num_games = 1231

main_urls = []
scrape_urls = []

for year in years:
    if year == 96 or year == 97 or year == 99 or year < 4:
        num_games = 29 * 41 + 1
    elif year == 98:
        num_games = 29 * 25 + 1
    elif year == 11:
        num_games = 30 * 33 + 1

    year = str(year).zfill(2)
    for game in range(1, num_games):
        game = str(game).zfill(5)
        main_urls.append('https://stats.nba.com/game/002' + year + game)
        scrape_urls.append(
            'https://stats.nba.com/stats/boxscoresummaryv2?GameID=002' + year + game)
    num_games = 1231

url_pairs = [[main_urls[i], scrape_urls[i]] for i in range(0, len(main_urls))]
url_pairs = [urls for urls in url_pairs if urls[0][30:37] != '1201214']


def get_csv(url_pairs, csv_name):
    start = datetime.now()
    with open(csv_name, 'w', newline='') as out_file:
        csv_w = csv.writer(out_file)

        main_indices = [0, 2, 4, 7, 8, 9, 10, 11, -1]
        misc_indices = [4, 5, 6, 11]
        counter = 0

        for game in url_pairs:
            print('Using: ', game[0][33:37])
            year = game[0][30:32]

            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0', 'Referer': game[0]}
            r = requests.get(game[1], headers=headers)
            j = r.json()

            for team in range(2):
                if int(year) > 13:
                    if team == 0:
                        main = [j['resultSets'][5]['rowSet'][1][i]
                                for i in main_indices]
                        try:
                            misc = [j['resultSets'][1]['rowSet'][team][i]
                                    for i in misc_indices]
                        except IndexError:
                            misc = ['NR', 'NR', 'NR', 'NR']
                    else:
                        main = [j['resultSets'][5]['rowSet'][0][i]
                                for i in main_indices]
                        try:
                            misc = [j['resultSets'][1]['rowSet'][team][i]
                                    for i in misc_indices]
                        except IndexError:
                            misc = ['NR', 'NR', 'NR', 'NR']
                else:
                    main = [j['resultSets'][5]['rowSet'][team][i]
                            for i in main_indices]
                    try:
                        misc = [j['resultSets'][1]['rowSet'][team][i]
                                for i in misc_indices]
                    except IndexError:
                        misc = ['NR', 'NR', 'NR', 'NR']

                row = main + misc
                csv_w.writerow(row)
            counter += 1
            print(f'Finished cicle {counter} {datetime.now()-start}')


if __name__ == '__main__':
    years_to_get = input('Two last digits of desired year, split with space: ')
    years_to_get = years_to_get.split()
    desired_urls = [urls for urls in url_pairs if urls[0]
                    [-7:-5] in years_to_get]

    csv_name = input('Name of the CSV file: ')

    get_csv(desired_urls, csv_name)
