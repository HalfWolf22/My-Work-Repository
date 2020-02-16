import requests
from datetime import datetime
import csv

years = list(range(96, 100)) + list(range(0, 19))
num_games = 1231

main_urls = []
scrape_urls = []
date_urls = []

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
            f'https://stats.nba.com/stats/boxscoretraditionalv2?EndPeriod=10&EndRange=28800&GameID=002{year}{game}&RangeType=0&SeasonType=Regular+Season&StartPeriod=1&StartRange=0')
        date_urls.append(
            'https://stats.nba.com/stats/boxscoresummaryv2?GameID=002' + year + game)
    num_games = 1231


url_pairs = [[main_urls[i], scrape_urls[i],
              date_urls[i]] for i in range(0, len(main_urls))]


def get_csv(url_pairs, csv_name):
    start = datetime.now()
    with open(csv_name, 'w', newline='') as out_file:
        csv_w = csv.writer(out_file)

        scrape_indices = [0, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                          18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        counter = 0

        for game in url_pairs:
            print('Using: ', game[0][33:37])
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0', 'Referer': game[0]}
            r = requests.get(game[2], headers=headers)
            j = r.json()

            date = [j['resultSets'][0]['rowSet'][0][0][:10]]
            away_team = j['resultSets'][0]['rowSet'][0][5][-6:-3]

            r = requests.get(game[1], headers=headers)
            j = r.json()

            for instance in j['resultSets'][0]['rowSet']:
                stats = [instance[i] for i in scrape_indices]
                if stats[1] == away_team:
                    away_home = ['Away']
                else:
                    away_home = ['Home']

                row = date + away_home + stats
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
