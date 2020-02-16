import requests
from datetime import datetime
import csv

years = ['1996-97', '1997-98', '1998-99', '1999-00', '2000-01', '2001-02',
         '2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08',
         '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14',
         '2014-15', '2015-16', '2016-17', '2017-18', '2018-19']


def get_csv(seasons, csv_name):
    start = datetime.now()
    scrape_indices = [1, 3, 6]
    counter = 0

    with open(csv_name, 'w', newline='') as out_file:
        csv_w = csv.writer(out_file)
        for season in seasons:
            print('Using: ', season)
            curr_season = [season[2:4]]
            url = f'https://stats.nba.com/stats/leaguedashplayerbiostats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&Season={season}&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='

            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0', 'Referer': url}
            r = requests.get(url, headers=headers)
            j = r.json()

            for instance in j['resultSets'][0]['rowSet']:
                write = curr_season + [instance[i] for i in scrape_indices]
                csv_w.writerow(write)

            counter += 1
            print(f'Finished cicle {counter} {datetime.now()-start}')


if __name__ == '__main__':
    csv_name = input('Name of the CSV file: ')
    get_csv(years, csv_name)
