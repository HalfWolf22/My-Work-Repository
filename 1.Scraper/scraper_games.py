import csv
import requests
import time
from datetime import datetime

#####################
# The scraper had to be split because pauses
# that had to be made made the code execute too slowly.
# The scraper as a whole could work if this chunk of code
# was just added to the end of the get_data function
# from scraper_matches but it is too slow because there
# are no real checkpoints besides writing to a CSV.
#
# NOTE:
# There is a specified input that the user must provide
# for the name of the csv file to avoid forgetting to change
# the csv name within the code and in doing so, overwriting
# the file(s) with the games that have already been scraped.
# (Clearly not happened to me, just in case so it does not
# happen to somebody else :D)
#####################

#####################
key = 'RGAPI-43e9d385-ec08-4063-b017-fe8980ab031d'
#####################

# Matches to scrape
with open('matches.csv', newline='') as file:
    r = csv.reader(file)
    matches = [i for slist in list(r)
               for i in slist]  # unpacking list of lists

def get_data():

    start = datetime.now()

    #####################
    # Getting the ML Data
    team1_indices = list(range(1, 12)) + [13]
    team2_indices = [4, 5] + list(range(7, 12)) + [13]
    player_stats = [0] + list(range(9, 16)) + [21, 26, 30, 33, 35, 41, 45]
    # See raw JSON on the Riot API website or the rest of this project for columns

    with open(csv_name, 'w', newline='', encoding='utf-8') as out_file:
        csv_w = csv.writer(out_file)

        for idx, match in enumerate(matches):  # Put indices to scrape here
            if idx != 0:
                if idx % 100 == 0:
                    print('\n')
                    print('Done ', idx, ' out of ',
                          len(matches), 'matches.')
                    print('Time elapsed: ', datetime.now() - start)
                    print('\n')
                    time.sleep(80)

            match_url = 'https://eun1.api.riotgames.com/lol/match/v4/matches/' + \
                str(match) + '?api_key=' + key

            r = requests.get(match_url)
            j = r.json()

            try:
                if j['gameMode'] == 'CLASSIC' and j['gameType'] == 'MATCHED_GAME' and j['gameDuration'] > 240:
                    row = [match]
                    row = row + [list(j['teams'][0].values())[i]
                                 for i in team1_indices]
                    row = row + [list(j['teams'][1].values())[i]
                                 for i in team2_indices]

                    for participant in j['participants']:
                        row = row + [list(participant['stats'].values())[i]
                                     for i in player_stats]

                        row = row + [participant['timeline']['role'],
                                     participant['timeline']['lane']]
                        row = row + [i for i in list(participant['timeline']
                                                     ['creepsPerMinDeltas'].values())]

                    csv_w.writerow(row)

            except (KeyError, ConnectionError) as e:
                pass
    #####################


if __name__ == '__main__':
    csv_name = input('Name of the Csv: ')
    if csv_name[-4:] != '.csv':
        csv_name += '.csv'

    print('Started')
    get_data()
    print('Done\n...')
