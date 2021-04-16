import csv
import requests
import time
from datetime import datetime


#####################
key = 'RGAPI-964004e6-9832-4894-a913-152301f1f036'
current_patch = '10.19'
#####################


def get_data(server, start, stop, csv_name, key):

    # Matches to scrape
    with open('matches_' + server + '.csv', newline='') as file:
        r = csv.reader(file)
        matches = [i for slist in list(r) for i in slist]  # unpacking list of lists

        start_time = datetime.now()

        #####################
        # Getting the ML Data
        team1_indices = [1] + [8, 9, 10, 11]
        team2_indices = [8, 9, 10, 11]
        player_stats = [9, 10, 11, 41, 45]
        # See raw JSON on the Riot API website or the rest of this project for columns

        with open(csv_name, 'w', newline='', encoding='utf-8') as out_file:
            csv_w = csv.writer(out_file)

            for idx, match in enumerate(matches[start:stop]):  # Put indices to scrape here
                if idx != 0:
                    if idx % 100 == 0:
                        print('\n')
                        print('Done ', idx, ' out of ', stop - start, 'matches.')
                        print('Time elapsed: ', datetime.now() - start_time)
                        print('\n')
                        time.sleep(80)

                match_url = 'https://' + server + '1.api.riotgames.com/lol/match/v4/matches/' + str(match) + '?api_key=' + key

                r = requests.get(match_url)
                j = r.json()

                try:
                    if j['gameMode'] == 'CLASSIC' and j['gameType'] == 'MATCHED_GAME' and j['gameDuration'] >= 1500 and j['gameVersion'][:5] == current_patch:
                        row = [match] + [j['gameDuration']]
                        row = row + [list(j['teams'][0].values())[i] for i in team1_indices]
                        row = row + [list(j['teams'][1].values())[i] for i in team2_indices]

                        for participant in j['participants']:
                            row = row + [participant['championId']]
                            row = row + [list(participant['stats'].values())[i] for i in player_stats]

                            row = row + [participant['timeline']['role'], participant['timeline']['lane']]

                        csv_w.writerow(row)

                except (KeyError, ConnectionError) as e:
                    pass
        #####################


if __name__ == '__main__':
    server = input('Server: ')
    start = int(input('Start: '))
    stop = int(input('Stop: '))
    csv_name = 'games_' + server + input('CSV name index: ') + '.csv'

    print('Started')
    get_data(server, start, stop, csv_name, key)
    print('Done\n...')
