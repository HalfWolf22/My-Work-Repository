import csv
import numpy as np
import requests
import time
from datetime import datetime


#####################
key = 'RGAPI-7d79e785-a999-4930-921b-3b03bf9529ab'
#servers = ['eun', 'euw', 'na']
#####################

def get_data(server=input('Server: '), key=key, more_players_threshold=2000):

    csv_name = 'matches_' + server + '.csv'

    start = datetime.now()

    #####################
    # Getting random Summoner names from featured Games

    random_game_url = 'https://' + server + '1.api.riotgames.com/lol/spectator/v4/featured-games?api_key=' + key
    initial_summoner_names = []

    r = requests.get(random_game_url)
    j = r.json()

    import json
    with open('dump', 'w') as out:
        json.dump(j, out)

    for game in j['gameList']:
        for player in game['participants']:
            if player['bot'] is False and player['summonerName'] not in initial_summoner_names:
                initial_summoner_names.append(player['summonerName'])
    #####################

    print('\n')
    print('Got initial matches.')
    print('Time elapsed: ', datetime.now() - start)
    print('\n')

    #####################
    # Getting the Acount and Summoner ID based on Summoner names
    summoners = np.empty((0, 3))
    # Columns: Summoner Name, ID, Accont ID

    for name in initial_summoner_names[:49]:

        summoner_url = 'https://' + server + '1.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + \
            name + '?api_key=' + key

        r = requests.get(summoner_url)
        j = r.json()

        row = np.array([[name, j['id'], j['accountId']]])
        summoners = np.append(summoners, row, axis=0)
    #####################

    print('\n')
    print('Got initial summoners.')
    print('Time elapsed: ', datetime.now() - start)
    print('\n')

    #####################
    # Getting Match IDs for previous 100 matches for summoners we currently have
    matches = []

    for accid in summoners[:, 2]:

        acc_url = 'https://' + server + '1.api.riotgames.com/lol/match/v4/matchlists/by-account/' + \
            accid + '?api_key=' + key

        r = requests.get(acc_url)
        j = r.json()

        try:
            for game in j['matches']:
                if game['gameId'] not in matches:
                    matches.append(game['gameId'])
        except KeyError:
            print(j)
    #####################

    time.sleep(120)
    print('\n')
    print('Started getting the rest of the Summoners.')
    print('Time elapsed: ', datetime.now() - start)
    print('\n')

    #####################
    # Getting more Players from matches we currently have to Get more Matches

    for idx, match in enumerate(matches[:more_players_threshold]):
        if idx != 0:
            if idx % 100 == 0:
                print('\n')
                print('Done ', idx, ' out of ',
                      len(matches[:more_players_threshold]), ' matches.')
                print('Time elapsed: ', datetime.now() - start)
                print('\n')
                time.sleep(120)

        match_url = 'https://' + server + '1.api.riotgames.com/lol/match/v4/matches/' + \
            str(match) + '?api_key=' + key

        r = requests.get(match_url)
        j = r.json()

        try:
            for participant in j['participantIdentities']:
                player_info = participant['player']

                row = np.array(
                    [[player_info['summonerName'], player_info['summonerId'], player_info['accountId']]])
                if row not in summoners:
                    summoners = np.append(summoners, row, axis=0)
        except (KeyError, ConnectionError) as e:
            time.sleep(1)
    #####################

    print('\n')
    print('# of obtained summoners: ', summoners.shape[0])
    print('Time elapsed: ', datetime.now() - start)
    print('\n')
    time.sleep(120)

    #####################
    # Getting the rest of the Matches

    with open(csv_name, 'w', newline='', encoding='utf-8') as out_file:
        csv_w = csv.writer(out_file)
        # CSV for checkpoint

        for idx, accid in enumerate(summoners[49:, 2]):
            if idx != 0:
                if idx % 100 == 0:
                    print('\n')
                    print('Done ', idx, ' out of ',
                          len(summoners[49:, 2]), ' summoners.')
                    print('Time elapsed: ', datetime.now() - start)
                    print('\n')
                    time.sleep(120)

            acc_url = 'https://' + server + '1.api.riotgames.com/lol/match/v4/matchlists/by-account/' + \
                accid + '?api_key=' + key

            r = requests.get(acc_url)
            j = r.json()

            try:
                for game in j['matches']:
                    if game['gameId'] not in matches:
                        matches.append(game['gameId'])

                        csv_w.writerow([game['gameId']])
            except (KeyError, ConnectionError) as e:
                time.sleep(1)
    #####################

    print('\n')
    print('# of matches:', len(matches))
    print('Time elapsed: ', datetime.now() - start)
    print('Done')
    input('...')


if __name__ == '__main__':
    get_data()
