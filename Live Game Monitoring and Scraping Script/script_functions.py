import requests
import json
import datetime as dt

from parameters import LINE_W, LEAGUES

###################
# AUXILIARY FUNCTIONS #
###################


# Get the json
def get_json(url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0',
               'Referer': url, 'x-api-key': '0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z'}

    r = requests.get(url, headers=headers)
    j = r.json()

    return j


# Dump json for preview
def dump_json(j, file_name):
    if file_name[-4:] != '.txt':
        file_name += '.txt'

    with open(file_name, 'w') as o:
        json.dump(j, o)


# Formatting datetime for querrying
def get_time():
    # Converting time zones
    curr_time = str(dt.datetime.now() - dt.timedelta(hours=2, seconds=22))
    # Rounding to nearest 10 seconds
    curr_time = curr_time[:-8] + '0'

    return curr_time[:10] + 'T' + curr_time[11:] + 'Z'


# Convert querry dt string to dt
def get_time_todt(to_convert):
    return dt.datetime.strptime(to_convert, '%Y-%m-%dT%H:%M:%SZ')


# Move ANSI cursor up X lines
def move_up(x):
    print('\033[A' * x)


# Delete ANSI up X lines
def del_up(x):
    move_up(x)
    print((' ' * LINE_W + '\n') * (x - 1))
    move_up(x + 1)


# Center-printing stats
def print_mid(statline, b_stat='', r_stat=''):
    b_stat, r_stat = str(b_stat), str(r_stat)
    print(b_stat.rjust(30, ' ') + statline.center(LINE_W - 60, ' ') + r_stat.ljust(30, ' '))


###################
# SCRAPER FUNCTIONS #
###################

# Get ongoing matches
def get_live_matches(fin_matches):
    live = get_json('https://esports-api.lolesports.com/persisted/gw/getLive?hl=en-US')

    # Return empty list instead of None if there aren't any matches
    events = live['data']['schedule']['events']
    if events is None:
        events = []

    # Filter current events to get matches that are in progress
    matches = [e for e in events if e['state'] == 'inProgress' and e['type'] == 'match']

    # Filter current matches by the league
    matches = [m for m in matches if m['league']['name'] in LEAGUES]

    # Filter by blacklisted and completed
    matches = [m for m in matches if m['id'] not in fin_matches]

    return matches


# Get upcoming games for specified leagues
def get_schedule():
    schedule = get_json(
        'https://esports-api.lolesports.com/persisted/gw/getSchedule?hl=en-US')

    # Filter matches by league
    schedule = [e for e in schedule['data']['schedule']['events'] if e['type'] == 'match' and e['league']['name'] in LEAGUES]

    # Filter by time (only display upcoming)
    schedule = [e for e in schedule if e['state'] == 'unstarted']

    return schedule


# Get match data
def get_match(sample_match_id):
    sample_match = get_json('https://esports-api.lolesports.com/persisted/gw/getEventDetails?hl=en-US&id=' + sample_match_id)
    return sample_match['data']['event']['match']


# Get game data
def get_game(game_id):
    return get_json('https://feed.lolesports.com/livestats/v1/window/' + game_id + '?startingTime=' + get_time())
