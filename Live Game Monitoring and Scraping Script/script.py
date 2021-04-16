# Core
import joblib
import datetime as dt
import time
import numpy as np
import csv


# Console control
import colorama  # Cursor control
import cursor  # Hide cursor
import os
import warnings


# Import parameters and functions
from parameters import LINE_W, LOGGING
from live_bets import init_driver, get_odds, get_last_odds
from script_functions import get_json, dump_json, get_time, get_time_todt, move_up, del_up, print_mid
from script_functions import get_live_matches, get_schedule, get_match, get_game


#############
# PARAMETERS #
#############

# Init colorama
colorama.init()

# Hide the cmd cursor
cursor.hide()

# Enable "Ctrl-C" Keyboard interupt to properly exit
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# Outdated model version warning
warnings.simplefilter("ignore", UserWarning)

os.system(f'mode con: cols={LINE_W} lines=33')

model = joblib.load('lr_pipeline.pkl')

fin_matches = []  # Finished or blacklisted matches ids


########
# SCRIPT #
########

def main(csv_file=0):

    matches = get_live_matches(fin_matches)

    #################################
    # If there aren't any active matches currently #
    #################################

    while matches == []:
        print_mid('No matches currently availabile')
        print('\n')

        upcoming_match = get_schedule()[0]

        # Teams that are playing in the next match
        upcoming_teams = upcoming_match['match']['teams']
        upcoming_time = get_time_todt(upcoming_match['startTime']) - get_time_todt(get_time())

        print_mid(upcoming_teams[0]['name'] + ' vs ' + upcoming_teams[1]['name'] + ' (' + upcoming_match['league']['name'] + ')')

        if upcoming_time < dt.timedelta(0):
            print_mid('Starts: soon')
        else:
            print_mid(f'Starts in: {upcoming_time}')

        time.sleep(5)

        matches = get_live_matches(fin_matches)

        # Update printed text - when while isn't/is broken
        if matches == []:
            move_up(6)
        else:
            del_up(6)

    ######################################
    # Select which match to follow if there are multiple #
    ######################################

    if len(matches) > 1:
        for idx, match in enumerate(matches):
            teams = matches[idx]['match']['teams']
            print(str(idx) + '.', teams[0]['name'], 'vs', teams[1]['name'])

        selected_match = int(input('\nEnter the index of the desired match: '))
        sample_match_id = matches[selected_match]['id']

        sample_league = matches[selected_match]['league']['name']
        del_up(5)
    else:
        sample_match_id = matches[0]['id']
        sample_league = matches[0]['league']['name']

    sample_match = get_match(sample_match_id)

    ###############
    # Match metadata #
    ###############

    print_mid('Initializing Live Bet Scraper...')
    driver = init_driver()
    del_up(2)

    team1 = [sample_match['teams'][0]['name'], sample_match['teams'][0]['id']]
    team2 = [sample_match['teams'][1]['name'], sample_match['teams'][1]['id']]

    game_count = len(sample_match['games'])

    #################
    # Match is in progress #
    #################

    game_states = [g['state'] for g in sample_match['games']]

    while 'unstarted' in game_states or 'inProgress' in game_states:

        ##########################################
        # Pre-game show - all games in the match are 'unstarted' #
        ##########################################

        if all(i == 'unstarted' for i in game_states):
            curr_game = sample_match['games'][0]
        else:
            curr_game = [g for g in sample_match['games'] if g['state'] == 'inProgress'][0]

        game_id = curr_game['id']
        game_num = curr_game['number']

        ########################################
        # Pre-game - match ongoing but no games are played #
        ########################################

        print_mid(f'Game {game_num} about to begin: {team1[0]} vs {team2[0]}')

        game = None
        while game is None:
            try:
                game = get_game(game_id)
                del_up(2)
            except:
                time.sleep(5)

        #########################
        # Game metadata - blue, red side #
        #########################

        if team1[1] == game['gameMetadata']['blueTeamMetadata']['esportsTeamId']:
            b_team = team1[0]
            r_team = team2[0]
        else:
            b_team = team2[0]
            r_team = team1[0]

        # Print teams for orientation in the stats table
        def print_teams():
            print_mid(sample_league + f', Game {game_num}/{game_count}', b_team, r_team)

        b_players = [[p['role'], p['championId']] for p in game['gameMetadata']['blueTeamMetadata']['participantMetadata']]
        r_players = [[p['role'], p['championId']] for p in game['gameMetadata']['redTeamMetadata']['participantMetadata']]

        #######
        # Stats #
        #######

        game_stats = game['frames'][-1]  # Latest stats

        while game_stats['gameState'] != 'finished':
            b_stats = game_stats['blueTeam']
            r_stats = game_stats['redTeam']

            # Blue_TtlGold, Purp_TtlGold
            b_gold, r_gold = b_stats['totalGold'], r_stats['totalGold']

            # Game about to begin
            b_gold, r_gold = b_stats['totalGold'], r_stats['totalGold']

            # Blue_KillsTower, Blue_KillsInhib, Blue_KillsBaron
            b_tower, b_inhib, b_baron = b_stats['towers'], b_stats['inhibitors'], b_stats['barons']

            # Purp_KillsTower, Purp_KillsInhib, Purp_KillsBaron
            r_tower, r_inhib, r_baron = r_stats['towers'], r_stats['inhibitors'], r_stats['barons']

            # Blue_Dragons, Purp_Dragons
            b_dragon, r_dragon = len(b_stats['dragons']), len(r_stats['dragons'])

            # Blue_Kills, Blue_Assists
            b_kills = sum([p['kills'] for p in b_stats['participants']])
            b_assists = sum([p['assists'] for p in b_stats['participants']])

            # Purp_Kills, Purp_Assists
            r_kills = sum([p['kills'] for p in r_stats['participants']])
            r_assists = sum([p['assists'] for p in r_stats['participants']])

            features = [b_tower, b_inhib, b_baron, b_dragon, r_tower, r_inhib, r_baron, r_dragon, b_kills, b_assists, b_gold, r_kills, r_assists, r_gold]
            diff_features = [b_tower - r_tower, b_inhib - r_inhib, b_baron - r_baron, b_dragon - r_dragon, b_kills - r_kills, b_assists - r_assists, b_gold - r_gold]  # Model features

            ####################
            # Display game metadata #
            ####################

            print_mid('TEAMS')
            print_teams()

            for b, r in zip(b_players, r_players):  # Lines 236 and 237
                print_mid(b[0][0].upper() + b[0][1:], b[1], r[1])
            print('\n')

            ################
            # Display game stats #
            ################

            print_mid('STATS')
            print_teams()

            stats_to_print = [['Gold', b_gold, r_gold], ['Kills', b_kills, r_kills], ['Assists', b_assists, r_assists], ['Towers', b_tower, r_tower], ['Dragons', b_dragon, r_dragon], ['Inhibitors', b_inhib, r_inhib], ['Barons', b_baron, r_baron]]
            for stat, b, r in stats_to_print:
                print_mid(stat, b, r)
            print('\n')

            ###################
            # Odds and probabilities #
            ###################

            print_mid('LIVE ODDS')
            print_teams()

            probabilities = model.predict_proba(np.array(diff_features).reshape(1, -1)).round(3)[0][::-1].tolist()  # model returns reversed probs
            odds = [round(1 / max(p, 0.001), 2) for p in probabilities]

            # Shrinking model odds to adjust them to bookies' odds - 1 / (1 + profit margin ~ 7.5 %) ~= 0.93
            adj_odds = [max(round(o * 0.93, 2), 1.001) for o in odds]

            # Bookie odds
            # If the game is last game
            if game_num == game_count:
                book_odds = get_last_odds(driver, b_team, r_team)
            else:
                book_odds = get_odds(driver, b_team, r_team, game_num)

            # Align odds with teams
            if b_team == book_odds[0][0]:
                book_odds = [book_odds[0][1], book_odds[1][1]]
            else:
                book_odds = [book_odds[1][1], book_odds[0][1]]

            # Print odds
            odds_to_print = [['Pred. Probabilities'] + probabilities, ['Adjusted pred. odds'] + adj_odds, ['Bookie Odds'] + book_odds]
            for prob, b, r in odds_to_print:
                print_mid(prob, b, r)

            ########
            # Logging #
            ########

            if LOGGING is True:
                csv_file.writerow([sample_match_id, b_team, r_team] + features + book_odds)

            time.sleep(1)

            # Re-scraping for the next iteration
            # Try/Except is here if the iteration ended too soon (API bug / code running too fast lol)
            game = get_game(game_id)
            new_game_stats = 0
            while new_game_stats == 0:
                try:
                    new_game_stats = game['frames'][-1]
                except KeyError:
                    time.sleep(1)

            game_stats = new_game_stats

            # Move cursor up to print stats in the next iteration
            if game_stats['gameState'] != 'finished':
                move_up(26)
            else:
                del_up(26)

        ##########################################################
        # Sample the next game in the match (match is finished if game is "best of 1") #
        ##########################################################

        while game_states[game_num - 1] == 'inProgress':
            # Match is done within game_stats but not within game_states
            print_mid(f'{b_team} vs {r_team}, game {game_num} done, waiting for API clearence')
            time.sleep(1)

            sample_match = get_match(sample_match_id)
            game_states = [g['state'] for g in sample_match['games']]

            if game_states[game_num - 1] == 'inProgress':
                move_up(3)
            else:
                del_up(3)

    ###########################################
    # If all games are completed but the event hasn't finished #
    ###########################################

    if 'unstarted' not in game_states:
        fin_matches.append(sample_match_id)


###############
# RUN THE SCRIPT #
###############

# App header
print_mid('***LEAGUE OF LEGENDS ESPORTS LIVE***')
print_mid('(Press "Ctrl+C" to exit the program properly)')
print('\n')

try:
    while True:
        if LOGGING is True:
            file = open('Session_' + get_time().replace(':', '_') + '.csv', 'w', newline='')
            csv_file = csv.writer(file, delimiter=',')

            main(csv_file)

        else:
            main()

except KeyboardInterrupt:
    pass

# Cleaning the bets scraping instance/s
os.startfile('cleanup.pyw')

# Closing the log file
if LOGGING is True:
    file.close()

# ADD TO CHECK THE SCRAPER IF THERE ARE MORE GAMES WITH THE SAME TEAM NAME
