from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

import time
import os


#############
# PARAMETERS #
#############

options = Options()
options.headless = False
profile = webdriver.FirefoxProfile()
profile.set_preference("permissions.default.image", 2)


###########
# FUNCTIONS #
###########

def await_element(driver, xpath):
    element_present = EC.presence_of_element_located((By.XPATH, xpath))
    WebDriverWait(driver, 30).until(element_present)


def init_driver(options=options, profile=profile):
    driver = webdriver.Firefox(firefox_profile=profile, options=options, executable_path='geckodriver.exe', service_log_path=os.devnull)
    driver.maximize_window()

    driver.get('https://www.lsbet.com/live')

    # Cookies popup
    cookies_button = '//button[@class="c012 c0180"]'
    await_element(driver, cookies_button)
    time.sleep(2.5)
    driver.find_element_by_xpath(cookies_button).click()

    esports_button = '//div[./h4/span[contains(., "E-Sports")]]'
    await_element(driver, esports_button)
    driver.find_element_by_xpath(esports_button).click()

    # Scroll down so all esports gams come into view
    driver.execute_script('window.scrollTo(0, 450)')

    return driver


def get_odds(driver, team1, team2, game_num):
    match_button = '//a[contains(., "' + team1 + ' - ' + team2 + '")]|//a[contains(., "' + team2 + ' - ' + team1 + '")]|//a[contains(., "' + team1 + '")]|//a[contains(., "' + team2 + '")]'
    await_element(driver, match_button)
    driver.find_element_by_xpath(match_button).click()

    map_div = '//div[@class="osg-market"][./h3/span[contains(., "Map ' + str(game_num) + ' Winner - Live Game")]]'
    try:
        await_element(driver, map_div)
    except:
        print('Game number: ', game_num)
        raise AwaitError
    curr_div = driver.find_element_by_xpath(map_div)

    odds = curr_div.text.split('\n')[1:]
    try:
        odds = [[odds[0], odds[1]], [odds[2], odds[3]]]
    except:
        odds = [[team1, '-'], [team2, '-']]

    return odds


# If the game is the last game in series (because usually we are looking at odds for a particular game in the match but if the series is extended to the last match
# in the series, the winner of the last match is the same thing as the winner of the last game, so in a best of 3, if a 3rd game is played, there won't be Map 3 winner
# but match winner)
def get_last_odds(driver, team1, team2):
    match_button = '//a[contains(., "' + team1 + ' - ' + team2 + '")]|//a[contains(., "' + team2 + ' - ' + team1 + '")]|//a[contains(., "' + team1 + '")]|//a[contains(., "' + team2 + '")]'
    await_element(driver, match_button)
    driver.find_element_by_xpath(match_button).click()

    match_div = '//div[@class="osg-market"][./h3/span[contains(., "Match Up Winner - Live Game")]]'
    await_element(driver, match_div)
    curr_div = driver.find_element_by_xpath(match_div)

    odds = curr_div.text.split('\n')[1:]
    try:
        odds = [[odds[0], odds[1]], [odds[2], odds[3]]]
    except:
        odds = [[team1, '-'], [team2, '-']]

    return odds
