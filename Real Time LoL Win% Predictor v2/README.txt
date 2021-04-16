--INTRODUCTION--

Enjoy! :)


*easydatascience is a custom library with which I operate faster. I ignored it in some cases because even if it is 
found in some directories for conviniance, it also might not be used in all of the directories where it is located.

--------------------------------------------------------------------------------------------------------------------
--IMPORTANT NOTES--

- The API key used in the 2 initial scrapers is required to be manually obtained on the Riot's API website (you will
  also need a Riot account). The key has to be regenerated every 24 hours.

- Things that could've been done better but require much more time:
    - Beta Regression
    - Narrowing the problem down to just close cases (cases where Gold Difference distributions on y = 1 and
      y = 0 overlap)
    - Getting the train data from the same batch as the test data
    - Getting more test data
    - Introduce "Late Game Champion influence" feature/s

* test_set.csv - Some things to note about this file:
    - The script doesn't have a way to deal with the final outcome of a game (y) so the final outcomes have been
    manually added
    - My dumb ass kept saving the MatchID column as numeric instead of as a string so it got corrupt in the process.
    That is why it is present before cleaning but removed afterwards.
    - Arbitrary GameIDs have been manually added instead of MatchIDs
--------------------------------------------------------------------------------------------------------------------

--DIRECTORIES--

1. Scraper
     |
     |--- GameIDs
            |
            |--- scraper_matches.py - Scraper of game IDs (automatic). There is a 20 per second, 100 per 2 minutes
				      request limit you can make to the Riot's API, that is the reason for the 
				      ".sleep()" commands.
     |
     |--- GameStats
            |
            |--- matches_... .csv - GameIDs obtained from scraper_matches.py
            |
            |--- scraper_games.py - Scraper of game stats that uses the previously obtained game IDs. The scraper 
				    was split in 2 for readability purposes and so it could be easier to work with.

2. Models
     |
     |--- Merge and Clean
            |
            |--- games_... .csv - Games' stats scraped from GameStats
            |
            |--- Merge and Clean.ipynb - Mergin and cleaning all the csvs so that they could be used in future
                                         analysis. I tried my best to explain some of the less comperhensible code
                                         with comments within the file but basically, the messy "role rearranging"
                                         part is taking chunks of rows within the each row and rearanging them.
     |
     |--- data.csv - Final data ready for use.
     |
     |--- Models.ipynb - EDA and predictive models. The whole process and important notes that are associated with
                                                    are contained in the markdown cells within the file.

3. Test
    |
    |--- Live Game Monitoring and Scraping Script
           |
           |--- Logs - Folder used to store csvs that've been scraped
           |
           |--- Merge and Clean
                  |
                  |--- Session... .csv - Concated log files (done in a throwaway .ipynb notebook)
                  |
                  |--- Merge and Clean.ipynb - Processing the data so it is ready to be used as test data
           |
           |--- geckodriver.exe - File that makes it possible for the script to run Firefox with python
           |
           |--- lr_pipeline.pkl - Pipline used to transform and model the data
           |
           |--- Riot Script README.txt - Readme that contains all information about running the script (app)
           |
           |--- SCRIPT -
                  |
                  |--- script.py - Executable .py file (app) that uses all other .py files
                  |
                  |--- parameters.py - Changeable parameters that the script uses (LINE_W / line width of 122 is
                                       optimal for the size of comand prompt that opens up on default, on my machine
                                       at least; I can't guarantee that the script is going to work flawlessly for 
                                       the servers that aren't already listed, use at your own risk!)
                  |
                  |--- script_functions.py - Scraping and other functions that the script uses
                  |
                  |--- live_bets.py - Parameters and functions used for scraping live betting odds
                  |
                  |--- cleanup.py - Clears junk made by the script (more on it in the comments within the file)
    |
    |--- test_set.csv - Test set produced derived from Live Game Script (look at the notes for more info*)
    |
    |--- lr_pipeline.pkl - Pipline used to transform and model the data
    |
    |--- Test.ipynb - Notebook used to test the model/s

Presentation.pptx - Presentation used to summarize the entire project and preset findings

README.txt
--------------------------------------------------------------------------------------------------------------------
