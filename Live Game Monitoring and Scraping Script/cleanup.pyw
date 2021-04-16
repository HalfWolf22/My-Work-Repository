# As addresed in the following issues:
#
# https://stackoverflow.com/questions/60025735/selenium-doesnt-quit-on-keyboardinterrupt-python
# https://github.com/SeleniumHQ/selenium/issues/6826
#
# The bug where Selenium WebDriver becomes unresponsive when a
# KeyboardInterrupt occurs is still persistant and hasn't been resolved yet.
# This is the most clean way to deal with the issue of the Selenium tabs
# being open even after the main script has been closed.
#
# NOTE: This terminates ALL the Firefox processes.

import psutil

process = "firefox.exe"

for p in psutil.process_iter():
    if p.name() == process:
        p.kill()
