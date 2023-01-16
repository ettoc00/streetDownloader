from pathlib import Path

from streetdownloader.common import Location


def user_input():
    import chromedriver_autoinstaller
    from selenium.webdriver import Chrome
    from selenium.webdriver.common.by import By
    from tkinter.filedialog import askdirectory
    chromedriver_autoinstaller.install()

    print('Folder:')
    folder = Path(askdirectory(title='Select Folder'))
    print(folder)

    chrome_driver = Chrome()
    chrome_driver.get('https://www.google.com/maps')
    if _t := chrome_driver.find_elements(By.CSS_SELECTOR, 'form button'):
        _t[0].click()

    def get_coords(prompt):
        input(prompt)
        d = str(chrome_driver.current_url).split('@', 1)[-1].split(',', 2)
        if len(d) < 2:
            return 0
        x, y, *_ = d
        loc = Location(float(x), float(y))
        print(loc)
        return loc

    print('Please navigate to your desired locations on the Google Maps page that has been opened in your browser.')
    print('Once you have selected your location, press enter on the terminal to continue.')
    l1 = get_coords('Location 1:')
    l2 = get_coords('Location 2:')

    chrome_driver.quit()
    return folder, l1, l2
