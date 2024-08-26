import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime


def generateData(url,symbol):
    # Headers to simulate a request from a web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # Send a request to fetch the HTML content of the page
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        logging.info("Successfully fetched the webpage content")
    else:
        logging.error(f"Failed to fetch the webpage content, status code: {response.status_code}")
        return
        response.raise_for_status()


    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table containing the news data
    news_table = soup.find("table", {"id": "news-table"})
    if news_table:
        logging.info("Found the news table")
    else:
        logging.error("News table not found")
        return
        raise ValueError("News table not found")


    # List to store the extracted data
    data = []

    # Find all rows in the table
    rows = news_table.find_all("tr", {"class": "cursor-pointer has-label"}) if news_table else []


    for idx, row in enumerate(rows):
        try:
            time_element = row.find("td", {"align": "right"})
            link_element = row.find("a", {"class": "tab-link-news"})

            if not time_element or not link_element:
                logging.warning(f"Skipping row {idx}: Missing required elements")
                continue

            time = time_element.text.strip()
            headline = link_element.text.strip()
            symbol = symbol

            data.append([time, headline,symbol])
            logging.debug(f"Row {idx}: time = {time}, headline = {headline},symbol={symbol}")

        except Exception as e:
            logging.error(f"Error processing row {idx}: {e}")

    return data

# method to convert hh:mm to mmm-dd-yy
def fill_date(times):
    filled_times = []
    next_date = None

    for i in range(len(times) - 1, -1, -1):
        if '-' in times[i]:  # Full date-time string
            filled_times.append(times[i])
            next_date = datetime.strptime(times[i], '%b-%d-%y %I:%M %p')
        elif 'AM' in times[i] or 'PM' in times[i]:  # Time string with AM/PM
            filled_times.append(times[i])
            next_date = datetime.strptime(times[i], '%I:%M %p')
        else:  # Only date string
            if next_date:
                filled_times.append(next_date.strftime('%b-%d-%y') + ' ' + times[i])
            else:
                filled_times.append(times[i])

    return filled_times[::-1]




def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # nifty_10_symbols = ["HDB"]
    nifty_10_symbols = ["HDB","IBN","INFY","MMYT","RDY","RNW","SIFY","WIT","WNS","YTRA","ZCAR"]

    data = []
    dfList = []

    for symbol in nifty_10_symbols:
        url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
        dataVal = generateData(url,symbol)
        if dataVal is not None:
            tempDf = pd.DataFrame(dataVal, columns=["time", "headline","symbol"])
            #pd.concat([df,tempDf],ignore_index=True)
            dfList.append(tempDf)
            # Display the DataFrame
        else:
            logging.info("No data to create a DataFrame")

    df = pd.concat(dfList)

    # df['dateTime'] = fill_date(df['time'])
    df.to_csv("finviz-2-consolidated.csv")
    print("done!")

main()