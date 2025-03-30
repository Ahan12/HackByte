import requests

# URL to the CSV file served by the device with IP 192.168.80.166
url = "http://192.168.80.166:5001/download_csv"

# Make the GET request to download the CSV file
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Write the content of the response (CSV data) to a local file
    with open('downloaded_file.csv', 'wb') as file:
        file.write(response.content)
    print("File downloaded and saved as 'downloaded_file.csv'")
else:
    print("Failed to fetch CSV. Status code:", response.status_code)
