import requests


def download_page(url):
    # Download the content of the URL
    try:
        response = requests.get(url)
    except:
        return None
    return response


if __name__ == '__main__':
    print("wtf")
    researcherURL = "https://scholar.google.ca/citations?hl=en&user=RvyPyJ0AAAAJ"

    with open("_Amin Azmoodeh_ - _Google Scholar_.html", "r") as file:
        contents = file.read()
        print(contents)
   # page_data = download_page(researcherURL)
   # print(page_data)