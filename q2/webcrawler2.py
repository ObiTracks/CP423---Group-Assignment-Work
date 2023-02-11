import argparse
import hashlib
import json
import sys

import requests
from bs4 import BeautifulSoup


def get_hash(url):
    # Calculate the hash value for the URL
    hash = hashlib.sha256()
    hash.update(url.encode())
    return hash.hexdigest()


def download_page(url):
    # Download the content of the URL
    try:
        response = requests.get(url)
    except:
        return None
    return response


def extract_data(soup, data):

    # researcher name
    name = soup.find("div", class_="gsc_prf_in").text
    data["researcher_name"] = name.strip()

    # researcher caption
    caption = soup.find("div", class_="gsc_prf_il").text
    data["researcher_caption"] = caption.strip()

    # researcher institution
    institution = soup.find("div", class_="gsc_prf_ivh").text
    data["researcher_institution"] = institution.strip()

    # researcher keywords
    keywords = [keyword.text for keyword in soup.find_all("a", class_="gsc_prf_inta")]
    data["researcher_keywords"] = keywords

    # researcher imageURL
    img = soup.find("img", class_="gsc_prf_pct")
    data["researcher_imgURL"] = img["src"]

    # researcher citations
    citations = soup.find("td", class_="gsc_rsb_std").text
    data["researcher_citations"] = {"all": citations.strip()}

    # researcher h-index
    hindex = soup.find("td", class_="gsc_rsb_hindex").text
    data["researcher_hindex"] = {"all": hindex.strip()}

    # researcher i10-index
    i10index = soup.find("td", class_="gsc_rsb_i10index").text
    data["researcher_i10index"] = {"all": i10index.strip()}

    # researcher coauthors
    coauthors = []
    for coauthor in soup.find_all("tr", class_="gsc_a_tr"):
        # coauthors name
        name = coauthor.find("a", class_="gsc_a_at").text

        # coauthors title
        title = coauthor.find("div", class_="gsc_a_t").text

        # coauthors link
        link = coauthor.find("a", class_="gsc_a_at")["href"]

        coauthors.append({"coauthor_name": name, "coauthor_title": title, "coauthor_link": link})
    data["researcher_coauthors"] = coauthors

    # researcher papers
    papers = []
    for paper in soup.find_all("tr", class_="gsc_a_tr"):
        # paper title
        title = paper.find("a", class_="gsc_a_at").text

        # paper authors
        authors = paper.find("div", class_="gs_gray").text

        # paper journal
        journal = paper.find("div", class_="gs_gray").find_next("div").text

        # paper citedby
        citedby = paper.find("div", class_="gsc_a_ac gs_ibl").text

        # paper citedby
        year = paper.find("div", class_="gsc_a_h gsc_a_hc gs_ibl").text


if __name__ == '__main__':
    # args stuff
    # parser = argparse.ArgumentParser(description='Web Crawler 2')
    # parser.add_argument('researcherURL', help='The researcherURL to crawl')
    # args = parser.parse_args()
    # researcherURL = args.researcherURL

    researcherURL = "https://scholar.google.ca/citations?hl=en&user=RvyPyJ0AAAAJ"
    page_data = download_page(researcherURL)
    print(page_data)


    soup = BeautifulSoup(page_data.content, 'html.parser')
    print(soup)
    data = {}
    # extract_data(soup, data)

    json_data = json.dumps(data)
    print(json_data)