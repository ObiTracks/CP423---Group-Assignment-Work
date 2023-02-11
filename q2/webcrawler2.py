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
    name = soup.find("div", id="gsc_prf_in").text
    data["researcher_name"] = name.strip()

    # researcher caption
    caption = soup.find("div", id="gsc_prf_il").text
    data["researcher_caption"] = caption.strip()

    # researcher institution
    institution = soup.find("div", id="gsc_prf_ivh").text
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
    for coauthor in soup.find_all("ul", class_="gsc_a_tr"):
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

def extract_data_2(soup, data):
    # Researcher name
    name = soup.find("div", id="gsc_prf_in").text
    data["researcher_name"] = name.strip()

    # Researcher caption and institute
    # Note: both caption and institute are under the same class, so we need to convert to text, strip it, then split it
    entire_caption = soup.find("div", class_="gsc_prf_il").text.strip().split(',')

    caption = entire_caption[0]
    institution = entire_caption[1]

    data["researcher_caption"] = caption.strip()
    data["researcher_institution"] = institution.strip()

    # Researcher keywords
    keywords = [keyword.text for keyword in soup.find_all("a", class_="gsc_prf_inta")]
    data["researcher_keywords"] = keywords

    # Researcher imageURL
    img = soup.find("img", id="gsc_prf_pup-img")
    data["researcher_imgURL"] = img["src"]

    # Researcher citations
    citation_elements = soup.find_all("td", class_="gsc_rsb_std")
    citations = [citation_element.text.strip() for citation_element in citation_elements]

    data["researcher_citations"] = {"all": citations[0], "Since 2018": citations[1]}
    data["researcher_hindex"] = {"all": citations[2], "Since 2018": citations[3]}
    data["researcher_i10index"] = {"all": citations[4], "Since 2018": citations[5]}

    # Coauthors
    coauthors = []
    for coauthor in soup.find("ul", class_="gsc_rsb_a"):

        # coauthors name
        coauthor_name = coauthor.find('a', {'tabindex': '-1'}).text

        # coauthors title
        coauthor_title = coauthor.find("span", class_="gsc_rsb_a_ext").text

        # coauthors link
        link = "https://scholar.google.ca" + coauthor.find("a", {'tabindex': '-1'})["href"]

        coauthors.append({"coauthor_name": coauthor_name, "coauthor_title": coauthor_title, "coauthor_link": link})
    data["researcher_coauthors"] = coauthors

    papers = []
    paper_count = 0
    for paper in soup.find_all("tr", class_="gsc_a_tr"):
        paper_count += 1

        # Paper title
        title = paper.find("a", class_="gsc_a_at").text

        # Paper authors
        authors = paper.find("div", class_="gs_gray").text

        # Paper journal title
        # Note: \u00c2\u00a0\u00e2\u20ac\u00a6 represents the &nbps tag and i replace it with ellipse
        journal_title = paper.find("div", class_="gs_gray").find_next("div").text.replace("\u00c2\u00a0\u00e2\u20ac\u00a6", "...")

        # Paper citedby
        citedby = paper.find("a", class_="gsc_a_ac gs_ibl").text

        # Paper year
        year = paper.find("span", class_="gsc_a_h gsc_a_hc gs_ibl").text

        papers.append({"paper_count": paper_count, "paper_title": title, "paper_author": authors, "paper_journal": journal_title, "paper_citedby": citedby, "paper_year": year})
    data["researcher_paper"] = papers

if __name__ == '__main__':
    # args stuff
    # parser = argparse.ArgumentParser(description='Web Crawler 2')
    # parser.add_argument('researcherURL', help='The researcherURL to crawl')
    # args = parser.parse_args()
    # researcherURL = args.researcherURL

    researcherURL = "https://scholar.google.ca/citations?hl=en&user=RvyPyJ0AAAAJ"
    #page_data = download_page(researcherURL)
    #print(page_data)

    with open("_Amin Azmoodeh_ - _Google Scholar_.html", "r") as file:
        contents = file.read()
        soup = BeautifulSoup(contents, 'html.parser')
        elements = soup.find_all(class_=True)
        # for element in elements:
            # print(element.get("class"))
    data = {}

    extract_data_2(soup, data)

    # Convert to JSON then print, I'm using an indent to make it more readable
    json_data = json.dumps(data, indent=3)
    print(json_data)