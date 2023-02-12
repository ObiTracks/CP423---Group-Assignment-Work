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

def extract_paper_data(papers, paper):

    # Paper title
    title = paper.find("a", class_="gsc_oci_title_link").text

    entries = paper.find("div", id="gsc_oci_table").find_all("div", class_="gs_scl")

    authors = None
    publication_date = None
    citedby = None
    journal = None

    for entry in entries:
        field = entry.find("div", class_="gsc_oci_field").text.strip()
        value = entry.find("div", class_="gsc_oci_value").text

        if field == "Authors":
            authors = value
        elif field == "Journal" or field == "Source":
            journal = value
        elif field == "Publication date":
            publication_date = value.split("/")[0]
        elif field == "Source":
            title = value
        elif field == "Total citations":
            citedby = entry.find("div", class_="gsc_oci_value").find("a").text.split(" ")[2]

    papers.append({"paper_title": title, "paper_authors": authors, "paper_journal": journal, "paper_citedby": citedby,"paper_year": publication_date})
    return papers

def extract_data(soup, data):
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

        # Coauthors name
        coauthor_name = coauthor.find('a', {'tabindex': '-1'}).text

        # Coauthors title
        coauthor_title = coauthor.find("span", class_="gsc_rsb_a_ext").text

        # Coauthors link
        link = "https://scholar.google.ca" + coauthor.find("a", {'tabindex': '-1'})["href"]

        coauthors.append({"coauthor_name": coauthor_name, "coauthor_title": coauthor_title, "coauthor_link": link})
    data["researcher_coauthors"] = coauthors

    papers = []

    for paper in soup.find("tbody", id="gsc_a_b"):
        paper_link = "https://scholar.google.ca" + paper.find("a", class_="gsc_a_at")["href"]

        paper_page_data = download_page(paper_link)
        paper_soup = BeautifulSoup(paper_page_data.content, "html.parser")

        papers = extract_paper_data(papers, paper_soup)

    data["researcher_paper"] = papers

if __name__ == '__main__':
    # args stuff
    parser = argparse.ArgumentParser(description='Web Crawler 2')
    parser.add_argument('researcherURL', help='The researcherURL to crawl')
    args = parser.parse_args()
    researcherURL = args.researcherURL

    researcher_page_data = download_page(researcherURL)
    researcher_soup = BeautifulSoup(researcher_page_data.content, "html.parser")

    data = {}
    extract_data(researcher_soup, data)

    # Save JSON to a file
    with open("webcrawler2.json", "w") as file:
        json.dump(data, file)