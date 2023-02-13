# Instructions to run
# python webcrawler2.py <researcherURL>

# Example (run in terminal)
# python webcrawler2.py 

import argparse
import hashlib
import json

import requests
from bs4 import BeautifulSoup


def get_hash(url):
    # Calculate the hash value for the URL
    hash = hashlib.sha256()
    hash.update(url.encode())
    return hash.hexdigest()


def create_hash_file(url, data):
    # Create a hash of the url as the filename, then save the webpage data to that file
    filename = f"{get_hash(url)}.txt"
    with open(filename, "w") as file:
        file.write(data)


def create_json_file(url, data):
    # Create a hash of the url as the filename, then save the json data to that file
    filename = f"{get_hash(url)}.json"
    with open(filename, "w") as file:
        json.dump(data, file)


def download_page(url):
    # Download page, throw an error if page isn't downloaded properly

    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code != 200:
            raise Exception(f"Request failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error while downloading page: {e}")
        return None
    return response


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

    # All of them are in the same element, so I stripped it in sections and then just assign the corresponding section
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

    # Papers are their own webpage, so we must do extra to get their data
    papers = []
    # For every paper in the list
    for paper in soup.find("tbody", id="gsc_a_b"):
        # Get the new link for the paper
        paper_link = "https://scholar.google.ca" + paper.find("a", class_="gsc_a_at")["href"]

        # Download that data ( Not asked to save it )
        paper_page_data = download_page(paper_link)

        # Convert to soup
        paper_soup = BeautifulSoup(paper_page_data.content, "html.parser")

        # Extract the data using the function
        papers = extract_paper_data(papers, paper_soup)

    data["researcher_paper"] = papers


def extract_paper_data(papers, paper):
    # Paper title
    title = paper.find("a", class_="gsc_oci_title_link").text

    # Paper objects in the list
    entries = paper.find("div", id="gsc_oci_table").find_all("div", class_="gs_scl")

    # These are the data we are looking for
    authors = None
    publication_date = None
    citedby = None
    journal = None

    # Check every field for one of the data points, if it matches we know the corresponding value is our data we need
    for entry in entries:
        field = entry.find("div", class_="gsc_oci_field").text.strip()
        value = entry.find("div", class_="gsc_oci_value").text

        if field == "Authors":
            authors = value

        # Note: Some papers listed either Journal or Source but I did not encounter both at the same time
        #       Therefore I am writing it to use either as I think its an alternative
        elif field == "Journal" or field == "Source":
            journal = value
        elif field == "Publication date":
            publication_date = value.split("/")[0]
        elif field == "Source":
            title = value
        elif field == "Total citations":
            citedby = entry.find("div", class_="gsc_oci_value").find("a").text.split(" ")[2]

    # Once all the data is found, append it to the paper array and repeat for all the papers in the list
    papers.append({"paper_title": title, "paper_authors": authors, "paper_journal": journal, "paper_citedby": citedby,
                   "paper_year": publication_date})
    return papers


if __name__ == '__main__':
    # args stuff
    parser = argparse.ArgumentParser(description='Web Crawler 2')
    parser.add_argument('researcherURL', help='The researcherURL to crawl')
    args = parser.parse_args()
    researcherURL = args.researcherURL

    # Download the page
    researcher_page_data = download_page(researcherURL)

    # Save the page to text file, using hash as name
    create_hash_file(researcherURL, researcher_page_data)

    # Convert the response to bs4 soup
    researcher_soup = BeautifulSoup(researcher_page_data.content, "html.parser")

    # Create empty dict
    data = {}

    # Fill the dict with the json data
    extract_data(researcher_soup, data)

    # Save JSON to a file
    create_json_file(researcherURL, data)
