# Instructions to run
# python webcrawler1.py <url> --maxdepth <depth> --rewrite <True/False> --verbose <True/False>
# Example (run in terminal)
# python webcrawler1.py https://www.wlu.ca/ --maxdepth 2 --rewrite True --verbose True

import os
import hashlib
import requests
import re
import datetime
import sys
import argparse

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

def extract_links(content):
    # Extract all hyperlinks from the downloaded content
    links = re.findall('<a href="(.*?)"', content)
    return links

def write_file(filename, content):
    # Write the downloaded content to a file
    folder = 'crawled_data'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename)

    with open(file_path, 'w') as f:
        f.write(content)

def log_crawl(log_file, url, depth, response):
    # Append the crawling information to the log file
    with open(log_file, 'a') as f:
        f.write(f"{get_hash(url)},{url},{datetime.datetime.now()},{response.status_code}\n")

def crawl(url, depth, max_depth, rewrite, verbose, log_file):
    # Crawl through a web site
    if depth > max_depth:
        return
    response = download_page(url)
    if response is None:
        return
    if response.status_code != 200:
        log_crawl(log_file, url, depth, response)
        return
    filename = f"{get_hash(url)}.txt"
    if rewrite or not file_exists(filename):
        content = response.text
        write_file(filename, content)
        log_crawl(log_file, url, depth, response)
        if verbose:
            print(f"{url},{depth}")
        links = extract_links(content)
        for link in links:
            crawl(link, depth + 1, max_depth, rewrite, verbose, log_file)

def file_exists(filename):
    # Check if a file exists
    try:
        with open(filename):
            return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description='Web Crawler')
    parser.add_argument('initialURL', help='The initial URL to crawl')
    parser.add_argument('--maxdepth', type=int, default=1, help='The maximum depth to crawl')
    parser.add_argument('--rewrite', type=bool, default=False, help='Whether to rewrite the file if it exists')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print URL and depth')
    args = parser.parse_args()
    initial_url = args.initialURL
    max_depth = args.maxdepth
    rewrite = args.rewrite
    verbose = args.verbose
    log_file = 'crawler1.log'
    crawl(initial_url, 1, max_depth, rewrite, verbose, log_file)

if __name__ == '__main__':
    main()