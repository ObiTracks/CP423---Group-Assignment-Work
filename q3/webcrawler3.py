import hashlib
import re
import requests
import matplotlib.pyplot as plt
import numpy as np
import sys


def webcrawler3(url):
    content_of_html = requests.get(url).text
    hash_value = hashlib.sha256(url.encode()).hexdigest()
    content_of_html = re.sub(r'<[^>]*>', '1', content_of_html)
    content_of_html = re.sub(r'\b\w+\b', '0', content_of_html)

    final_content = content_of_html.split()

    fMax, iBest, jBest = -1, -1, -1

    for i in range(len(final_content)):
        for j in range(i + 1, len(final_content)):
            f = final_content[i:j].count('0')
            if f > fMax:
                fMax = f
                iBest = i
                jBest = j

    with open(f"{hash_value}.txt", 'w') as f:
        f.write(' '.join(final_content[iBest:jBest]))

    a, b = np.meshgrid(np.arange(len(final_content)), np.arange(len(final_content)))
    c = np.zeros_like(a)
    for i in range(len(final_content)):
        for j in range(i + 1, len(final_content)):
            c[i][j] = final_content[i:j].count('0')
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot_surface(a, b, c)
    plt.show()


if __name__ == '__main__':
    url = sys.argv[1]
    webcrawler3(url)
