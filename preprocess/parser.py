import urllib.request
import ast
import numpy as np
import matplotlib.pyplot as plt


def get_html(url):
	# with urllib.request.urlopen(url) as response:
	# 	html = response.read().decode('ascii')
	# 	return html
	req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	with urllib.request.urlopen(req) as response:
		html = response.read().decode('ascii')
		return html

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"


if __name__ == '__main__':
	url = "https://uselectionatlas.org/RESULTS/national.php?f=1&year=2012&off=0&elect=0"
	# leip = get_html(url)

	opener = AppURLopener()
	with opener.open(url) as response:
		html = response.read().decode('ascii')
		print(html)