import urllib.request
import ast
import numpy as np
import matplotlib.pyplot as plt

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

def get_html(url):
	# with urllib.request.urlopen(url) as response:
	# 	html = response.read().decode('ascii')
	# 	return html

	# req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	# with urllib.request.urlopen(req) as response:
	# 	html = response.read().decode('ascii')
	# 	return html

	opener = AppURLopener()
	with opener.open(url) as response:
		html = response.read().decode('ascii')
		return html


if __name__ == '__main__':
	year = 2012
	url = f"https://uselectionatlas.org/RESULTS/national.php?year={year}"
	leip = get_html(url)
	print(leip)
