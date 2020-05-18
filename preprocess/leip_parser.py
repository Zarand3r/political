import urllib.request
from bs4 import BeautifulSoup
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
		html = response.read().decode()
		return html

# def fetch_data:
# 	return

def get_links(html):
	url_list = []
	soup = BeautifulSoup(html, "html.parser")
	for link in soup.findAll('area'):
	    url_list.append(link.get('href'))
	return url_list

def get_votes(html):
	print(html)
	soup = BeautifulSoup(html, "html.parser")
	table = soup.find_all('tbody')[0]
	print(table)
	for row in table.find_all('tr'):
		print(row)
	return row

def load_general_election():
	leip_site = "https://uselectionatlas.org/RESULTS/"
	# years_list = np.arange(1940, 2020, step=4)
	years_list = np.arange(1980, 2020, step=4)
	data = []

	# Sliced the lists for testing purposes, so we dont need to go through everything
	for year in years_list[0:1]:
		us_url = leip_site+ f"national.php?year={year}"
		us_html = get_html(us_url)
		states_url_list = get_links(us_html)
		for state_url in states_url_list[0:1]:
			state_url = leip_site + state_url
			state_html = get_html(state_url)
			counties_url_list = get_links(state_html)
			for county_url in counties_url_list[0:1]:
				county_url = leip_site + county_url
				county_html = get_html(county_url)
				votes = get_votes(county_html)
				# This currently just prints out the rows in the county html corresponding canddiates and votes
				# Next step is to append [county fips, year, candidate, vote] to the data variable at this bottom loop
				# Then, we can return the data list as a list of lists, and write it into a csv

def load_primary_election():
	# replace the fips at the end for a quick way to get votes for each county
	#https://uselectionatlas.org/RESULTS/statesub.php?year=2012&fips=36061

	# Alternatively, start at this url and do the same nested for loop as for the general election
	# Beware that getting the urls at each level will take an additional step
	# The href are have url parameters formatted with semicolon, wont work as links, need to reformat
	#https://uselectionatlas.org/RESULTS/national.php?f=1&year=2020&elect=1




if __name__ == '__main__':
	load_data()
