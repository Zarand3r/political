import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool
import matplotlib.pyplot as plt
import ast
import csv
import os
import re
numeric = re.compile(r'[^\d.]+')

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
		html = response.read().decode("latin1")
		return html

def write_to_csv(results, output_file):
	with open(output_file, 'w') as submission_file:
		writer = csv.writer(submission_file, delimiter=',')
		writer.writerows(results)

def get_links(html, filter=None):
	url_list = []
	soup = BeautifulSoup(html, "html.parser")
	for link in soup.findAll('area'):
		url = link.get('href')
		if filter=="state":
			if str(url)[0:5] != filter:
				continue 
		url_list.append(url)
	return url_list

def get_votes(html):
	soup = BeautifulSoup(html, "html.parser")
	try:
		table = soup.find_all('tbody')[0]
	except IndexError:
		return None

	candidates = []
	votes = []
	for row in table.find_all('tr'):
		cell_list = row.find_all("td")
		candidates.append(cell_list[1].text)
		votes.append(numeric.sub('', str(cell_list[-1].text)))
	return list(zip(candidates, votes))

def get_county_fips(url):
	fips = url.split("=")
	fips = fips[2]
	fips = numeric.sub('',fips)
	return fips

def load_general_election(output_file="../data/general.csv"):
	leip_site = "https://uselectionatlas.org/RESULTS/"
	# years_list = np.arange(1940, 2020, step=4)
	years_list = np.arange(2000, 2020, step=4)
	data = []

	# Sliced the lists for testing purposes, so we dont need to go through everything
	for year in years_list:
		us_url = leip_site+ f"national.php?year={year}"
		us_html = get_html(us_url)
		states_url_list = get_links(us_html,filter="state")
		for state_url in states_url_list:
			state_url = leip_site + state_url
			state_html = get_html(state_url)
			counties_url_list = get_links(state_html)
			for county_url in counties_url_list:
				county_url = leip_site + county_url
				county_html = get_html(county_url)
				fips = get_county_fips(county_url)
				election_results = get_votes(county_html)
				for result in election_results:
					data.append([year, fips, result[0], result[1]])
					# append the fips code too. 
					# issue: for early dates, county urls dont exist and instead we go need to go to the state level county datagraph and parse results from the table there
					# update: my ip address got blacklisted for too many queires. going anywhere on the site directs me to Forbidden, when im on browser. Parser still seems to work.
					# solution: use vpn?
	write_to_csv(data, output_file=output_file)
	return data

def load_primary_election(output_file="../data/primary.csv"):
	fips_key = pd.read_csv("fips_key.csv", encoding="latin-1")
	fips_list = fips_key["FIPS"]
	total = len(fips_list)
	years_list = np.arange(2000, 2020, step=4)

	democratic_data = []
	output_file1 = output_file[:-4]+"_democratic.csv"
	for year in years_list:
		error = 0
		for index, fips in enumerate(fips_list): 
			print(f"{index+1} / {total}")
			county_url = f"https://uselectionatlas.org/RESULTS/statesub.php?year={year}&fips={fips}&elect=1&evt=P"
			print(county_url)
			county_html = get_html(county_url)
			election_results = get_votes(county_html)
			if election_results is None:
				error += 1
			else:
				for result in election_results:
					democratic_data.append([year, fips, result[0], result[1]])
			if error > 3:
				break
		write_to_csv(democratic_data, output_file=output_file1)

	republican_data = []
	output_file2 = output_file[:-4]+"_republican.csv"
	for year in years_list:
		error = 0
		for index, fips in enumerate(fips_list): 
			print(f"{index+1} / {total}")
			county_url = f"https://uselectionatlas.org/RESULTS/statesub.php?year={year}&fips={fips}&elect=2&evt=P"
			print(county_url)
			county_html = get_html(county_url)
			election_results = get_votes(county_html)
			if election_results is None:
				error += 1
			else:
				for result in election_results:
					republican_data.append([year, fips, result[0], result[1]])
			if error > 3:
				break
		write_to_csv(republican_data, output_file=output_file2)

	return (democratic_data, republican_data)

# def load_single_primary_election(year, fips_list):
# 	total = len(fips_list)
# 	data = []
# 	for index, fips in enumerate(fips_list): 
# 		print(f"{index+1} / {total}")
# 		county_url = f"https://uselectionatlas.org/RESULTS/statesub.php?year={year}&fips={fips}"
# 		print(county_url)
# 		county_html = get_html(county_url)
# 		election_results = get_votes(county_html)
# 		for result in election_results:
# 			data.append([year, fips, result[0], result[1]])
# 	return data

# def multi_load_primary_election(output_file="../data/primary.csv"):
# 	fips_key = pd.read_csv("fips_key.csv", encoding="latin-1")
# 	fips_list = fips_key["FIPS"][0:10]
# 	years_list = np.arange(2000, 2020, step=4)
# 	primary_results = []

# 	pool = Pool(os.cpu_count()) ## According to TA this will saturate more cores in the hpc?
# 	results = pool.starmap(load_single_primary_election, zip(years_list, repeat(fips_list)))
	
# 	for result in results:
# 		primary_results = primary_results + list(result)

# 	write_to_csv(primary_results, output_file=output_file)
# 	return primary_results


if __name__ == '__main__':
	# load_general_election()
	load_primary_election()
	# multi_load_primary_election()
