Ps120 Project

The county-level election data from 2000 to 2016 is from the MIT Election Lab
- Reformat to make the columns fips code, candidate, vote percentage for that year. Sort from left to right by year

Make a nested parser for the Leip data start here https://uselectionatlas.org/RESULTS/national.php?year=2004&f=0&off=0&elect=0 
(view-source:https://uselectionatlas.org/RESULTS/national.php?year=2004&f=0&off=0&elect=0)
- Vary the years in the url parameter
- Parse out all the state urls https://uselectionatlas.org/RESULTS/state.php?year=2004&fips=1&f=0&off=0&elect=0 or just vary the year fips url parameter
- Parse out all the county urls https://uselectionatlas.org/RESULTS/statesub.php?year=2004&fips=1091&f=0&off=0&elect=0 or just vary the fips parameter
- Parse out the votes from the table of the source of each county view-source:https://uselectionatlas.org/RESULTS/statesub.php?year=2004&fips=1091&f=0&off=0&elect=0

SVD Implementation
- Documentation: https://surprise.readthedocs.io/en/stable/getting_started.html

Prototype uses the surprise library SVD (try the SVDpp too)
For time SVD++, take inspiration from here https://github.com/leo1023/timeSVDpp_librec_based/blob/master/timeSVDpp.py