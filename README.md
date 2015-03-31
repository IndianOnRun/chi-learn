# chi-learn

## In One Sentence
Given records of crimes committed in Chicago from 2001 to day x, can we predict how many violent crimes will be committed on day x + 1?

## Setup Instructions
### Easy way: use global environment.
Run with Python 3.4. Make sure you have pandas and scikit-learn installed. Ipython will be nice for tinkering in notebooks.

### Tricky but "better" way: use a virtual environment.
+ There are some prereqs that might not come bundled with Python 3.4. I needed to `sudo apt-get install python3.4-dev` to install certain packages.
+ Also, I needed to `sudo apt-get install gfortran libopenblas-dev liblapack-dev` to get linear algebra dependencies for numpy. Does installing this with the system's package manager defeat the purpose of a virtual environment? A little, but FORTRAN libraries probably don't change too often.
+ Make a new virtual environment with Python 3.4 like so: `virtualenv --no-site-packages -p /usr/bin/python3.4 env`
+ Activate your virtual environment: `source env/bin/activate`
+ Clone this repo.
+ From the chi-learn directory, install dependencies in your virtual environment with pip: `pip install -r requirements.txt`
+ You may want to download the most recent full dataset [from Chicago's data portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2). Choose "export as CSV" and save it in /data.

## Dependencies
We're using pandas and scikit-learn. Check out requirements.txt for specific versions.

## Origins
Chi-learn is Will Engler, Kevin Minkus, and Joel Roggeman's term project for Pitt CS 1675 (Machine Learning) Spring 2015 with Dr. Rebecca Hwa.