# Exercise 02.4 :: AC Benchmanrk
This part of the project is designed to run SMAC on a cluster to get
an algorithm configuration benchmark.

## Installation
First of all, install Python3.6 and its dev files:
```bash
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.6
$ sudo apt-get install python3.6-dev
$ sudo apt-get install python3.6-tk
```

Install `swig3.0`:
```bash
$ sudo apt-get remove swig
$ sudo apt-get install swig3.0
$ sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
```

Create a virtualenv:
```bash
$ virtualenv -p python3.6 venv
$ source venv/bin/activate
```

Install required packages:
```bash
$ pip install Cython
$ pip install pyrfr==0.8.0 --no-cache
$ pip install numpy scipy sklearn matplotlib
$ pip install git+https://github.com/openml/openml-python.git
```

Install SMAC3 from my own repository to avoid the bug with `pyrfr`:
```bash
$ pip install git+https://github.com/angellandros/SMAC3.git@development
```