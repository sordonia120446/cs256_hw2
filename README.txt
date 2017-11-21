#########################################################################################
SETUP
#########################################################################################

1) Virtualenv 

Set up a virtual environment.  If you do not have virtualenv, you can get it 
via pip.  Execute "pip install virtualenv" to get it. 

2) Python 3.6

Install python 3.6 as well.  If don't have homebrew, get it using

```
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Then, add `export PATH=/usr/local/bin:/usr/local/sbin:$PATH` to your `.profile` or other 
bash config file.  Finally, execute

`brew install python3`

3) Establish virtual environment

Figure out where your python3.6 is located by running `which python3.6`.  If 
nothing is outputted, go back to step 2.

Run `virtualenv --python=<path-to-python-3.6> venv`, then `source venv/bin/activate` 
to start up the env.

Finally, run `pip install -r requirements/36_requirements.txt` from the main directory 
to install depedencies--namely, tensorflow.

#########################################################################################
DATA GENERATION
#########################################################################################

Since homework 2 was done on 2.7, you'll need to use that for the zener card generation.  It 
requires numpy and Pillow.  Run `python zener_generator.py <folder-name> <num-examples>`.


#########################################################################################
EXECUTION
#########################################################################################

Run the normal command-line arguments as outlined in homework 4.

If this or anything else in this process doesn't work or is confusing, please let us know asap.

#########################################################################################
Project Group
#########################################################################################

Jason Chee
Nick Mattheus
Samuel Ordonia

