#########################################################################################
SETUP
#########################################################################################

-----------------------------------------------------------------------------------------
1) Virtualenv 

Set up a virtual environment.  If you do not have virtualenv, you can get it 
via pip.  Execute "pip install virtualenv" to get it.  Installation should work with 
pip2 or pip3.

-----------------------------------------------------------------------------------------
2) Python 3.6

Install python 3.6 as well.  If don't have homebrew, get it using

```
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Then, add `export PATH=/usr/local/bin:/usr/local/sbin:$PATH` to your `.profile` or other 
bash config file.  Finally, execute

`brew install python3`

to install python 3.6.

-----------------------------------------------------------------------------------------
3) Establish virtual environment

Figure out where your python3.6 is located by running `which python3.6`.  If 
nothing is outputted, go back to step 2.  The path is usuall `/usr/local/bin/python3.6` or 
/usr/bin/python3.6`.

Run `virtualenv --python=<path-to-python-3.6> venv`, then `source venv/bin/activate` 
to start up the env.

Finally, run `pip install -r requirements/36_requirements.txt` from the main directory 
to install depedencies--namely, tensorflow.

#########################################################################################
DATA GENERATION
#########################################################################################

Run `python zener_generator.py <folder-name> <num-examples>`.  It's the same arguments 
as in Homework 2.

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
