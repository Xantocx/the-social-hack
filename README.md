# the-social-hack

## Introduction

While Jupyter Notebooks have their advantages, they also have their disadvantages. Mostly, they get very hard to read, when more than 1000 lines of code are needed. So we created our own custom Python package, we called NewsTracker. It provides several convenient classes to analyze data on Reddit and Twitter. Then, we built two notebooks, one for Reddit and one for Twitter, which actually perform the analysis, using the package. This way, the files become much more easy to understand.

## Requirements

In order to run this project, you need to install the requirements in the "requirements.txt" file.

## How to Set Up:

To run this project effectively, multiple API keys for Google, Reddit, and Twitter are required. These keys and all other configuration is handled via the NewsTracker/Config.py file, in the class "Configuration". You can initialize such a configuration, by using "Configuration.load_from(filename)". Provide as filename a configuration file. To create such a file, we recommend to use the ".env-template", providing possible fields. Just replace the "???" with the required values. Note that unused fields can be deleted or commented out by the use of "#".

WARNING: We recommend that you do not commit you configuration file to any public repositories, as this might compromise your personal API keys! Consider adding that file to your ".gitignore".