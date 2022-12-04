# the-social-hack

## How to Set Up:

To run this project effectively, multiple API keys for Google, Reddit, and Twitter are required. These keys and all other configuration is handled via the NewsTracker/Config.py file, in the class "Configuration". You can initialize such a configuration, by using "Configuration.load_from(filename)". Provide as filename a configuration file. To create such a file, we recommend to use the ".env-template", providing possible fields. Just replace the "???" with the required values. Note that unused fields can be deleted or commented out by the use of "#".

WARNING: We recommend that you do not commit you configuration file to any public repositories, as this might compromise your personal API keys! Concider adding that file to your ".gitignore".