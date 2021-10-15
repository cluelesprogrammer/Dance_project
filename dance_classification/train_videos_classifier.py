import configparser

config = configparser.ConfigParser()
config.read('videos_config.ini')
print(config['DATASETINFO'])
