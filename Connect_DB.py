'''
Created on 8 Jul 2017

@author: EByrn
'''

#import sqlalchemy
from sqlalchemy import create_engine

# To create engine - adapted from previous projects and http://docs.sqlalchemy.org/en/latest/core/engines.html
def connect_db(URI, PORT, DB, USER, password):
    ''' Function to connect to the database '''
    try:
        fh = open(password)
        PASSWORD = fh.readline().strip()
        engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(USER, PASSWORD, URI, PORT, DB), echo=True)
        return engine
    except Exception as e:
        print("Error Type: ", type(e))
        print("Error Details: ", e)