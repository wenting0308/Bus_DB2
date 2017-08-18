import csv
import logging
import pandas as pd
from sqlalchemy import create_engine

# To create engine - adapted from previous projects and http://docs.sqlalchemy.org/en/latest/core/engines.html
def connect_db(URI, PORT, DB, USER, password):
    ''' Function to connect to the database '''
    try:
        PASSWORD = password
        engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(USER, PASSWORD, URI, PORT, DB), echo=True)
        return engine
    except Exception as e:
        print("Error Type: ", type(e))
        print("Error Details: ", e)

def create_tables():
    """ Create tables in database. """
    
    engine = connect_db('127.0.0.1', '3306', 'DBus', 'root', 'team1010')
    
    try:
        
        ## ---------------------- Table1: routes ---------------------- ##
        stm = """DROP TABLE IF EXISTS `DBus`.`routes`;"""
        engine.execute(stm)
        
        stm = """CREATE TABLE IF NOT EXISTS `DBus`.`routes` (
                `route_short_name` VARCHAR(10) NOT NULL,
                `trip_headsign` VARCHAR(100) NOT NULL,
                PRIMARY KEY (`route_short_name`, `trip_headsign`));"""
        engine.execute(stm)        
        
        ## ---------------------- Table2: routes_stops ---------------------- ##
        stm = """DROP TABLE IF EXISTS `DBus`.`routes_stops`;"""
        engine.execute(stm)
        
        stm = """CREATE TABLE IF NOT EXISTS `DBus`.`routes_stops` (
                `route_short_name` VARCHAR(10) NOT NULL,
                `trip_headsign` VARCHAR(100) NOT NULL,
                `stop_sequence` INT NOT NULL,
                `stop_id` VARCHAR(10) NOT NULL,
                `stop_name` VARCHAR(50) NOT NULL);""" 
                
        engine.execute(stm)
        
        ## ---------------------- Table3: routes_stops_service_days ---------------------- ##
        stm = """DROP TABLE IF EXISTS `DBus`.`routes_stops_service_days`;"""
        engine.execute(stm)
        
        # 0807 revised: Cause the table is too big, change table structure.
        stm = """CREATE TABLE IF NOT EXISTS `DBus`.`routes_stops_service_days` (
              `trip_id` VARCHAR(30) NOT NULL,
              `route_short_name` VARCHAR(10) NOT NULL,
              `trip_headsign` VARCHAR(100) NOT NULL,
              `stop_id` VARCHAR(10) NOT NULL,
              `service_day` VARCHAR(20) NOT NULL,
              PRIMARY KEY (`trip_id`, `trip_headsign`, `stop_id`, `service_day`));"""
                
        engine.execute(stm)
        
        ## ---------------------- Table3-1: trip_stops ---------------------- ##
        # 0807 revised: Cause the table is too big, change table structure.
        
        stm = """DROP TABLE IF EXISTS `DBus`.`trip_stops`;"""
        engine.execute(stm)
        
        stm = """CREATE TABLE IF NOT EXISTS `DBus`.`trip_stops` (
              `trip_id` VARCHAR(30) NOT NULL,
              `stop_id` VARCHAR(10) NOT NULL,
              PRIMARY KEY (`trip_id`, `stop_id`));"""
        
        engine.execute(stm)    
        
        ## ---------------------- Table4: routes_ssid_array ---------------------- ##
        stm = """DROP TABLE IF EXISTS `DBus`.`routes_ssid_array`;"""
        engine.execute(stm)
        
        stm = """CREATE TABLE IF NOT EXISTS `DBus`.`routes_ssid_array` (
              `trip_id` VARCHAR(30) NOT NULL,
              `stop_sequence` INT NOT NULL,
              `stop_id` VARCHAR(10) NOT NULL,
              `pass_stop_id` VARCHAR(1000) NULL,
              `pass_ssid` VARCHAR(1500) NULL,
              PRIMARY KEY (`trip_id`, `stop_sequence`));"""
                
        engine.execute(stm)
        
        ## ---------------------- Table5: routes_timetables ---------------------- ##
        stm = """DROP TABLE IF EXISTS `DBus`.`routes_timetables`;"""
        engine.execute(stm)
        
        stm = """CREATE TABLE IF NOT EXISTS `DBus`.`routes_timetables` (
              `trip_id` VARCHAR(30) NOT NULL,
              `departure_time` VARCHAR(45) NOT NULL,
              `service_day` VARCHAR(10) NOT NULL);""" 
                
        engine.execute(stm)
        
    except Exception as e:
        print("Error Type: ",  type(e))
        print("Error Details: ", e)

def insert_data():
    
    # Link to DB
    engine = connect_db('127.0.0.1', '3306', 'DBus', 'root', 'team1010')
    
    # Turn off logging
    logging.basicConfig()
    logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
    

    
    #Insert data to tables
    ## ---------------------- Table1: routes ---------------------- ##
    stm = ('DELETE FROM routes WHERE `route_short_name` is not null')
    engine.execute(stm)
    df = pd.read_csv('./DB_input_csv/table1_routes.csv')
    df.to_sql(con=engine, name= 'routes', index = False, if_exists='append')
    
    ## ---------------------- Table2: routes_stop ---------------------- ##
    stm = ('DELETE FROM routes_stops WHERE `route_short_name` is not null')
    engine.execute(stm)
    df = pd.read_csv('./DB_input_csv/table2_routes_stops.csv')
    df['stop_id'] = df['stop_id'].apply(lambda x: str(x).zfill(4))
    df.to_sql(con=engine, name= 'routes_stops', index = False, if_exists='append')
    
    ## ---------------------- Table3: routes_stops_service_days ---------------------- ##
    stm = ('DELETE FROM routes_stops_service_days WHERE `trip_id` is not null')
    engine.execute(stm)
    
    df = pd.read_csv('./DB_input_csv/table3_routes_stops_service_days.csv')
    df['stop_id'] = df['stop_id'].apply(lambda x: str(x).zfill(4))
    df.to_sql(con=engine, name= 'routes_stops_service_days', index = False, if_exists='append')

    ## ---------------------- Table3-1: trip_stops ---------------------- ##
    stm = ('DELETE FROM trip_stops WHERE `trip_id` is not null')
    engine.execute(stm)
    
    df = pd.read_csv('./DB_input_csv/table3-1_trip_stops.csv')
    df['stop_id'] = df['stop_id'].apply(lambda x: str(x).zfill(4))
    df.to_sql(con=engine, name= 'trip_stops', index = False, if_exists='append')
    
    ## ---------------------- Table4: routes_ssid_array ---------------------- ##
    stm = ('DELETE FROM routes_ssid_array WHERE `trip_id` is not null')
    engine.execute(stm)
    df = pd.read_csv('./DB_input_csv/table4_routes_ssid_array.csv')
    df['stop_id'] = df['stop_id'].apply(lambda x: str(x).zfill(4))
    df.to_sql(con=engine, name= 'routes_ssid_array', index = False, if_exists='append')
    
    ## ---------------------- Table5: routes_timetable ---------------------- ##
    stm = ('DELETE FROM routes_timetables WHERE `trip_id` is not null')
    engine.execute(stm)
    
    df = pd.read_csv('./DB_input_csv/table5_routes_timetable.csv')
    df.to_sql(con=engine, name= 'routes_timetables', index = False, if_exists='append')
    
    
create_tables()   
insert_data()
print('Done')