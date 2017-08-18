import pandas as pd
import numpy as np
import requests
import json
import datetime
import statsmodels as stm
import statsmodels.formula.api as sm
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.linear_model import LinearRegression as LinR
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.metrics.scorer import make_scorer
from scipy import stats
from scipy.stats import randint
from sklearn.externals import joblib
from flask import Flask, flash, render_template, request, abort
from sqlalchemy import create_engine
import ast
import logging
import time

logging.basicConfig()
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)


def connect_db(URI, PORT, DB, USER, password):
    ''' Function to connect to the database '''
    try:

        PASSWORD = password
        engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(USER, PASSWORD, URI, PORT, DB), echo=True)
        return engine
    except Exception as e:
        print("Error Type: ", type(e))
        print("Error Details: ", e)

# ==================================
# Get trip_id list
# ==================================
def get_trip_id(req):
    """ This function return possible list of trip_id and timetable,

    base on the request form from frontend. Input parameter is dict datatype
    which store the form information."""

    # engine = connect_db('team1010.cnmhll8wqxlt.us-west-2.rds.amazonaws.com', '3306', 'DBus', 'Team1010_User', 'DubBus_Team1010')
    engine = connect_db('127.0.0.1', '3306', 'DBus', 'root', 'team1010')

    sql = """
        SELECT DISTINCT rss.trip_id FROM
        DBus.routes_stops_service_days as rss, DBus.trip_stops as ts
        WHERE rss.trip_id = ts.trip_id AND
        rss.route_short_name = %s AND rss.trip_headsign = %s
        AND rss.stop_id = %s AND ts.stop_id = %s AND rss.service_day = %s;
        """
    rows = engine.execute(sql, req['route'], req['direction'], req['orig_stop_id'], req['dest_stop_id'],
                          req['day']).fetchall()

    engine.dispose()

    # Return list
    trip_id_list = [x[0] for x in rows]

    return trip_id_list


# ==================================
# Get timetable list
# ==================================
def get_timetable(trip_id, req):

    engine = connect_db('127.0.0.1', '3306', 'DBus', 'root', 'team1010')

    sql = """
        SELECT DISTINCT departure_time
        FROM routes_timetables
        WHERE trip_id = %s AND service_day = %s;
        """
    rows = engine.execute(sql, trip_id, req['day']).fetchall()
    timetable_list = [x[0] for x in rows]

    return timetable_list


# ==================================
# Get weather information
# ==================================
def get_weather_info(req):
    """Function return the rain and wind speed information of the date request."""

    CITYID = "2964574"
    WEATHER = "http://api.openweathermap.org/data/2.5/forecast"
    APIKEY = "89b3e577901486c8ad601fab00edd389"

    r = requests.get(WEATHER, params={"APPID": APIKEY, "id": CITYID})
    js = json.loads(r.text)

    for i in range(len(js['list']) - 1, 0, -1):
        date, time = js['list'][i]['dt_txt'].split(' ')
        time = datetime.datetime.strptime(time, "%H:%M:%S")
        req_time = datetime.datetime.strptime(req['time'], "%H:%M")

        wind_speed = 0.0
        rain = 0.0

        if date == req['date'] and time <= req_time:
            wind_speed = js['list'][i]['wind']['speed']
            if js['list'][i]['rain'] != {}:
                rain = js['list'][i]['rain']['3h']/3
            break

    return rain, wind_speed


# ====================================
# Get SSID and passed stops array list
# ====================================
def get_SSID_array(trip_id, req):
    """ Function return passed by SSID and stops array of origin stop and destinatioin stop. """

    engine = connect_db('127.0.0.1', '3306', 'DBus', 'root', 'team1010')

    sql = """
        SELECT pass_ssid FROM routes_ssid_array
        WHERE trip_id = %s AND stop_id = %s; """

    row1 = engine.execute(sql, trip_id, req['orig_stop_id']).fetchall()
    row2 = engine.execute(sql, trip_id, req['dest_stop_id']).fetchall()

    orig_ssid_array = ast.literal_eval(row1[0][0])
    dest_ssid_array = ast.literal_eval(row2[0][0])

    return orig_ssid_array, dest_ssid_array


# ==================================
# Get accumulate travel time
# ==================================
def get_predictive_travel_time(trip_id, req):
    """Function return three accumulate predictive time,
    (1) start - origin_stop (2) origin_stop - dest_stop (3) start_stop - dest_stop (4) travel time array"""

    # Get ssid array
    orig_ssid_array, dest_ssid_array = get_SSID_array(trip_id, req)

    # Get weather info
    rain, wind_speed = get_weather_info(req)

    # Prepare input parameter for model as dataframe
    in_df = pd.DataFrame({'WindSpeed': wind_speed, 'Rain': rain,
                          'Day': req['day'], 'HourFrame': int(req['time'].split(':')[0])}, index=[0])

    # Get predictive time of each SSID and sum up (unit: sec)
    # 0808 revised: return travling time of each segment  and arrival time of each stop
    sum_travel_time = 0
    depart_orig_travel_time = 0
    orig_dest_travel_time = 0
    travel_time_list = []
    flag = False

    for ssid in dest_ssid_array:

        # Prepare dataframe to predict
        feature = pd.read_csv('SSID_model_features.csv')
        feature['SSID'] = feature['SSID'].apply(lambda x: str(x).zfill(8))
        frame = feature[feature.SSID == ssid]
        frame.dropna(axis=1, how='all', inplace=True)
        frame.drop(['SSID'], axis=1, inplace=True)
        frame.reset_index(drop=True, inplace=True)
        frame.columns = frame.iloc[0]
        frame.drop(0, axis=0, inplace=True)

        # Change data type
        set_to_zero = []
        frame['Rain'] = frame['Rain'].astype('float32')
        frame['Rain'] = frame['WindSpeed'].astype('float32')
        frame['JPID_length'] = frame['JPID_length'].astype(str).astype(int)
        for c in frame.columns:
            if c.find('HF') != -1 or c.find('Day') != -1 or c.find('SchoolHoliday') != -1:
                frame[c] = frame[c].apply(lambda x: int(float(x)))

        # Set value for input dataframe
        frame.set_value(index=1, col='Rain', value=rain)
        frame.set_value(index=1, col='WindSpeed', value=wind_speed)
        frame.set_value(index=1, col='HF_' + req['time'].split(':')[0], value=1)
        frame.set_value(index=1, col='Day_' + req['day'], value=1)

        # Get pickle file
        model = joblib.load('./SSID_XXXX_model_pkls/' + ssid + '.pkl')

        travel_time = model.predict(frame)[0]
        sum_travel_time += travel_time

        if flag:
            travel_time_list.append(travel_time)

        if ssid == orig_ssid_array[-1]:
            flag = True
            depart_orig_travel_time = sum_travel_time

    orig_dest_travel_time = sum_travel_time - depart_orig_travel_time

    return depart_orig_travel_time, orig_dest_travel_time, sum_travel_time, travel_time_list



# ==================================
# Get predictive timetable
# ==================================
def get_predictive_timetable(req):
    """ Function return predictive timetable of origin and destination stop,

    which includes previous, recommand and the next timetable.
    Also, return travel time between origin and destination stops.(Revised) """

    # Get available trip_id
    trip_id_list = get_trip_id(req)
    all_orig_time_list = []

    # For each trip_id in list
    for trip_id in trip_id_list:

        # Get departure timetable list
        depart_times = get_timetable(trip_id, req)

        # Get predictive traveling time
        depart_orig_tt, orig_dest_tt, sum_tt, tt_list = get_predictive_travel_time(trip_id, req)

        # Request time: convert to "datetime"
        req_t = datetime.datetime.strptime(req['time'], "%H:%M")

        # Get ideal depart time = request time - predictive travel time
        ideal_depart_t = (req_t - datetime.timedelta(seconds=sum_tt)).time()

        # Store actual depart time base on timetales
        actual_depart_t = ""

        # Get recommend schedule time of previous, recommand and next
        for i in range(len(depart_times)):

            # Depart time: convert to "datetime.time"
            depart_t = datetime.datetime.strptime(depart_times[i], "%H:%M:%S").time()
            diff = datetime.datetime.strptime(depart_times[i], "%H:%M:%S") - (
            req_t - datetime.timedelta(seconds=sum_tt))

            # First time that is larger than idea time but difference cannot over 4 hours
            if (depart_t >= ideal_depart_t):
                # Consider first and last bus of that day.
                actual_depart_t = depart_times[i]
                break

        if actual_depart_t != "":
            at = (
            datetime.datetime.strptime(actual_depart_t, "%H:%M:%S") + datetime.timedelta(seconds=depart_orig_tt)).time()
            all_orig_time_list.append(at.strftime("%H:%M:%S"))

        # Sort by time and pick the cloest time
        all_orig_time_list.sort()
        predictive_timetable_orig = all_orig_time_list[0]

        # Get all arrival time of stops passed by
        predictive_timetable_all = []
        acc_tt = predictive_timetable_orig
        predictive_timetable_all.append(acc_tt)

        for tt in tt_list:
            temp = (datetime.datetime.strptime(acc_tt, "%H:%M:%S") + datetime.timedelta(seconds=tt)).time()
            acc_tt = temp.strftime("%H:%M:%S")
            predictive_timetable_all.append(acc_tt)

    return predictive_timetable_all, orig_dest_tt


logging.basicConfig()
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)


tp1 = time.time()

trip_id = '0-46A-y12-1.322.O'
req = {'route': '046a', 'direction': "Phoenix Pk Gate - Queen's Road",
       'orig_stop_id': '0808', 'dest_stop_id': '0811', 'day':'Monday', 'time':'11:00', 'date': '2017-08-08'}

answer = get_predictive_timetable(req)
print("Time", int(time.time() - tp1))
print(answer)
